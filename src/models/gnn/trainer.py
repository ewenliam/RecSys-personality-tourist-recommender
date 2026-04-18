"""
Training Pipeline for GNN Models.

Key optimization: encode all nodes ONCE per epoch using sparse convolution,
then train the link predictor on mini-batches using cached embeddings.
This reduces per-epoch cost from ~3460 full-graph passes to 1.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config.settings import GNNConfig, get_config, CHECKPOINT_DIR
from src.utils.helpers import save_checkpoint, EarlyStopping
from src.utils.metrics import RecommendationMetrics
from .ltgnn import LTGNN, gBCELoss, BPRLoss
from .evr_sampler import MiniBatchLoader, TrainTestSplit

logger = logging.getLogger(__name__)


@dataclass
class GNNTrainingMetrics:
    """Container for GNN training metrics."""
    loss: float
    auc: float = 0.0
    precision_at_10: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_10: float = 0.0


class GNNTrainer:
    """
    Trainer for GNN recommendation models.

    Training strategy (per epoch):
    1. Encode all nodes ONCE with no_grad (sparse conv, ~1 second)
    2. Train link predictor on mini-batches using cached embeddings (~30 seconds)
    3. One full forward+backward through entire model to update encoder (~2 seconds)

    This replaces the naive approach of encoding the full 321K-node graph
    5 times per mini-batch (3,460 encodes/epoch -> 2 encodes/epoch).
    """

    def __init__(
        self,
        model: LTGNN,
        user_features: torch.Tensor,
        venue_features: torch.Tensor,
        train_edge_index: torch.Tensor,
        val_edge_index: torch.Tensor,
        config: Optional[GNNConfig] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
        use_gbce: bool = True,
        gbce_t: float = 0.8,
    ):
        self.model = model
        self.user_features = user_features
        self.venue_features = venue_features
        self.train_edge_index = train_edge_index
        self.val_edge_index = val_edge_index
        self.config = config or get_config().gnn
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR / "gnn"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Validate and move to device
        self.model.to(self.device)
        self.user_features = self.user_features.to(self.device)
        self.venue_features = self.venue_features.to(self.device)

        num_users = self.user_features.size(0)
        num_venues = self.venue_features.size(0)

        # Filter edges that reference non-existent nodes BEFORE moving to GPU
        self.train_edge_index = self._validate_edges(
            train_edge_index, num_users, num_venues, "train"
        ).to(self.device)
        self.val_edge_index = self._validate_edges(
            val_edge_index, num_users, num_venues, "val"
        ).to(self.device)

        # Loss
        if use_gbce:
            self.criterion = gBCELoss(t=gbce_t)
        else:
            self.criterion = nn.BCELoss()

        self.bpr_loss = BPRLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=5, mode="max")

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_history = []
        self.val_history = []

        # Mini-batch loader
        self.train_loader = MiniBatchLoader(
            edge_index=self.train_edge_index.cpu(),
            num_users=self.user_features.size(0),
            num_venues=self.venue_features.size(0),
            batch_size=self.config.batch_size,
            num_negatives=4,
            shuffle=True,
        )

    @staticmethod
    def _validate_edges(
        edge_index: torch.Tensor,
        num_users: int,
        num_venues: int,
        label: str = "",
    ) -> torch.Tensor:
        """Filter edges with out-of-bounds node indices.

        Ensures edge_index[0] < num_users and edge_index[1] < num_venues.
        Logs a warning if any edges are dropped.
        """
        if edge_index.numel() == 0:
            return edge_index

        valid = (
            (edge_index[0] >= 0) & (edge_index[0] < num_users) &
            (edge_index[1] >= 0) & (edge_index[1] < num_venues)
        )
        n_valid = valid.sum().item()
        n_total = edge_index.size(1)

        if n_valid < n_total:
            max_u = edge_index[0].max().item()
            max_v = edge_index[1].max().item()
            logger.warning(
                f"{label} edges: {n_total - n_valid}/{n_total} dropped "
                f"(max_user={max_u} vs num_users={num_users}, "
                f"max_venue={max_v} vs num_venues={num_venues})"
            )
            edge_index = edge_index[:, valid]

        logger.info(f"{label} edges validated: {edge_index.size(1)} edges OK")
        return edge_index

    def train_epoch(self) -> GNNTrainingMetrics:
        """
        Train for one epoch using cached embeddings.

        Phase 1: Encode all nodes once (no_grad, sparse conv).
        Phase 2: Train link predictor on mini-batches.
        Phase 3: One full forward+backward to update GNN encoder.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # -- Phase 1: Encode all nodes ONCE (cheap with sparse conv) --
        with torch.no_grad():
            user_emb, venue_emb = self.model.encode(
                self.user_features,
                self.venue_features,
                self.train_edge_index,
            )
        # Detach so backward only flows through link_predictor
        user_emb = user_emb.detach()
        venue_emb = venue_emb.detach()

        # -- Phase 2: Train link predictor on mini-batches --
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
        )

        for batch in progress_bar:
            self.optimizer.zero_grad()

            pos_users = batch["pos_users"].to(self.device)
            pos_venues = batch["pos_venues"].to(self.device)
            neg_venues = batch["neg_venues"].to(self.device)

            # Look up cached embeddings (instant, no graph computation)
            u_emb = user_emb[pos_users]
            pos_v_emb = venue_emb[pos_venues]

            # Positive scores
            pos_combined = torch.cat([u_emb, pos_v_emb], dim=1)
            pos_scores = self.model.link_predictor(pos_combined).squeeze(-1)

            # Negative scores
            neg_scores_list = []
            for i in range(neg_venues.size(1)):
                neg_v_emb = venue_emb[neg_venues[:, i]]
                neg_combined = torch.cat([u_emb, neg_v_emb], dim=1)
                neg_scores_list.append(
                    self.model.link_predictor(neg_combined).squeeze(-1)
                )
            neg_scores = torch.stack(neg_scores_list, dim=1)

            # Loss
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)
            pos_loss = self.criterion(torch.sigmoid(pos_scores), pos_labels)
            neg_loss = self.criterion(torch.sigmoid(neg_scores), neg_labels)
            loss = pos_loss + neg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # -- Phase 3: Update GNN encoder weights --
        # One full forward+backward so projections and GNN weights get gradients
        self._update_encoder()

        avg_loss = total_loss / max(num_batches, 1)
        return GNNTrainingMetrics(loss=avg_loss)

    def _update_encoder(self, max_sub_edges: int = 100_000):
        """
        Update GNN encoder weights via subgraph sampling.

        Instead of encoding ALL 321K nodes with gradients (which OOMs
        on 8GB VRAM), we:
        1. Sample a batch of training edges for loss computation.
        2. Expand to include 1-hop neighbors (for message-passing context).
        3. Cap the subgraph to max_sub_edges to bound memory.
        4. Remap indices to a compact local space.
        5. Forward+backward on the subgraph only.

        This gives the encoder meaningful gradients while keeping peak
        GPU memory under ~2GB for the gradient pass.
        """
        self.optimizer.zero_grad()

        num_users = self.user_features.size(0)
        num_venues = self.venue_features.size(0)
        num_edges = self.train_edge_index.size(1)

        # 1. Sample edges for loss computation
        sample_size = min(4096, num_edges)
        perm = torch.randperm(num_edges, device=self.device)[:sample_size]

        # 2. Find all unique users/venues in the sample
        sampled_users = torch.unique(self.train_edge_index[0, perm])
        sampled_venues = torch.unique(self.train_edge_index[1, perm])

        # 3. Expand: include ALL edges touching sampled users or venues
        #    (gives the GNN proper 1-hop neighborhood context)
        u_flag = torch.zeros(num_users, dtype=torch.bool, device=self.device)
        u_flag[sampled_users] = True
        v_flag = torch.zeros(num_venues, dtype=torch.bool, device=self.device)
        v_flag[sampled_venues] = True

        neighbor_mask = (
            u_flag[self.train_edge_index[0]] |
            v_flag[self.train_edge_index[1]]
        )
        sub_edge_indices = neighbor_mask.nonzero(as_tuple=True)[0]

        # 4. Cap subgraph size (always keep the loss-batch edges)
        if sub_edge_indices.size(0) > max_sub_edges:
            keep = torch.randperm(
                sub_edge_indices.size(0), device=self.device
            )[:max_sub_edges]
            sub_edge_indices = torch.unique(
                torch.cat([perm, sub_edge_indices[keep]])
            )

        sub_edges = self.train_edge_index[:, sub_edge_indices]

        # 5. Get compact set of nodes in the subgraph
        sub_users = torch.unique(sub_edges[0])
        sub_venues = torch.unique(sub_edges[1])

        # 6. Build local index remapping (global -> local)
        user_remap = torch.full(
            (num_users,), -1, dtype=torch.long, device=self.device
        )
        user_remap[sub_users] = torch.arange(
            sub_users.size(0), device=self.device
        )
        venue_remap = torch.full(
            (num_venues,), -1, dtype=torch.long, device=self.device
        )
        venue_remap[sub_venues] = torch.arange(
            sub_venues.size(0), device=self.device
        )

        local_edges = torch.stack([
            user_remap[sub_edges[0]],
            venue_remap[sub_edges[1]],
        ])

        # 7. Extract subgraph features
        sub_user_x = self.user_features[sub_users]
        sub_venue_x = self.venue_features[sub_venues]

        # 8. Invalidate full-graph adj cache (subgraph has different topology)
        saved_adj = self.model._cached_adj
        saved_edge_id = self.model._cached_edge_id
        self.model._cached_adj = None
        self.model._cached_edge_id = None

        try:
            # 9. Forward pass on SUBGRAPH with gradients
            user_emb, venue_emb = self.model.encode(
                sub_user_x, sub_venue_x, local_edges,
            )

            # 10. Compute loss on original batch edges (in local coords)
            loss_users = self.train_edge_index[0, perm]
            loss_venues = self.train_edge_index[1, perm]
            local_u = user_remap[loss_users]
            local_v = venue_remap[loss_venues]

            # Positive scores
            u = user_emb[local_u]
            pos_v = venue_emb[local_v]
            pos_combined = torch.cat([u, pos_v], dim=1)
            pos_scores = self.model.link_predictor(pos_combined).squeeze(-1)

            # Negative scores (from subgraph venues for valid indexing)
            neg_idx = torch.randint(
                0, venue_emb.size(0), (sample_size,), device=self.device
            )
            neg_v = venue_emb[neg_idx]
            neg_combined = torch.cat([u, neg_v], dim=1)
            neg_scores = self.model.link_predictor(neg_combined).squeeze(-1)

            loss = (
                self.criterion(
                    torch.sigmoid(pos_scores), torch.ones_like(pos_scores)
                )
                + self.criterion(
                    torch.sigmoid(neg_scores), torch.zeros_like(neg_scores)
                )
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        finally:
            # 11. Always invalidate subgraph adj so next encode() rebuilds
            self.model._cached_adj = None
            self.model._cached_edge_id = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self, edge_index: torch.Tensor = None) -> GNNTrainingMetrics:
        """Evaluate the model using cached embeddings."""
        self.model.eval()

        if edge_index is None:
            edge_index = self.val_edge_index

        # Encode ALL nodes once for evaluation (memory-safe path)
        user_emb, venue_emb = self.model.encode_inference(
            self.user_features,
            self.venue_features,
            self.train_edge_index,
        )
        # Move to device for metric computation
        user_emb = user_emb.to(self.device)
        venue_emb = venue_emb.to(self.device)

        # Sample edges for AUC
        num_eval = min(1000, edge_index.size(1))
        perm = torch.randperm(edge_index.size(1))[:num_eval]
        eval_edges = edge_index[:, perm]

        users = eval_edges[0]
        venues = eval_edges[1]

        # Positive scores from cached embeddings
        pos_scores = self.model.predict_from_embeddings(
            user_emb, venue_emb, users, venues
        )
        pos_scores = torch.sigmoid(pos_scores)

        # Negative scores
        neg_venues = torch.randint(
            0, self.venue_features.size(0),
            (num_eval,), device=self.device
        )
        neg_scores = self.model.predict_from_embeddings(
            user_emb, venue_emb, users, neg_venues
        )
        neg_scores = torch.sigmoid(neg_scores)

        # AUC
        labels = torch.cat([
            torch.ones(num_eval, device=self.device),
            torch.zeros(num_eval, device=self.device),
        ])
        scores = torch.cat([pos_scores, neg_scores])

        sorted_indices = torch.argsort(scores)
        sorted_labels = labels[sorted_indices]
        num_pos = sorted_labels.sum()
        num_neg = len(sorted_labels) - num_pos
        rank_sum = (
            sorted_labels
            * torch.arange(1, len(sorted_labels) + 1, device=self.device)
        ).sum()
        auc = (rank_sum - num_pos * (num_pos + 1) / 2) / (
            num_pos * num_neg + 1e-8
        )

        # Ranking metrics (pass cached embeddings to avoid re-encoding)
        metrics = self._compute_ranking_metrics(
            edge_index, cached_embeddings=(user_emb, venue_emb)
        )

        return GNNTrainingMetrics(
            loss=0.0,
            auc=auc.item(),
            precision_at_10=metrics.get("precision@10", 0.0),
            recall_at_10=metrics.get("recall@10", 0.0),
            ndcg_at_10=metrics.get("ndcg@10", 0.0),
        )

    def _compute_ranking_metrics(
        self,
        edge_index: torch.Tensor,
        k: int = 10,
        num_users: int = 100,
        cached_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Compute ranking metrics for a sample of users."""
        unique_users = torch.unique(edge_index[0])
        if len(unique_users) > num_users:
            sample_idx = torch.randperm(len(unique_users))[:num_users]
            sample_users = unique_users[sample_idx]
        else:
            sample_users = unique_users

        # Ground truth
        edge_cpu = edge_index.cpu()
        user_ground_truth = {}
        for i in range(edge_cpu.size(1)):
            user = edge_cpu[0, i].item()
            venue = edge_cpu[1, i].item()
            if user not in user_ground_truth:
                user_ground_truth[user] = set()
            user_ground_truth[user].add(venue)

        # Recommendations
        metrics_calc = RecommendationMetrics()
        all_metrics = []

        for user_idx in sample_users.tolist():
            if user_idx not in user_ground_truth:
                continue

            ground_truth = user_ground_truth[user_idx]

            rec_indices, rec_scores = self.model.recommend(
                self.user_features,
                self.venue_features,
                self.train_edge_index,
                user_idx,
                k=k * 2,
                cached_embeddings=cached_embeddings,
            )

            recommendations = rec_indices.cpu().tolist()

            user_metrics = {
                f"precision@{k}": metrics_calc.precision_at_k(
                    recommendations, ground_truth, k
                ),
                f"recall@{k}": metrics_calc.recall_at_k(
                    recommendations, ground_truth, k
                ),
                f"ndcg@{k}": metrics_calc.ndcg_at_k(
                    recommendations, ground_truth, k
                ),
            }
            all_metrics.append(user_metrics)

        if not all_metrics:
            return {}

        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics

    def train(self) -> Dict:
        """Run the full training loop."""
        logger.info(f"Starting GNN training on {self.device}")
        logger.info(f"Train edges: {self.train_edge_index.size(1)}")
        logger.info(f"Val edges: {self.val_edge_index.size(1)}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            val_metrics = self.evaluate()
            self.val_history.append(val_metrics)

            self.scheduler.step()

            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_metrics.loss:.4f} | "
                f"Val AUC: {val_metrics.auc:.4f} | "
                f"Val NDCG@10: {val_metrics.ndcg_at_10:.4f}"
            )

            is_best = val_metrics.ndcg_at_10 > self.best_metric
            if is_best:
                self.best_metric = val_metrics.ndcg_at_10

            self.save_checkpoint(is_best=is_best)

            if self.early_stopping(val_metrics.ndcg_at_10):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return {
            "train_history": [vars(m) for m in self.train_history],
            "val_history": [vars(m) for m in self.val_history],
            "best_metric": self.best_metric,
        }

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save a training checkpoint."""
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }

        filepath = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt"
        save_checkpoint(state, filepath, is_best=is_best)

    def load_checkpoint(self, filepath: Path) -> None:
        """Load a training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch + 1}")

    @torch.no_grad()
    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get user and venue embeddings (memory-safe).

        Returns tensors on CPU to avoid holding large GPU allocations.
        Uses chunked projection so peak VRAM stays low even for 321K+ nodes.
        """
        self.model.eval()

        user_emb, venue_emb = self.model.encode_inference(
            self.user_features,
            self.venue_features,
            self.train_edge_index,
        )

        num_users = self.user_features.size(0)
        num_venues = self.venue_features.size(0)

        if user_emb.size(0) != num_users:
            raise RuntimeError(
                f"User embedding count mismatch: got {user_emb.size(0)}, "
                f"expected {num_users}"
            )
        if venue_emb.size(0) != num_venues:
            raise RuntimeError(
                f"Venue embedding count mismatch: got {venue_emb.size(0)}, "
                f"expected {num_venues}"
            )

        logger.info(
            f"Embeddings extracted: users {tuple(user_emb.shape)}, "
            f"venues {tuple(venue_emb.shape)}"
        )

        return user_emb, venue_emb

    @torch.no_grad()
    def recommend_for_user(
        self,
        user_idx: int,
        k: int = 10,
        exclude_visited: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """Get recommendations for a specific user."""
        self.model.eval()

        exclude = set()
        if exclude_visited:
            edge_cpu = self.train_edge_index.cpu()
            for i in range(edge_cpu.size(1)):
                if edge_cpu[0, i].item() == user_idx:
                    exclude.add(edge_cpu[1, i].item())

        indices, scores = self.model.recommend(
            self.user_features,
            self.venue_features,
            self.train_edge_index,
            user_idx,
            k=k,
            exclude_venues=exclude,
        )

        return indices.cpu().tolist(), scores.cpu().tolist()


def train_gnn(
    graph_path: Path,
    config: Optional[GNNConfig] = None,
    device: str = "cuda",
    num_epochs: int = 15,
    use_gbce: bool = True,
) -> Tuple[LTGNN, Dict]:
    """
    Train GNN model on a saved graph.

    Args:
        graph_path: Path to saved graph.
        config: GNN configuration.
        device: Device for training.
        num_epochs: Number of epochs.
        use_gbce: Use gBCE loss.

    Returns:
        Tuple of (trained model, training history).
    """
    from .graph_builder import HeteroGraphBuilder

    builder = HeteroGraphBuilder.load(graph_path, device=torch.device(device))

    user_features = builder.nodes["user"].features
    venue_features = builder.nodes["venue"].features

    edge_index = builder.edges[("user", "visits", "venue")].edge_index
    splitter = TrainTestSplit(edge_index, train_ratio=0.8, val_ratio=0.1)

    train_edges = splitter.get_train_edges()
    val_edges = splitter.get_val_edges()

    config = config or get_config().gnn
    config.num_epochs = num_epochs

    model = LTGNN(
        user_input_dim=user_features.size(1),
        venue_input_dim=venue_features.size(1),
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_iterations=config.fixed_point_iterations,
        dropout=config.dropout,
    )

    trainer = GNNTrainer(
        model=model,
        user_features=user_features,
        venue_features=venue_features,
        train_edge_index=train_edges,
        val_edge_index=val_edges,
        config=config,
        device=torch.device(device),
        use_gbce=use_gbce,
    )

    history = trainer.train()

    return model, history
