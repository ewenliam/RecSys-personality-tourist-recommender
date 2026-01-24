"""
Training Pipeline for GNN Models.

Includes training loop, evaluation, and checkpoint management.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

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
    """Trainer for GNN recommendation models."""

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
        """
        Initialize the trainer.

        Args:
            model: GNN model to train.
            user_features: User node features.
            venue_features: Venue node features.
            train_edge_index: Training edges.
            val_edge_index: Validation edges.
            config: GNN configuration.
            device: Device for training.
            checkpoint_dir: Directory for checkpoints.
            use_gbce: Use gBCE loss for calibration.
            gbce_t: gBCE temperature parameter.
        """
        self.model = model
        self.user_features = user_features
        self.venue_features = venue_features
        self.train_edge_index = train_edge_index
        self.val_edge_index = val_edge_index
        self.config = config or get_config().gnn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR / "gnn"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move to device
        self.model.to(self.device)
        self.user_features = self.user_features.to(self.device)
        self.venue_features = self.venue_features.to(self.device)
        self.train_edge_index = self.train_edge_index.to(self.device)
        self.val_edge_index = self.val_edge_index.to(self.device)

        # Setup loss
        if use_gbce:
            self.criterion = gBCELoss(t=gbce_t)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.bpr_loss = BPRLoss()

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=10, mode="max")

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_history = []
        self.val_history = []

        # Create data loaders
        self.train_loader = MiniBatchLoader(
            edge_index=self.train_edge_index.cpu(),
            num_users=self.user_features.size(0),
            num_venues=self.venue_features.size(0),
            batch_size=self.config.batch_size,
            num_negatives=4,
            shuffle=True,
        )

    def train_epoch(self) -> GNNTrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
        )

        for batch in progress_bar:
            self.optimizer.zero_grad()

            # Get batch data
            pos_users = batch["pos_users"].to(self.device)
            pos_venues = batch["pos_venues"].to(self.device)
            neg_venues = batch["neg_venues"].to(self.device)

            # Positive scores
            pos_scores = self.model(
                self.user_features,
                self.venue_features,
                self.train_edge_index,
                pos_users,
                pos_venues,
            )

            # Negative scores (for each negative sample)
            neg_scores_list = []
            for i in range(neg_venues.size(1)):
                neg_score = self.model(
                    self.user_features,
                    self.venue_features,
                    self.train_edge_index,
                    pos_users,
                    neg_venues[:, i],
                )
                neg_scores_list.append(neg_score)

            neg_scores = torch.stack(neg_scores_list, dim=1)

            # Compute loss
            # Option 1: BCE loss with positives=1, negatives=0
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)

            pos_loss = self.criterion(torch.sigmoid(pos_scores), pos_labels)
            neg_loss = self.criterion(torch.sigmoid(neg_scores), neg_labels)
            loss = pos_loss + neg_loss

            # Option 2: BPR loss (commented out, can switch)
            # mean_neg_scores = neg_scores.mean(dim=1)
            # loss = self.bpr_loss(pos_scores, mean_neg_scores)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches

        return GNNTrainingMetrics(loss=avg_loss)

    @torch.no_grad()
    def evaluate(self, edge_index: torch.Tensor = None) -> GNNTrainingMetrics:
        """Evaluate the model."""
        self.model.eval()

        if edge_index is None:
            edge_index = self.val_edge_index

        # Sample some edges for evaluation
        num_eval = min(1000, edge_index.size(1))
        perm = torch.randperm(edge_index.size(1))[:num_eval]
        eval_edges = edge_index[:, perm]

        # Compute scores for positive pairs
        users = eval_edges[0]
        venues = eval_edges[1]

        pos_scores = self.model.predict(
            self.user_features,
            self.venue_features,
            self.train_edge_index,
            users,
            venues,
        )

        # Compute scores for random negatives
        neg_venues = torch.randint(
            0, self.venue_features.size(0),
            (num_eval,), device=self.device
        )

        neg_scores = self.model.predict(
            self.user_features,
            self.venue_features,
            self.train_edge_index,
            users,
            neg_venues,
        )

        # Compute AUC
        labels = torch.cat([
            torch.ones(num_eval, device=self.device),
            torch.zeros(num_eval, device=self.device),
        ])
        scores = torch.cat([pos_scores, neg_scores])

        # Simple AUC calculation
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        num_pos = sorted_labels.sum()
        num_neg = len(sorted_labels) - num_pos
        rank_sum = (sorted_labels * torch.arange(1, len(sorted_labels) + 1, device=self.device)).sum()
        auc = (rank_sum - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg + 1e-8)

        # Compute ranking metrics
        metrics = self._compute_ranking_metrics(edge_index)

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
    ) -> Dict[str, float]:
        """Compute ranking metrics for a sample of users."""
        # Sample users
        unique_users = torch.unique(edge_index[0])
        if len(unique_users) > num_users:
            sample_idx = torch.randperm(len(unique_users))[:num_users]
            sample_users = unique_users[sample_idx]
        else:
            sample_users = unique_users

        # Get ground truth for each user
        edge_cpu = edge_index.cpu()
        user_ground_truth = {}
        for i in range(edge_cpu.size(1)):
            user = edge_cpu[0, i].item()
            venue = edge_cpu[1, i].item()
            if user not in user_ground_truth:
                user_ground_truth[user] = set()
            user_ground_truth[user].add(venue)

        # Generate recommendations
        metrics_calc = RecommendationMetrics()
        all_metrics = []

        for user_idx in sample_users.tolist():
            if user_idx not in user_ground_truth:
                continue

            ground_truth = user_ground_truth[user_idx]

            # Get recommendations
            rec_indices, rec_scores = self.model.recommend(
                self.user_features,
                self.venue_features,
                self.train_edge_index,
                user_idx,
                k=k * 2,  # Get more to filter training items
            )

            recommendations = rec_indices.cpu().tolist()

            # Compute metrics
            user_metrics = {
                f"precision@{k}": metrics_calc.precision_at_k(recommendations, ground_truth, k),
                f"recall@{k}": metrics_calc.recall_at_k(recommendations, ground_truth, k),
                f"ndcg@{k}": metrics_calc.ndcg_at_k(recommendations, ground_truth, k),
            }
            all_metrics.append(user_metrics)

        # Average metrics
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

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.evaluate()
            self.val_history.append(val_metrics)

            # Step scheduler
            self.scheduler.step()

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_metrics.loss:.4f} | "
                f"Val AUC: {val_metrics.auc:.4f} | "
                f"Val NDCG@10: {val_metrics.ndcg_at_10:.4f}"
            )

            # Save checkpoint
            is_best = val_metrics.ndcg_at_10 > self.best_metric
            if is_best:
                self.best_metric = val_metrics.ndcg_at_10

            self.save_checkpoint(is_best=is_best)

            # Early stopping
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
        """Get user and venue embeddings."""
        self.model.eval()

        user_emb, venue_emb = self.model.encode(
            self.user_features,
            self.venue_features,
            self.train_edge_index,
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

        # Get visited venues to exclude
        exclude = set()
        if exclude_visited:
            edge_cpu = self.train_edge_index.cpu()
            for i in range(edge_cpu.size(1)):
                if edge_cpu[0, i].item() == user_idx:
                    exclude.add(edge_cpu[1, i].item())

        # Get recommendations
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
    num_epochs: int = 100,
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

    # Load graph
    builder = HeteroGraphBuilder.load(graph_path, device=torch.device(device))

    # Get features
    user_features = builder.nodes["user"].features
    venue_features = builder.nodes["venue"].features

    # Get edges and split
    edge_index = builder.edges[("user", "visits", "venue")].edge_index
    splitter = TrainTestSplit(edge_index, train_ratio=0.8, val_ratio=0.1)

    train_edges = splitter.get_train_edges()
    val_edges = splitter.get_val_edges()

    # Create model
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

    # Create trainer
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

    # Train
    history = trainer.train()

    return model, history
