"""
Training Pipeline for Heterogeneous GNN.

Trains HeteroGNN on PyG HeteroData with gBCE loss, negative sampling,
early stopping, and ranking metrics evaluation.
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
from .hetero_gnn import HeteroGNN
from .ltgnn import gBCELoss, BPRLoss

logger = logging.getLogger(__name__)


@dataclass
class HeteroGNNMetrics:
    """Container for heterogeneous GNN training metrics."""
    loss: float
    auc: float = 0.0
    precision_at_10: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_10: float = 0.0


class HeteroGNNTrainer:
    """
    Trainer for HeteroGNN on PyG HeteroData.

    Handles:
      - Mini-batch training with negative sampling
      - gBCE loss for calibrated predictions
      - Per-epoch validation with ranking metrics
      - Checkpoint management and early stopping
    """

    def __init__(
        self,
        model: HeteroGNN,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        train_user_venue_edges: torch.Tensor,
        val_user_venue_edges: torch.Tensor,
        num_venues: int,
        config: Optional[GNNConfig] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
        use_gbce: bool = True,
        gbce_t: float = 0.8,
        num_negatives: int = 4,
    ):
        """
        Initialize the heterogeneous GNN trainer.

        Args:
            model: HeteroGNN model.
            x_dict: Node features per type.
            edge_index_dict: All edge indices per type (for message passing).
            train_user_venue_edges: Training user-venue edges [2, N_train].
            val_user_venue_edges: Validation user-venue edges [2, N_val].
            num_venues: Total number of venue nodes.
            config: GNN configuration.
            device: Training device.
            checkpoint_dir: Directory for saving checkpoints.
            use_gbce: Whether to use gBCE loss.
            gbce_t: gBCE temperature parameter.
            num_negatives: Number of negative samples per positive.
        """
        self.model = model
        self.config = config or get_config().gnn
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR / "gnn_hetero"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.num_negatives = num_negatives
        self.num_venues = num_venues

        # Move data to device
        self.model.to(self.device)
        self.x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
        self.edge_index_dict = {k: v.to(self.device) for k, v in edge_index_dict.items()}
        self.train_edges = train_user_venue_edges.to(self.device)
        self.val_edges = val_user_venue_edges.to(self.device)

        # Build positive set for negative sampling
        self.user_positives: dict[int, set] = {}
        train_cpu = train_user_venue_edges.cpu().numpy()
        for i in range(train_cpu.shape[1]):
            u, v = int(train_cpu[0, i]), int(train_cpu[1, i])
            if u not in self.user_positives:
                self.user_positives[u] = set()
            self.user_positives[u].add(v)

        # Loss
        if use_gbce:
            self.criterion = gBCELoss(t=gbce_t)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6,
        )

        # Early stopping on NDCG@10
        self.early_stopping = EarlyStopping(patience=10, mode="max")

        # State
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_history: list[HeteroGNNMetrics] = []
        self.val_history: list[HeteroGNNMetrics] = []

    def _sample_negatives(self, users: np.ndarray) -> np.ndarray:
        """Sample negative venue indices for each user."""
        negatives = []
        for user in users:
            positives = self.user_positives.get(int(user), set())
            user_negs = []
            attempts = 0
            while len(user_negs) < self.num_negatives and attempts < 100:
                neg = np.random.randint(0, self.num_venues)
                if neg not in positives:
                    user_negs.append(neg)
                attempts += 1
            # Pad if needed
            while len(user_negs) < self.num_negatives:
                user_negs.append(np.random.randint(0, self.num_venues))
            negatives.append(user_negs)
        return np.array(negatives)

    def train_epoch(self) -> HeteroGNNMetrics:
        """Train for one epoch with mini-batching over edges."""
        self.model.train()
        num_edges = self.train_edges.size(1)
        batch_size = self.config.batch_size

        perm = torch.randperm(num_edges)
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(
            range(0, num_edges, batch_size),
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
        )

        for start in progress:
            end = min(start + batch_size, num_edges)
            batch_idx = perm[start:end]

            pos_users = self.train_edges[0, batch_idx]
            pos_venues = self.train_edges[1, batch_idx]

            # Negative sampling
            neg_venues = self._sample_negatives(pos_users.cpu().numpy())
            neg_venues = torch.tensor(neg_venues, dtype=torch.long, device=self.device)

            self.optimizer.zero_grad()

            # Positive scores
            pos_scores = self.model(
                self.x_dict, self.edge_index_dict,
                pos_users, pos_venues,
            )

            # Negative scores (average over negative samples)
            neg_scores_list = []
            for j in range(self.num_negatives):
                ns = self.model(
                    self.x_dict, self.edge_index_dict,
                    pos_users, neg_venues[:, j],
                )
                neg_scores_list.append(ns)
            neg_scores = torch.stack(neg_scores_list, dim=1)

            # gBCE loss on positive and negative predictions
            pos_loss = self.criterion(
                torch.sigmoid(pos_scores),
                torch.ones_like(pos_scores),
            )
            neg_loss = self.criterion(
                torch.sigmoid(neg_scores),
                torch.zeros_like(neg_scores),
            )
            loss = pos_loss + neg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return HeteroGNNMetrics(loss=avg_loss)

    @torch.no_grad()
    def evaluate(
        self,
        edge_index: Optional[torch.Tensor] = None,
        k: int = 10,
        num_eval_users: int = 100,
    ) -> HeteroGNNMetrics:
        """
        Evaluate with AUC and ranking metrics.

        Args:
            edge_index: Edges to evaluate on (default: validation edges).
            k: K for ranking metrics.
            num_eval_users: Number of users to sample for ranking eval.
        """
        self.model.eval()
        edge_index = edge_index if edge_index is not None else self.val_edges

        # --- AUC ---
        num_eval = min(1000, edge_index.size(1))
        sample_perm = torch.randperm(edge_index.size(1))[:num_eval]
        eval_edges = edge_index[:, sample_perm]

        users = eval_edges[0]
        venues = eval_edges[1]

        pos_scores = self.model.predict(
            self.x_dict, self.edge_index_dict, users, venues
        )
        neg_venues = torch.randint(
            0, self.num_venues, (num_eval,), device=self.device
        )
        neg_scores = self.model.predict(
            self.x_dict, self.edge_index_dict, users, neg_venues
        )

        labels = torch.cat([
            torch.ones(num_eval, device=self.device),
            torch.zeros(num_eval, device=self.device),
        ])
        scores = torch.cat([pos_scores, neg_scores])

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        num_pos = sorted_labels.sum()
        num_neg = len(sorted_labels) - num_pos
        rank_sum = (
            sorted_labels
            * torch.arange(1, len(sorted_labels) + 1, device=self.device, dtype=torch.float)
        ).sum()
        auc = (rank_sum - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg + 1e-8)

        # --- Ranking metrics ---
        ranking_metrics = self._compute_ranking_metrics(
            edge_index, k=k, num_users=num_eval_users
        )

        return HeteroGNNMetrics(
            loss=0.0,
            auc=auc.item(),
            precision_at_10=ranking_metrics.get(f"precision@{k}", 0.0),
            recall_at_10=ranking_metrics.get(f"recall@{k}", 0.0),
            ndcg_at_10=ranking_metrics.get(f"ndcg@{k}", 0.0),
        )

    def _compute_ranking_metrics(
        self,
        edge_index: torch.Tensor,
        k: int = 10,
        num_users: int = 100,
    ) -> Dict[str, float]:
        """Compute ranking metrics for sampled users."""
        unique_users = torch.unique(edge_index[0])
        if len(unique_users) > num_users:
            sample_idx = torch.randperm(len(unique_users))[:num_users]
            sample_users = unique_users[sample_idx]
        else:
            sample_users = unique_users

        # Build ground truth
        edge_cpu = edge_index.cpu()
        user_gt: dict[int, set] = {}
        for i in range(edge_cpu.size(1)):
            u = edge_cpu[0, i].item()
            v = edge_cpu[1, i].item()
            if u not in user_gt:
                user_gt[u] = set()
            user_gt[u].add(v)

        calc = RecommendationMetrics()
        all_metrics = []

        for uid in sample_users.tolist():
            if uid not in user_gt:
                continue

            gt = user_gt[uid]
            rec_indices, _ = self.model.recommend(
                self.x_dict, self.edge_index_dict,
                user_idx=uid, k=k * 2,
            )
            recs = rec_indices.cpu().tolist()

            all_metrics.append({
                f"precision@{k}": calc.precision_at_k(recs, gt, k),
                f"recall@{k}": calc.recall_at_k(recs, gt, k),
                f"ndcg@{k}": calc.ndcg_at_k(recs, gt, k),
            })

        if not all_metrics:
            return {}

        return {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}

    def train(self) -> Dict:
        """Run the full training loop."""
        logger.info(f"Starting HeteroGNN training on {self.device}")
        logger.info(f"Train edges: {self.train_edges.size(1)}")
        logger.info(f"Val edges: {self.val_edges.size(1)}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            val_metrics = self.evaluate()
            self.val_history.append(val_metrics)

            self.scheduler.step()

            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Loss: {train_metrics.loss:.4f} | "
                f"AUC: {val_metrics.auc:.4f} | "
                f"NDCG@10: {val_metrics.ndcg_at_10:.4f}"
            )

            is_best = val_metrics.ndcg_at_10 > self.best_metric
            if is_best:
                self.best_metric = val_metrics.ndcg_at_10
            self._save_checkpoint(is_best=is_best)

            if self.early_stopping(val_metrics.ndcg_at_10):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return {
            "train_history": [vars(m) for m in self.train_history],
            "val_history": [vars(m) for m in self.val_history],
            "best_ndcg": self.best_metric,
        }

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save training checkpoint."""
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
        }
        filepath = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt"
        save_checkpoint(state, filepath, is_best=is_best)

    @torch.no_grad()
    def get_embeddings(self) -> dict[str, torch.Tensor]:
        """Extract learned embeddings for all node types."""
        self.model.eval()
        return self.model.encode(self.x_dict, self.edge_index_dict)
