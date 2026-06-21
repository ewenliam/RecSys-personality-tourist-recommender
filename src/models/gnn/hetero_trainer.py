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
        temperature: float = 0.1,
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
        # Temperature for cosine-similarity logits.  L2-normalized dot products
        # live in [-1, 1]; dividing by tau (<1) expands them to [-1/tau, 1/tau]
        # so BPR can push positives well above negatives.  Without this the
        # logit range is too compressed and the loss plateaus near 0.45 with
        # the model unable to separate positives from negatives.  Since this
        # scales ALL scores uniformly, it does not change the argsort ranking
        # used at inference - it only sharpens the training gradient.
        self.temperature = temperature

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

        # Popularity-weighted negative-sampling distribution.
        # Uniform random negatives are too easy: a random venue out of 85K is
        # trivially distinguishable from a visited one, so the model achieves
        # high AUC (pos vs random neg) while learning nothing useful for top-K
        # ranking, where it must beat the few hundred *plausible* venues.
        # Popular venues are the hard negatives - plausible for many users, so
        # the model must learn to recommend them only when they actually match,
        # rather than defaulting to popularity.  degree^0.75 smoothing (the
        # word2vec trick) tempers the head so the most popular venues do not
        # dominate every batch.
        venue_deg = torch.zeros(num_venues, dtype=torch.float)
        vids, counts = torch.unique(train_user_venue_edges[1], return_counts=True)
        venue_deg[vids.long()] = counts.float()
        self.neg_sample_weights = (venue_deg + 1.0).pow(0.75).to(self.device)

        # Loss
        # BPR directly optimizes the ranking: the model must score positive
        # (user, venue) pairs higher than negative pairs.  Combined with
        # L2-normalized dot-product scoring, this aligns training with the
        # cosine similarity used for retrieval at inference.
        self.bpr_loss = BPRLoss()
        # Keep gBCE as optional fallback (used only for AUC reporting)
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

    @torch.no_grad()
    def _encode_all_no_grad(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode all nodes without gradient (for evaluation metrics)."""
        self.model.eval()
        emb_dict = self.model.encode(self.x_dict, self.edge_index_dict)
        self.model.train()
        return emb_dict["user"], emb_dict["venue"]

    def train_epoch(self, num_grad_steps: int = 200) -> HeteroGNNMetrics:
        """
        Train for one epoch: multiple BPR steps, each with a full-graph forward.

        Because dot-product scoring has no separate link predictor weights,
        the encoder IS the scoring function and must receive real gradients.
        Caching (encode once, detach, then backward) would give zero gradient
        since the computation graph is broken at the detach point.

        Instead, we run num_grad_steps full-graph forward passes, each using
        a different random edge sample for BPR loss.  With 1 SAGEConv layer,
        each pass takes ~2-3 s on this GPU, so 200 steps ≈ 7 min/epoch.
        This is far cheaper than the old approach (1 full pass per mini-batch
        = 3,750 passes × 5 negatives = 18,750 passes per epoch).

        Args:
            num_grad_steps: Number of full-graph forward+backward passes per
                epoch.  Each step samples 4096 random edges for BPR loss.
        """
        self.model.train()
        num_edges = self.train_edges.size(1)
        step_size = min(4096, num_edges)

        total_loss = 0.0

        progress = tqdm(
            range(num_grad_steps),
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
        )

        for step in progress:
            self.optimizer.zero_grad()

            # Full-graph forward WITH gradient (encoder gets updated)
            emb_dict = self.model.encode(self.x_dict, self.edge_index_dict)
            user_emb = emb_dict["user"]
            venue_emb = emb_dict["venue"]

            # Sample a random edge batch for BPR loss
            perm = torch.randperm(num_edges, device=self.device)[:step_size]
            pos_users = self.train_edges[0, perm]
            pos_venues = self.train_edges[1, perm]

            # Mix of easy (uniform) and hard (popularity-weighted) negatives.
            # Easy negatives keep training stable; hard negatives (popular,
            # plausible venues) force the model to learn genuine collaborative
            # preference for top-K ranking.  Averaging BPR over several
            # negatives also lowers gradient variance.
            n_hard = self.num_negatives // 2
            n_easy = self.num_negatives - n_hard
            neg_hard = torch.multinomial(
                self.neg_sample_weights, step_size * n_hard, replacement=True
            ).view(step_size, n_hard)
            neg_easy = torch.randint(
                0, self.num_venues, (step_size, n_easy), device=self.device
            )
            neg_venues = torch.cat([neg_hard, neg_easy], dim=1)

            u = user_emb[pos_users]                       # [B, d]
            pos_v = venue_emb[pos_venues]                 # [B, d]
            neg_v = venue_emb[neg_venues]                 # [B, n_neg, d]

            # Temperature-scaled cosine similarity (embeddings L2-normalized).
            pos_scores = (u * pos_v).sum(dim=-1) / self.temperature        # [B]
            neg_scores = (u.unsqueeze(1) * neg_v).sum(dim=-1) / self.temperature  # [B, n_neg]

            # BPR over each negative, then average across negatives.
            loss = self.bpr_loss(
                pos_scores.unsqueeze(1).expand_as(neg_scores),
                neg_scores,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return HeteroGNNMetrics(loss=total_loss / max(num_grad_steps, 1))

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

        # Encode once for both AUC and ranking evaluation
        user_emb, venue_emb = self._encode_all_no_grad()

        # --- AUC (dot product scores via cached embeddings) ---
        num_eval = min(1000, edge_index.size(1))
        sample_perm = torch.randperm(edge_index.size(1))[:num_eval]
        eval_edges = edge_index[:, sample_perm]

        users = eval_edges[0]
        venues = eval_edges[1]

        pos_scores = torch.sigmoid(
            (user_emb[users] * venue_emb[venues]).sum(dim=-1)
        )
        neg_venues = torch.randint(
            0, self.num_venues, (num_eval,), device=self.device
        )
        neg_scores = torch.sigmoid(
            (user_emb[users] * venue_emb[neg_venues]).sum(dim=-1)
        )

        labels = torch.cat([
            torch.ones(num_eval, device=self.device),
            torch.zeros(num_eval, device=self.device),
        ])
        scores = torch.cat([pos_scores, neg_scores])

        # Wilcoxon-Mann-Whitney AUC: sort ascending (rank 1 = lowest score).
        # Positives that score higher than negatives end up at high rank positions,
        # giving rank_sum >> n_pos*(n_pos+1)/2 -> AUC close to 1.
        sorted_indices = torch.argsort(scores, descending=False)
        sorted_labels = labels[sorted_indices]
        num_pos = sorted_labels.sum()
        num_neg = len(sorted_labels) - num_pos
        rank_sum = (
            sorted_labels
            * torch.arange(1, len(sorted_labels) + 1, device=self.device, dtype=torch.float)
        ).sum()
        auc = (rank_sum - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg + 1e-8)

        # --- Ranking metrics (use cached embeddings, not model.recommend()) ---
        ranking_metrics = self._compute_ranking_metrics(
            edge_index, user_emb=user_emb, venue_emb=venue_emb,
            k=k, num_users=num_eval_users
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
        user_emb: torch.Tensor,
        venue_emb: torch.Tensor,
        k: int = 10,
        num_users: int = 100,
    ) -> Dict[str, float]:
        """
        Compute ranking metrics using pre-computed embeddings.

        Uses cosine similarity (dot product on L2-normalized embeddings)
        directly, bypassing the GNN re-encode.  This is identical to what
        evaluate_hybrid.py does, so training metrics are now directly
        comparable to the final recommendation evaluation.
        """
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
            user_gt.setdefault(u, set()).add(v)

        calc = RecommendationMetrics()
        all_metrics = []

        venue_emb_cpu = venue_emb.cpu().numpy()

        for uid in sample_users.tolist():
            if uid not in user_gt:
                continue

            gt = user_gt[uid]
            u_vec = user_emb[uid].cpu().numpy()  # already L2-normalized

            # Cosine similarity = dot product (both sides unit norm)
            scores = venue_emb_cpu @ u_vec           # [n_venues]
            top_recs = np.argsort(-scores)[:k * 2].tolist()

            all_metrics.append({
                f"precision@{k}": calc.precision_at_k(top_recs, gt, k),
                f"recall@{k}": calc.recall_at_k(top_recs, gt, k),
                f"ndcg@{k}": calc.ndcg_at_k(top_recs, gt, k),
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

            # Evaluate on 500 users so per-epoch NDCG is stable enough for
            # reliable early-stopping and best-model selection.
            val_metrics = self.evaluate(num_eval_users=500)
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
        """Extract learned embeddings for all node types (on CPU)."""
        self.model.eval()
        emb_dict = self.model.encode(self.x_dict, self.edge_index_dict)
        return {k: v.cpu() for k, v in emb_dict.items()}
