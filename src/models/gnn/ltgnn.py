"""
Linear-Time Graph Neural Network (LTGNN).

Implements a scalable GNN using:
- Sparse matrix multiplication for O(|E|) convolution
- Fixed-point iteration for multi-hop information
- Cached embeddings for efficient mini-batch training
- gBCE loss for confidence calibration

Reference:
    Zhang et al. "Linear-Time Graph Neural Networks for Scalable Recommendations"
    WWW 2024
"""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config.settings import GNNConfig, get_config

logger = logging.getLogger(__name__)


class LightGCNConv(nn.Module):
    """
    Light Graph Convolution using sparse matrix multiplication.

    Instead of gathering x[row] (which creates a [num_edges, hidden_dim]
    tensor that blows up GPU memory), we build a sparse normalized
    adjacency matrix and do adj @ x. Same math, O(|E|) memory.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Propagate features through sparse adjacency.

        Args:
            x: Node features [num_nodes, hidden_dim].
            adj: Pre-built sparse adjacency [num_nodes, num_nodes].

        Returns:
            Propagated features [num_nodes, hidden_dim].
        """
        return torch.sparse.mm(adj, x)


def build_sparse_adj(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tensor:
    """
    Build a sparse normalized adjacency matrix.

    This replaces the per-call gather/scatter pattern with a single
    sparse matrix that can be reused across iterations.

    Args:
        edge_index: Edge indices [2, num_edges].
        num_nodes: Total number of nodes.
        edge_weight: Optional edge weights.
        normalize: Apply symmetric normalization (D^-0.5 A D^-0.5).

    Returns:
        Sparse COO tensor [num_nodes, num_nodes].
    """
    row, col = edge_index

    if normalize:
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        if edge_weight is None:
            values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    else:
        if edge_weight is not None:
            values = edge_weight
        else:
            values = torch.ones(edge_index.size(1), device=edge_index.device)

    adj = torch.sparse_coo_tensor(
        edge_index, values, (num_nodes, num_nodes)
    ).coalesce()

    return adj


class FixedPointLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_iterations: int = 5,
        alpha: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.alpha = alpha

        self.conv = LightGCNConv(normalize=True)
        self.dropout = nn.Dropout(dropout)

        # Weights: num_iterations + 1 to include h0
        self.iteration_weights = nn.Parameter(
            torch.ones(num_iterations + 1) / (num_iterations + 1)
        )

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Fixed-point iteration with sparse adjacency.

        Accumulates the weighted sum incrementally instead of storing
        all iteration embeddings in a list.  This halves peak memory
        during inference (3 live tensors vs 6+) while producing
        mathematically identical output.

        Args:
            x: Node features [num_nodes, hidden_dim].
            adj: Sparse adjacency matrix [num_nodes, num_nodes].
        """
        weights = F.softmax(self.iteration_weights, dim=0)

        h = x
        out = weights[0] * h  # h0 contribution

        for i in range(self.num_iterations):
            h_new = self.conv(h, adj)
            h = self.alpha * x + (1 - self.alpha) * h_new
            h = self.dropout(h)
            out = out + weights[i + 1] * h

        return out


class LTGNN(nn.Module):
    """
    Linear-Time Graph Neural Network for recommendations.

    Architecture:
    - Input projections (user_dim -> hidden, venue_dim -> hidden)
    - Fixed-point iteration with sparse LightGCN convolution
    - Output projection (hidden -> embedding_dim)
    - Link prediction MLP (embedding_dim * 2 -> 1)
    """

    def __init__(
        self,
        user_input_dim: int,
        venue_input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_iterations: int = 5,
        dropout: float = 0.2,
        config: Optional[GNNConfig] = None,
    ):
        super().__init__()
        self.config = config or get_config().gnn

        self.user_input_dim = user_input_dim
        self.venue_input_dim = venue_input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Input projections
        self.user_projection = nn.Sequential(
            nn.Linear(user_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.venue_projection = nn.Sequential(
            nn.Linear(venue_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fixed-point GNN layer
        self.gnn = FixedPointLayer(
            hidden_dim=hidden_dim,
            num_iterations=num_iterations,
            dropout=dropout,
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)

        # Link prediction MLP
        self.link_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Cached sparse adjacency (rebuilt when edge_index changes)
        self._cached_adj = None
        self._cached_edge_id = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_adj(
        self,
        edge_index: Tensor,
        num_nodes: int,
        num_users: int,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Get or build cached sparse adjacency matrix.

        Args:
            edge_index: [2, E] where row 0 = user indices in [0, num_users),
                        row 1 = venue indices in [0, num_venues).
                        Do NOT pass already-offset indices.
            num_nodes: num_users + num_venues.
            num_users: Number of user nodes (used to offset venue indices).
            edge_weight: Optional edge weights.
        """
        edge_id = id(edge_index)
        if self._cached_adj is not None and self._cached_edge_id == edge_id:
            return self._cached_adj

        num_venues = num_nodes - num_users

        # --- Validate indices BEFORE hitting CUDA kernels ---
        if edge_index.numel() > 0:
            max_user = edge_index[0].max().item()
            min_user = edge_index[0].min().item()
            max_venue = edge_index[1].max().item()
            min_venue = edge_index[1].min().item()

            if min_user < 0 or min_venue < 0:
                raise ValueError(
                    f"Negative indices in edge_index: "
                    f"user range [{min_user}, {max_user}], "
                    f"venue range [{min_venue}, {max_venue}]"
                )
            if max_user >= num_users:
                logger.warning(
                    f"User index {max_user} >= num_users {num_users}. "
                    f"Clamping to valid range."
                )
                edge_index = edge_index.clone()
                edge_index[0] = edge_index[0].clamp(0, num_users - 1)
            if max_venue >= num_venues:
                logger.warning(
                    f"Venue index {max_venue} >= num_venues {num_venues}. "
                    f"Clamping to valid range. If you used "
                    f"build_homogeneous_data(), pass raw edges instead."
                )
                edge_index = edge_index.clone()
                edge_index[1] = edge_index[1].clamp(0, num_venues - 1)

        # Build full bidirectional edge index
        # Venue indices are offset by num_users to share a single node space
        adjusted = edge_index.clone()
        adjusted[1] = adjusted[1] + num_users
        reverse = adjusted.flip(0)
        full_edge_index = torch.cat([adjusted, reverse], dim=1)

        if edge_weight is not None:
            full_weight = torch.cat([edge_weight, edge_weight])
        else:
            full_weight = None

        adj = build_sparse_adj(full_edge_index, num_nodes, full_weight)
        self._cached_adj = adj
        self._cached_edge_id = edge_id
        return adj

    def encode(
        self,
        user_x: Tensor,
        venue_x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode users and venues to embeddings.

        Uses sparse adjacency multiplication instead of dense
        gather/scatter - reduces memory from ~3GB to ~70MB per conv.

        Args:
            user_x: User features [num_users, user_input_dim].
            venue_x: Venue features [num_venues, venue_input_dim].
            edge_index: User-venue edge indices [2, num_edges].
            edge_weight: Optional edge weights.

        Returns:
            Tuple of (user_embeddings, venue_embeddings).
        """
        num_users = user_x.size(0)
        num_venues = venue_x.size(0)
        num_nodes = num_users + num_venues

        # Validate edge indices match feature dimensions
        if edge_index.numel() > 0:
            max_u = edge_index[0].max().item()
            max_v = edge_index[1].max().item()
            if max_u >= num_users or max_v >= num_venues:
                logger.warning(
                    f"Edge index OOB: max_user={max_u} (num_users={num_users}), "
                    f"max_venue={max_v} (num_venues={num_venues}). "
                    f"Filtering invalid edges."
                )
                valid = (edge_index[0] < num_users) & (edge_index[1] < num_venues)
                edge_index = edge_index[:, valid]
                if edge_weight is not None:
                    edge_weight = edge_weight[valid]
                logger.info(f"Kept {valid.sum().item()} / {valid.numel()} edges")

        # Project inputs to same dimension
        user_h = self.user_projection(user_x)
        venue_h = self.venue_projection(venue_x)

        # Combine for GNN (users then venues)
        x = torch.cat([user_h, venue_h], dim=0)

        # Build sparse adjacency (cached after first call)
        # Invalidate cache when edges were filtered
        adj = self._get_adj(edge_index, num_nodes, num_users, edge_weight)

        # Apply GNN with sparse conv
        h = self.gnn(x, adj)

        # Project to output dimension
        out = self.output_projection(h)

        # Split back to users and venues
        user_embeddings = out[:num_users]
        venue_embeddings = out[num_users:]

        return user_embeddings, venue_embeddings

    @torch.no_grad()
    def encode_inference(
        self,
        user_x: Tensor,
        venue_x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        chunk_size: int = 8192,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Memory-safe encoding for inference on ALL nodes.

        Processes projection layers in chunks to cap peak GPU memory,
        runs the GNN on compact hidden features (which are much smaller
        than the raw inputs), and returns embeddings on CPU.

        Args:
            user_x: User features [num_users, user_input_dim].
            venue_x: Venue features [num_venues, venue_input_dim].
            edge_index: User-venue edge indices [2, num_edges].
            edge_weight: Optional edge weights.
            chunk_size: Nodes per projection chunk.
            device: Device for GNN computation (default: model device).

        Returns:
            Tuple of (user_embeddings, venue_embeddings) on CPU.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        num_users = user_x.size(0)
        num_venues = venue_x.size(0)
        num_nodes = num_users + num_venues

        # Phase A: Project user features in chunks -> hidden_dim
        user_h_parts = []
        for start in range(0, num_users, chunk_size):
            end = min(start + chunk_size, num_users)
            chunk = user_x[start:end].to(device)
            user_h_parts.append(self.user_projection(chunk).cpu())
            del chunk
        user_h = torch.cat(user_h_parts, dim=0)
        del user_h_parts

        # Phase B: Project venue features in chunks -> hidden_dim
        venue_h_parts = []
        for start in range(0, num_venues, chunk_size):
            end = min(start + chunk_size, num_venues)
            chunk = venue_x[start:end].to(device)
            venue_h_parts.append(self.venue_projection(chunk).cpu())
            del chunk
        venue_h = torch.cat(venue_h_parts, dim=0)
        del venue_h_parts

        # Phase C: Move compact hidden features to GPU for GNN
        # [num_nodes, hidden_dim] - much smaller than the raw input
        x = torch.cat([user_h, venue_h], dim=0).to(device)
        del user_h, venue_h

        # Build sparse adjacency (cached after first call)
        edge_index_dev = edge_index.to(device)
        ew_dev = edge_weight.to(device) if edge_weight is not None else None
        adj = self._get_adj(edge_index_dev, num_nodes, num_users, ew_dev)

        # GNN message passing (sparse matmul, O(|E|) memory)
        h = self.gnn(x, adj)
        del x

        # Phase D: Output projection in chunks, results to CPU
        user_emb_parts = []
        for start in range(0, num_users, chunk_size):
            end = min(start + chunk_size, num_users)
            user_emb_parts.append(
                self.output_projection(h[start:end]).cpu()
            )
        user_emb = torch.cat(user_emb_parts, dim=0)
        del user_emb_parts

        venue_emb_parts = []
        for start in range(0, num_venues, chunk_size):
            end = min(start + chunk_size, num_venues)
            offset = num_users + start
            venue_emb_parts.append(
                self.output_projection(h[offset:offset + (end - start)]).cpu()
            )
        venue_emb = torch.cat(venue_emb_parts, dim=0)
        del venue_emb_parts, h

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return user_emb, venue_emb

    def forward(
        self,
        user_x: Tensor,
        venue_x: Tensor,
        edge_index: Tensor,
        user_indices: Tensor,
        venue_indices: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for link prediction.

        Args:
            user_x: User features.
            venue_x: Venue features.
            edge_index: Edge indices for message passing.
            user_indices: User indices for prediction [batch_size].
            venue_indices: Venue indices for prediction [batch_size].
            edge_weight: Optional edge weights.

        Returns:
            Predicted scores [batch_size].
        """
        user_emb, venue_emb = self.encode(
            user_x, venue_x, edge_index, edge_weight
        )

        user_pred_emb = user_emb[user_indices]
        venue_pred_emb = venue_emb[venue_indices]

        combined = torch.cat([user_pred_emb, venue_pred_emb], dim=1)
        scores = self.link_predictor(combined).squeeze(-1)

        return scores

    def predict(
        self,
        user_x: Tensor,
        venue_x: Tensor,
        edge_index: Tensor,
        user_indices: Tensor,
        venue_indices: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict scores with sigmoid activation."""
        scores = self.forward(
            user_x, venue_x, edge_index,
            user_indices, venue_indices, edge_weight
        )
        return torch.sigmoid(scores)

    def predict_from_embeddings(
        self,
        user_emb: Tensor,
        venue_emb: Tensor,
        user_indices: Tensor,
        venue_indices: Tensor,
    ) -> Tensor:
        """Predict scores from pre-computed embeddings (for cached training)."""
        u = user_emb[user_indices]
        v = venue_emb[venue_indices]
        combined = torch.cat([u, v], dim=1)
        return self.link_predictor(combined).squeeze(-1)

    def recommend(
        self,
        user_x: Tensor,
        venue_x: Tensor,
        edge_index: Tensor,
        user_idx: int,
        k: int = 10,
        exclude_venues: Optional[set] = None,
        edge_weight: Optional[Tensor] = None,
        cached_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get top-k venue recommendations for a user.

        Args:
            user_x: User features.
            venue_x: Venue features.
            edge_index: Edge indices.
            user_idx: User index to recommend for.
            k: Number of recommendations.
            exclude_venues: Venue indices to exclude.
            edge_weight: Optional edge weights.
            cached_embeddings: Pre-computed (user_emb, venue_emb) to skip encode.

        Returns:
            Tuple of (venue_indices, scores).
        """
        self.eval()
        with torch.no_grad():
            if cached_embeddings is not None:
                user_emb, venue_emb = cached_embeddings
            else:
                user_emb, venue_emb = self.encode(
                    user_x, venue_x, edge_index, edge_weight
                )

            user_emb_single = user_emb[user_idx].unsqueeze(0)
            user_expanded = user_emb_single.expand(venue_emb.size(0), -1)
            combined = torch.cat([user_expanded, venue_emb], dim=1)
            scores = torch.sigmoid(
                self.link_predictor(combined).squeeze(-1)
            )

            if exclude_venues:
                for vid in exclude_venues:
                    scores[vid] = -float('inf')

            top_scores, top_indices = torch.topk(scores, k)

        return top_indices, top_scores


class gBCELoss(nn.Module):
    """
    Generalized Binary Cross-Entropy Loss.

    Adds calibration to reduce overconfidence in predictions.

    Reference:
        "Reducing Overconfidence in Sequential Recommendation" RecSys 2023
    """

    def __init__(self, t: float = 0.8, reduction: str = "mean"):
        super().__init__()
        self.t = t
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        calibrated_preds = torch.pow(predictions, 1.0 / self.t)

        eps = 1e-7
        loss = -(
            targets * torch.log(calibrated_preds + eps) +
            (1 - targets) * torch.log(1 - calibrated_preds + eps)
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        loss = -F.logsigmoid(pos_scores - neg_scores)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
