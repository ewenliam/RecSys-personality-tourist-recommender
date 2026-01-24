"""
Linear-Time Graph Neural Network (LTGNN).

Implements a scalable GNN using:
- Single propagation layer (avoids over-smoothing)
- Fixed-point iteration for multi-hop information
- EVR (Expected Variance Reduction) sampling
- Linear complexity O(|E|)

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
    Light Graph Convolution layer.

    Simplified message passing without feature transformation,
    following LightGCN design principles.
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize the layer.

        Args:
            normalize: Whether to apply symmetric normalization.
        """
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, feature_dim].
            edge_index: Edge indices [2, num_edges].
            edge_weight: Optional edge weights [num_edges].

        Returns:
            Aggregated features [num_nodes, feature_dim].
        """
        row, col = edge_index
        num_nodes = x.size(0)

        # Compute normalization
        if self.normalize:
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=x.device)

            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        else:
            norm = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1), device=x.device)

        # Message passing
        out = torch.zeros_like(x)
        out.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row] * norm.unsqueeze(1))

        return out


class FixedPointLayer(nn.Module):
    """
    Fixed-point iteration layer for multi-hop aggregation.

    Instead of stacking multiple GNN layers (which causes over-smoothing),
    we use fixed-point iteration to accumulate multi-hop information.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_iterations: int = 10,
        alpha: float = 0.5,
        dropout: float = 0.1,
    ):
        """
        Initialize the layer.

        Args:
            hidden_dim: Hidden dimension.
            num_iterations: Number of fixed-point iterations.
            alpha: Residual connection weight.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.alpha = alpha

        self.conv = LightGCNConv(normalize=True)
        self.dropout = nn.Dropout(dropout)

        # Learnable iteration weights
        self.iteration_weights = nn.Parameter(torch.ones(num_iterations) / num_iterations)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with fixed-point iteration.

        Args:
            x: Input features [num_nodes, feature_dim].
            edge_index: Edge indices [2, num_edges].
            edge_weight: Optional edge weights.

        Returns:
            Output features [num_nodes, feature_dim].
        """
        # Initial embedding
        h = x
        embeddings = [h]

        # Fixed-point iterations
        for i in range(self.num_iterations):
            # Message passing
            h_new = self.conv(h, edge_index, edge_weight)

            # Residual connection
            h = self.alpha * x + (1 - self.alpha) * h_new

            # Apply dropout
            h = self.dropout(h)

            embeddings.append(h)

        # Weighted combination of all iterations
        weights = F.softmax(self.iteration_weights, dim=0)
        out = torch.zeros_like(x)
        for i, emb in enumerate(embeddings):
            out = out + weights[i] * emb

        return out


class LTGNN(nn.Module):
    """
    Linear-Time Graph Neural Network for recommendations.

    Features:
    - Input projection layers for different node types
    - Fixed-point iteration for multi-hop aggregation
    - Link prediction head for user-venue scoring
    """

    def __init__(
        self,
        user_input_dim: int,
        venue_input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_iterations: int = 10,
        dropout: float = 0.2,
        config: Optional[GNNConfig] = None,
    ):
        """
        Initialize the LTGNN model.

        Args:
            user_input_dim: Dimension of user input features.
            venue_input_dim: Dimension of venue input features.
            hidden_dim: Hidden layer dimension.
            embedding_dim: Output embedding dimension.
            num_iterations: Fixed-point iterations.
            dropout: Dropout rate.
            config: GNN configuration.
        """
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

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(
        self,
        user_x: Tensor,
        venue_x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode users and venues to embeddings.

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

        # Project inputs to same dimension
        user_h = self.user_projection(user_x)
        venue_h = self.venue_projection(venue_x)

        # Combine for GNN (users then venues)
        x = torch.cat([user_h, venue_h], dim=0)

        # Adjust edge indices for combined graph
        adjusted_edge_index = edge_index.clone()
        adjusted_edge_index[1] = adjusted_edge_index[1] + num_users

        # Add reverse edges
        reverse_edge_index = adjusted_edge_index.flip(0)
        full_edge_index = torch.cat([adjusted_edge_index, reverse_edge_index], dim=1)

        if edge_weight is not None:
            full_edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
        else:
            full_edge_weight = None

        # Apply GNN
        h = self.gnn(x, full_edge_index, full_edge_weight)

        # Project to output dimension
        out = self.output_projection(h)

        # Split back to users and venues
        user_embeddings = out[:num_users]
        venue_embeddings = out[num_users:]

        return user_embeddings, venue_embeddings

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
        # Encode all nodes
        user_emb, venue_emb = self.encode(user_x, venue_x, edge_index, edge_weight)

        # Get embeddings for prediction pairs
        user_pred_emb = user_emb[user_indices]
        venue_pred_emb = venue_emb[venue_indices]

        # Predict scores
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
        """
        Predict scores with sigmoid activation.

        Returns:
            Predicted probabilities [batch_size].
        """
        scores = self.forward(
            user_x, venue_x, edge_index,
            user_indices, venue_indices, edge_weight
        )
        return torch.sigmoid(scores)

    def recommend(
        self,
        user_x: Tensor,
        venue_x: Tensor,
        edge_index: Tensor,
        user_idx: int,
        k: int = 10,
        exclude_venues: Optional[set] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get top-k venue recommendations for a user.

        Args:
            user_x: User features.
            venue_x: Venue features.
            edge_index: Edge indices.
            user_idx: User index to recommend for.
            k: Number of recommendations.
            exclude_venues: Venue indices to exclude (e.g., already visited).
            edge_weight: Optional edge weights.

        Returns:
            Tuple of (venue_indices, scores).
        """
        self.eval()
        with torch.no_grad():
            user_emb, venue_emb = self.encode(user_x, venue_x, edge_index, edge_weight)

            # Get user embedding
            user_emb_single = user_emb[user_idx].unsqueeze(0)

            # Score all venues
            user_expanded = user_emb_single.expand(venue_emb.size(0), -1)
            combined = torch.cat([user_expanded, venue_emb], dim=1)
            scores = torch.sigmoid(self.link_predictor(combined).squeeze(-1))

            # Exclude specified venues
            if exclude_venues:
                for vid in exclude_venues:
                    scores[vid] = -float('inf')

            # Get top-k
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
        """
        Initialize gBCE loss.

        Args:
            t: Calibration temperature (0 < t <= 1).
                Lower t = more penalty for overconfidence.
            reduction: Reduction method ("mean", "sum", "none").
        """
        super().__init__()
        self.t = t
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute gBCE loss.

        Args:
            predictions: Predicted probabilities [batch_size].
            targets: Binary labels [batch_size].

        Returns:
            Loss value.
        """
        # Apply temperature scaling
        calibrated_preds = torch.pow(predictions, 1.0 / self.t)

        # Standard BCE
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
    """
    Bayesian Personalized Ranking Loss.

    Pairwise ranking loss for recommendation.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pos_scores: Tensor,
        neg_scores: Tensor,
    ) -> Tensor:
        """
        Compute BPR loss.

        Args:
            pos_scores: Scores for positive items [batch_size].
            neg_scores: Scores for negative items [batch_size].

        Returns:
            Loss value.
        """
        loss = -F.logsigmoid(pos_scores - neg_scores)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
