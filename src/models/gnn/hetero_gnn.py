"""
Heterogeneous Graph Neural Network for Tourist Recommendations.

Implements GraphSAGE-based heterogeneous message passing that natively
respects different node types (User, Venue) and edge types (visits,
rev_visits, has_topic), replacing the homogeneous LTGNN approach.

Key advantages over homogeneous LTGNN:
  - No manual index offsetting or feature concatenation
  - Separate learned transformations per edge type
  - Attention-based aggregation across edge types (optional)
  - Personality embeddings preserved as first-class user features
  - Better cold-start performance via personality-informed initialization

Architecture:
  User nodes  --(BERT-MBTI CLS, 768d)--> Linear --> hidden_dim
  Venue nodes --(BERTopic dist, Kd)-----> Linear --> hidden_dim
  2x HeteroSAGEConv layers with ReLU + dropout
  Output projection --> embedding_dim per node type
  Link prediction MLP: concat(user_emb, venue_emb) --> score
"""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config.settings import GNNConfig, get_config

logger = logging.getLogger(__name__)


class HeteroSAGEConv(nn.Module):
    """
    Heterogeneous GraphSAGE convolution layer.

    Uses PyG's HeteroConv to apply separate SAGEConv transformations
    for each edge type, then aggregates results per node type.
    """

    def __init__(
        self,
        in_channels: dict[str, int],
        out_channels: int,
        edge_types: list[tuple[str, str, str]],
        aggr: str = "mean",
    ):
        """
        Initialize heterogeneous SAGE layer.

        Args:
            in_channels: Dict mapping node_type -> input dim.
            out_channels: Output dimension (same for all node types).
            edge_types: List of (src_type, rel_type, dst_type) tuples.
            aggr: Aggregation method for combining messages.
        """
        super().__init__()

        from torch_geometric.nn import SAGEConv, HeteroConv

        # Create a SAGEConv for each edge type
        convs = {}
        for edge_type in edge_types:
            src_type, _, dst_type = edge_type
            src_dim = in_channels.get(src_type, out_channels)
            dst_dim = in_channels.get(dst_type, out_channels)
            # SAGEConv aggregates: dst <- src messages
            convs[edge_type] = SAGEConv(
                (src_dim, dst_dim),
                out_channels,
                aggr=aggr,
            )

        self.conv = HeteroConv(convs, aggr="sum")

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple, Tensor],
    ) -> dict[str, Tensor]:
        """
        Forward pass: heterogeneous message passing.

        Args:
            x_dict: Dict mapping node_type -> features [num_nodes, in_dim].
            edge_index_dict: Dict mapping edge_type -> edge_index [2, E].

        Returns:
            Dict mapping node_type -> updated features [num_nodes, out_dim].
        """
        return self.conv(x_dict, edge_index_dict)


class HeteroGNN(nn.Module):
    """
    Full heterogeneous GNN for tourist venue recommendation.

    User nodes initialized with BERT-MBTI CLS embeddings capture
    personality-discriminative features. Venue nodes initialized
    with BERTopic distributions capture thematic profiles.
    Message passing across user-venue edges propagates personality
    signals to venues and venue signals to users.
    """

    def __init__(
        self,
        node_input_dims: dict[str, int],
        edge_types: list[tuple[str, str, str]],
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        config: Optional[GNNConfig] = None,
    ):
        """
        Initialize the heterogeneous GNN.

        Args:
            node_input_dims: Dict mapping node_type -> input feature dim.
                e.g. {"user": 768, "venue": 50}
            edge_types: List of (src, rel, dst) tuples defining the schema.
                e.g. [("user", "visits", "venue"), ("venue", "rev_visits", "user")]
            hidden_dim: Hidden layer dimension.
            embedding_dim: Output embedding dimension.
            num_layers: Number of HeteroSAGE layers.
            dropout: Dropout rate.
            config: GNN configuration.
        """
        super().__init__()
        self.config = config or get_config().gnn
        self.node_types = list(node_input_dims.keys())
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Input projections: project each node type to hidden_dim
        self.input_projections = nn.ModuleDict()
        for node_type, in_dim in node_input_dims.items():
            self.input_projections[node_type] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # Heterogeneous SAGE convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            # First layer: hidden_dim -> hidden_dim (all node types projected)
            in_channels = {nt: hidden_dim for nt in self.node_types}
            self.convs.append(
                HeteroSAGEConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    edge_types=edge_types,
                )
            )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projections per node type
        self.output_projections = nn.ModuleDict()
        for node_type in self.node_types:
            self.output_projections[node_type] = nn.Linear(hidden_dim, embedding_dim)

        # NOTE: no MLP link predictor here.  Scoring is done via dot product
        # on L2-normalized embeddings (= cosine similarity), so the training
        # objective directly optimizes the retrieval metric used at inference.
        # An MLP predictor would learn discriminative scores internally while
        # leaving the embeddings collapsed, making cosine-based retrieval
        # useless at evaluation time.

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple, Tensor],
    ) -> dict[str, Tensor]:
        """
        Encode all node types to embeddings.

        Args:
            x_dict: Dict mapping node_type -> raw features.
            edge_index_dict: Dict mapping edge_type -> edge indices.

        Returns:
            Dict mapping node_type -> embeddings [num_nodes, embedding_dim].
        """
        # Project inputs to hidden dimension
        h_dict = {}
        for node_type in x_dict:
            if node_type in self.input_projections:
                h_dict[node_type] = self.input_projections[node_type](x_dict[node_type])
            else:
                h_dict[node_type] = x_dict[node_type]

        # Apply heterogeneous SAGE layers
        for i, conv in enumerate(self.convs):
            h_new = conv(h_dict, edge_index_dict)

            # Residual connection + activation + dropout
            for node_type in h_new:
                if node_type in h_dict:
                    # Residual connection
                    h_new[node_type] = h_new[node_type] + h_dict[node_type]
                h_new[node_type] = F.relu(h_new[node_type])
                h_new[node_type] = self.dropout(h_new[node_type])

            h_dict = h_new

        # Project to embedding dimension and L2-normalize.
        # L2 normalization is critical: it maps all embeddings onto the unit
        # hypersphere so that dot product == cosine similarity.  Without it,
        # nodes with high-norm hidden representations dominate recommendations
        # regardless of actual affinity.  Combined with BPR loss, this ensures
        # the training objective directly optimizes cosine-based retrieval.
        emb_dict = {}
        for node_type in h_dict:
            if node_type in self.output_projections:
                raw = self.output_projections[node_type](h_dict[node_type])
                emb_dict[node_type] = F.normalize(raw, p=2, dim=-1)
            else:
                emb_dict[node_type] = F.normalize(h_dict[node_type], p=2, dim=-1)

        return emb_dict

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple, Tensor],
        user_indices: Tensor,
        venue_indices: Tensor,
    ) -> Tensor:
        """
        Forward pass: dot-product score between user and venue embeddings.

        Both embeddings are L2-normalized by encode(), so this is equivalent
        to cosine similarity.  Scores are in [-1, 1] (no sigmoid applied here;
        the BPR loss directly uses the raw dot products).

        Args:
            x_dict: Node features per type.
            edge_index_dict: Edge indices per type.
            user_indices: User indices for scoring [batch_size].
            venue_indices: Venue indices for scoring [batch_size].

        Returns:
            Dot-product scores [batch_size] in [-1, 1].
        """
        emb_dict = self.encode(x_dict, edge_index_dict)
        user_emb = emb_dict["user"][user_indices]   # [B, d], L2-normalized
        venue_emb = emb_dict["venue"][venue_indices]  # [B, d], L2-normalized
        return (user_emb * venue_emb).sum(dim=-1)   # cosine similarity

    def predict(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple, Tensor],
        user_indices: Tensor,
        venue_indices: Tensor,
    ) -> Tensor:
        """Dot-product score mapped to [0, 1] via sigmoid (for AUC metrics)."""
        scores = self.forward(x_dict, edge_index_dict, user_indices, venue_indices)
        return torch.sigmoid(scores)

    def recommend(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple, Tensor],
        user_idx: int,
        k: int = 10,
        exclude_venues: Optional[set] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get top-k venue recommendations via cosine similarity.

        Because encode() L2-normalizes outputs, dot product here is
        identical to cosine similarity (no extra normalize step needed).

        Args:
            x_dict: Node features per type.
            edge_index_dict: Edge indices per type.
            user_idx: User index to recommend for.
            k: Number of recommendations.
            exclude_venues: Venue indices to exclude.

        Returns:
            Tuple of (venue_indices, cosine_scores).
        """
        self.eval()
        with torch.no_grad():
            emb_dict = self.encode(x_dict, edge_index_dict)
            user_emb = emb_dict["user"][user_idx]   # [d], unit norm
            venue_emb = emb_dict["venue"]            # [n_v, d], unit norm

            scores = venue_emb @ user_emb            # cosine similarity [n_v]

            if exclude_venues:
                for vid in exclude_venues:
                    if vid < scores.size(0):
                        scores[vid] = -float("inf")

            top_scores, top_indices = torch.topk(scores, min(k, scores.size(0)))

        return top_indices, top_scores


def build_hetero_gnn_from_graph(
    graph_builder,
    hidden_dim: int = 128,
    embedding_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> HeteroGNN:
    """
    Factory: create a HeteroGNN from a populated HeteroGraphBuilder.

    Args:
        graph_builder: HeteroGraphBuilder with nodes and edges added.
        hidden_dim: Hidden dimension.
        embedding_dim: Output embedding dimension.
        num_layers: Number of SAGE layers.
        dropout: Dropout rate.

    Returns:
        Initialized HeteroGNN model.
    """
    # Extract input dimensions from graph
    node_input_dims = {}
    for node_type, node_data in graph_builder.nodes.items():
        node_input_dims[node_type] = node_data.features.size(1)

    # Extract edge types
    edge_types = list(graph_builder.edges.keys())

    model = HeteroGNN(
        node_input_dims=node_input_dims,
        edge_types=edge_types,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Created HeteroGNN: {len(node_input_dims)} node types, "
        f"{len(edge_types)} edge types, {n_params:,} parameters"
    )

    return model
