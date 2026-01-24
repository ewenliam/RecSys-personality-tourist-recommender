"""
Heterogeneous Graph Construction.

Builds a heterogeneous graph with:
- User nodes (MBTI embeddings)
- Venue nodes (topic embeddings)
- Context nodes (region, time, weather)
- Various edge types for relationships
"""
import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.config.settings import GNNConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class NodeData:
    """Container for node features and mappings."""
    features: Tensor
    ids: list
    id_to_idx: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id_to_idx:
            self.id_to_idx = {id_: idx for idx, id_ in enumerate(self.ids)}

    def __len__(self):
        return len(self.ids)

    def get_idx(self, node_id) -> Optional[int]:
        return self.id_to_idx.get(node_id)


@dataclass
class EdgeData:
    """Container for edge indices and attributes."""
    edge_index: Tensor  # [2, num_edges]
    edge_attr: Optional[Tensor] = None  # [num_edges, attr_dim]
    edge_type: Optional[str] = None

    def __len__(self):
        return self.edge_index.size(1)


class HeteroGraphBuilder:
    """Build heterogeneous graph for recommendation."""

    # Node type names
    USER = "user"
    VENUE = "venue"
    TOPIC = "topic"
    REGION = "region"
    TIME = "time"

    # Edge type names (source, relation, target)
    USER_VISITS_VENUE = ("user", "visits", "venue")
    USER_RATES_VENUE = ("user", "rates", "venue")
    VENUE_HAS_TOPIC = ("venue", "has_topic", "topic")
    VENUE_IN_REGION = ("venue", "in_region", "region")
    VENUE_REV_VISITS = ("venue", "rev_visits", "user")  # Reverse edges

    def __init__(
        self,
        config: Optional[GNNConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the graph builder.

        Args:
            config: GNN configuration.
            device: Device for tensors.
        """
        self.config = config or get_config().gnn
        self.device = device or torch.device("cpu")

        # Node storage
        self.nodes: dict[str, NodeData] = {}

        # Edge storage
        self.edges: dict[tuple, EdgeData] = {}

        # Metadata
        self.node_types = []
        self.edge_types = []

    def add_user_nodes(
        self,
        user_ids: list,
        embeddings: np.ndarray,
        mbti_types: Optional[list[str]] = None,
    ) -> None:
        """
        Add user nodes with MBTI embeddings.

        Args:
            user_ids: List of user IDs.
            embeddings: User embeddings [num_users, embedding_dim].
            mbti_types: Optional MBTI type labels.
        """
        features = torch.tensor(embeddings, dtype=torch.float32)

        self.nodes[self.USER] = NodeData(
            features=features,
            ids=user_ids,
        )

        if self.USER not in self.node_types:
            self.node_types.append(self.USER)

        logger.info(f"Added {len(user_ids)} user nodes")

    def add_venue_nodes(
        self,
        venue_ids: list,
        embeddings: np.ndarray,
        topic_ids: Optional[list[int]] = None,
    ) -> None:
        """
        Add venue nodes with topic embeddings.

        Args:
            venue_ids: List of venue IDs.
            embeddings: Venue embeddings [num_venues, embedding_dim].
            topic_ids: Optional topic assignments.
        """
        features = torch.tensor(embeddings, dtype=torch.float32)

        self.nodes[self.VENUE] = NodeData(
            features=features,
            ids=venue_ids,
        )

        if self.VENUE not in self.node_types:
            self.node_types.append(self.VENUE)

        logger.info(f"Added {len(venue_ids)} venue nodes")

    def add_topic_nodes(
        self,
        topic_ids: list[int],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add topic nodes.

        Args:
            topic_ids: List of topic IDs.
            embeddings: Topic embeddings [num_topics, embedding_dim].
        """
        features = torch.tensor(embeddings, dtype=torch.float32)

        self.nodes[self.TOPIC] = NodeData(
            features=features,
            ids=topic_ids,
        )

        if self.TOPIC not in self.node_types:
            self.node_types.append(self.TOPIC)

        logger.info(f"Added {len(topic_ids)} topic nodes")

    def add_region_nodes(
        self,
        region_ids: list[int],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add region (geographic) nodes.

        Args:
            region_ids: List of region IDs.
            embeddings: Region embeddings [num_regions, embedding_dim].
        """
        features = torch.tensor(embeddings, dtype=torch.float32)

        self.nodes[self.REGION] = NodeData(
            features=features,
            ids=region_ids,
        )

        if self.REGION not in self.node_types:
            self.node_types.append(self.REGION)

        logger.info(f"Added {len(region_ids)} region nodes")

    def add_user_venue_edges(
        self,
        user_ids: list,
        venue_ids: list,
        ratings: Optional[list[float]] = None,
        timestamps: Optional[list] = None,
    ) -> None:
        """
        Add user-venue interaction edges.

        Args:
            user_ids: List of user IDs.
            venue_ids: List of venue IDs.
            ratings: Optional rating values.
            timestamps: Optional interaction timestamps.
        """
        if self.USER not in self.nodes or self.VENUE not in self.nodes:
            raise ValueError("User and venue nodes must be added first")

        user_node = self.nodes[self.USER]
        venue_node = self.nodes[self.VENUE]

        # Convert IDs to indices
        src_indices = []
        dst_indices = []
        valid_ratings = []

        for i, (uid, vid) in enumerate(zip(user_ids, venue_ids)):
            user_idx = user_node.get_idx(uid)
            venue_idx = venue_node.get_idx(vid)

            if user_idx is not None and venue_idx is not None:
                src_indices.append(user_idx)
                dst_indices.append(venue_idx)
                if ratings is not None:
                    valid_ratings.append(ratings[i])

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

        edge_attr = None
        if valid_ratings:
            edge_attr = torch.tensor(valid_ratings, dtype=torch.float32).unsqueeze(1)

        # Forward edges
        self.edges[self.USER_VISITS_VENUE] = EdgeData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type="visits",
        )

        # Reverse edges (for message passing)
        self.edges[self.VENUE_REV_VISITS] = EdgeData(
            edge_index=edge_index.flip(0),
            edge_attr=edge_attr,
            edge_type="rev_visits",
        )

        if self.USER_VISITS_VENUE not in self.edge_types:
            self.edge_types.append(self.USER_VISITS_VENUE)
            self.edge_types.append(self.VENUE_REV_VISITS)

        logger.info(f"Added {len(src_indices)} user-venue edges")

    def add_venue_topic_edges(
        self,
        venue_ids: list,
        topic_ids: list[int],
        weights: Optional[list[float]] = None,
    ) -> None:
        """
        Add venue-topic association edges.

        Args:
            venue_ids: List of venue IDs.
            topic_ids: List of topic IDs.
            weights: Optional association weights.
        """
        if self.VENUE not in self.nodes or self.TOPIC not in self.nodes:
            raise ValueError("Venue and topic nodes must be added first")

        venue_node = self.nodes[self.VENUE]
        topic_node = self.nodes[self.TOPIC]

        src_indices = []
        dst_indices = []
        valid_weights = []

        for i, (vid, tid) in enumerate(zip(venue_ids, topic_ids)):
            venue_idx = venue_node.get_idx(vid)
            topic_idx = topic_node.get_idx(tid)

            if venue_idx is not None and topic_idx is not None:
                src_indices.append(venue_idx)
                dst_indices.append(topic_idx)
                if weights is not None:
                    valid_weights.append(weights[i])

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

        edge_attr = None
        if valid_weights:
            edge_attr = torch.tensor(valid_weights, dtype=torch.float32).unsqueeze(1)

        self.edges[self.VENUE_HAS_TOPIC] = EdgeData(
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type="has_topic",
        )

        if self.VENUE_HAS_TOPIC not in self.edge_types:
            self.edge_types.append(self.VENUE_HAS_TOPIC)

        logger.info(f"Added {len(src_indices)} venue-topic edges")

    def add_venue_region_edges(
        self,
        venue_ids: list,
        region_ids: list[int],
    ) -> None:
        """
        Add venue-region edges.

        Args:
            venue_ids: List of venue IDs.
            region_ids: List of region IDs.
        """
        if self.VENUE not in self.nodes or self.REGION not in self.nodes:
            raise ValueError("Venue and region nodes must be added first")

        venue_node = self.nodes[self.VENUE]
        region_node = self.nodes[self.REGION]

        src_indices = []
        dst_indices = []

        for vid, rid in zip(venue_ids, region_ids):
            venue_idx = venue_node.get_idx(vid)
            region_idx = region_node.get_idx(rid)

            if venue_idx is not None and region_idx is not None:
                src_indices.append(venue_idx)
                dst_indices.append(region_idx)

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

        self.edges[self.VENUE_IN_REGION] = EdgeData(
            edge_index=edge_index,
            edge_type="in_region",
        )

        if self.VENUE_IN_REGION not in self.edge_types:
            self.edge_types.append(self.VENUE_IN_REGION)

        logger.info(f"Added {len(src_indices)} venue-region edges")

    def build_pyg_data(self):
        """
        Build PyTorch Geometric HeteroData object.

        Returns:
            HeteroData object with all nodes and edges.
        """
        from torch_geometric.data import HeteroData

        data = HeteroData()

        # Add node features
        for node_type, node_data in self.nodes.items():
            data[node_type].x = node_data.features.to(self.device)
            data[node_type].num_nodes = len(node_data)

        # Add edge indices
        for edge_type, edge_data in self.edges.items():
            data[edge_type].edge_index = edge_data.edge_index.to(self.device)
            if edge_data.edge_attr is not None:
                data[edge_type].edge_attr = edge_data.edge_attr.to(self.device)

        logger.info(f"Built HeteroData with {len(self.node_types)} node types, "
                   f"{len(self.edge_types)} edge types")

        return data

    def build_homogeneous_data(self, target_edge_type: tuple = None):
        """
        Build homogeneous graph for simpler models.

        Focuses on user-venue interactions.

        Args:
            target_edge_type: Edge type to use (default: user-venue).

        Returns:
            PyG Data object.
        """
        from torch_geometric.data import Data

        target_edge_type = target_edge_type or self.USER_VISITS_VENUE

        if target_edge_type not in self.edges:
            raise ValueError(f"Edge type {target_edge_type} not found")

        edge_data = self.edges[target_edge_type]

        # Combine user and venue features
        user_features = self.nodes[self.USER].features
        venue_features = self.nodes[self.VENUE].features

        # Pad to same dimension if needed
        max_dim = max(user_features.size(1), venue_features.size(1))

        if user_features.size(1) < max_dim:
            padding = torch.zeros(user_features.size(0), max_dim - user_features.size(1))
            user_features = torch.cat([user_features, padding], dim=1)

        if venue_features.size(1) < max_dim:
            padding = torch.zeros(venue_features.size(0), max_dim - venue_features.size(1))
            venue_features = torch.cat([venue_features, padding], dim=1)

        # Offset venue indices
        num_users = len(self.nodes[self.USER])
        edge_index = edge_data.edge_index.clone()
        edge_index[1] += num_users

        # Combine features
        x = torch.cat([user_features, venue_features], dim=0)

        data = Data(
            x=x.to(self.device),
            edge_index=edge_index.to(self.device),
        )

        if edge_data.edge_attr is not None:
            data.edge_attr = edge_data.edge_attr.to(self.device)

        # Store metadata
        data.num_users = num_users
        data.num_venues = len(self.nodes[self.VENUE])

        return data

    def get_user_venue_matrix(self) -> torch.Tensor:
        """Get user-venue interaction matrix (sparse)."""
        if self.USER_VISITS_VENUE not in self.edges:
            raise ValueError("User-venue edges not found")

        edge_data = self.edges[self.USER_VISITS_VENUE]
        num_users = len(self.nodes[self.USER])
        num_venues = len(self.nodes[self.VENUE])

        # Create sparse tensor
        indices = edge_data.edge_index
        if edge_data.edge_attr is not None:
            values = edge_data.edge_attr.squeeze()
        else:
            values = torch.ones(indices.size(1))

        matrix = torch.sparse_coo_tensor(
            indices,
            values,
            size=(num_users, num_venues),
        )

        return matrix

    def get_statistics(self) -> dict:
        """Get graph statistics."""
        stats = {
            "node_types": len(self.node_types),
            "edge_types": len(self.edge_types),
            "nodes": {},
            "edges": {},
        }

        for node_type, node_data in self.nodes.items():
            stats["nodes"][node_type] = {
                "count": len(node_data),
                "feature_dim": node_data.features.size(1),
            }

        for edge_type, edge_data in self.edges.items():
            stats["edges"][str(edge_type)] = {
                "count": len(edge_data),
                "has_attr": edge_data.edge_attr is not None,
            }

        return stats

    def save(self, path: Union[str, Path]) -> None:
        """Save the graph to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save as PyG data
        data = self.build_pyg_data()
        torch.save(data, path / "hetero_graph.pt")

        # Save node ID mappings
        mappings = {
            node_type: {"ids": node_data.ids, "id_to_idx": node_data.id_to_idx}
            for node_type, node_data in self.nodes.items()
        }
        torch.save(mappings, path / "node_mappings.pt")

        logger.info(f"Saved graph to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[torch.device] = None):
        """Load a saved graph."""
        from torch_geometric.data import HeteroData

        path = Path(path)

        data = torch.load(path / "hetero_graph.pt", map_location=device)
        mappings = torch.load(path / "node_mappings.pt", map_location=device)

        builder = cls(device=device)

        # Reconstruct nodes
        for node_type in data.node_types:
            features = data[node_type].x
            ids = mappings[node_type]["ids"]
            id_to_idx = mappings[node_type]["id_to_idx"]

            builder.nodes[node_type] = NodeData(
                features=features,
                ids=ids,
                id_to_idx=id_to_idx,
            )
            builder.node_types.append(node_type)

        # Reconstruct edges
        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            edge_attr = data[edge_type].edge_attr if hasattr(data[edge_type], 'edge_attr') else None

            builder.edges[edge_type] = EdgeData(
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            builder.edge_types.append(edge_type)

        logger.info(f"Loaded graph from {path}")
        return builder


def build_recommendation_graph(
    user_profiles: pd.DataFrame,
    venue_topics: pd.DataFrame,
    interactions: pd.DataFrame,
    user_embeddings: np.ndarray,
    venue_embeddings: np.ndarray,
    topic_embeddings: Optional[np.ndarray] = None,
    region_embeddings: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> HeteroGraphBuilder:
    """
    Build complete recommendation graph.

    Args:
        user_profiles: DataFrame with user_id and MBTI info.
        venue_topics: DataFrame with venue_id, topic, and region.
        interactions: DataFrame with user_id, venue_id, rating.
        user_embeddings: User MBTI embeddings.
        venue_embeddings: Venue topic embeddings.
        topic_embeddings: Optional topic centroid embeddings.
        region_embeddings: Optional region embeddings.
        save_path: Path to save the graph.

    Returns:
        HeteroGraphBuilder instance.
    """
    builder = HeteroGraphBuilder()

    # Add user nodes
    user_ids = user_profiles["user_id"].tolist()
    builder.add_user_nodes(user_ids, user_embeddings)

    # Add venue nodes
    venue_ids = venue_topics["venue_id"].tolist()
    builder.add_venue_nodes(venue_ids, venue_embeddings)

    # Add topic nodes if embeddings provided
    if topic_embeddings is not None:
        topic_ids = sorted(venue_topics["topic"].unique())
        topic_ids = [t for t in topic_ids if t >= 0]  # Exclude outliers
        builder.add_topic_nodes(topic_ids, topic_embeddings)

        # Add venue-topic edges
        valid_topics = venue_topics[venue_topics["topic"] >= 0]
        builder.add_venue_topic_edges(
            valid_topics["venue_id"].tolist(),
            valid_topics["topic"].tolist(),
            valid_topics["topic_prob"].tolist() if "topic_prob" in valid_topics.columns else None,
        )

    # Add region nodes if embeddings provided
    if region_embeddings is not None and "region_id" in venue_topics.columns:
        region_ids = sorted(venue_topics["region_id"].dropna().unique())
        region_ids = [int(r) for r in region_ids if r >= 0]
        builder.add_region_nodes(region_ids, region_embeddings)

        # Add venue-region edges
        valid_regions = venue_topics[venue_topics["region_id"] >= 0]
        builder.add_venue_region_edges(
            valid_regions["venue_id"].tolist(),
            valid_regions["region_id"].astype(int).tolist(),
        )

    # Add user-venue interaction edges
    builder.add_user_venue_edges(
        interactions["user_id"].tolist(),
        interactions["business_id"].tolist() if "business_id" in interactions.columns else interactions["venue_id"].tolist(),
        interactions["stars"].tolist() if "stars" in interactions.columns else None,
    )

    # Save if path provided
    if save_path:
        builder.save(save_path)

    return builder
