"""Graph Neural Network module."""
from .graph_builder import (
    HeteroGraphBuilder,
    NodeData,
    EdgeData,
    build_recommendation_graph,
)
from .ltgnn import (
    LTGNN,
    LightGCNConv,
    FixedPointLayer,
    gBCELoss,
    BPRLoss,
)
from .gnn_v2 import (
    AugmentedLTGNN,
    build_user_user_edges,
    build_augmented_sparse_adj,
)
from .hetero_gnn import (
    HeteroGNN,
    HeteroSAGEConv,
    build_hetero_gnn_from_graph,
)
from .hetero_trainer import (
    HeteroGNNTrainer,
    HeteroGNNMetrics,
)
from .evr_sampler import (
    NeighborSampler,
    EVRSampler,
    MiniBatchLoader,
    TrainTestSplit,
)
from .trainer import (
    GNNTrainer,
    GNNTrainingMetrics,
    train_gnn,
)

__all__ = [
    # Graph building
    "HeteroGraphBuilder",
    "NodeData",
    "EdgeData",
    "build_recommendation_graph",
    # Homogeneous models
    "LTGNN",
    "LightGCNConv",
    "FixedPointLayer",
    "gBCELoss",
    "BPRLoss",
    # V2: User-User edge augmentation
    "AugmentedLTGNN",
    "build_user_user_edges",
    "build_augmented_sparse_adj",
    # Heterogeneous models
    "HeteroGNN",
    "HeteroSAGEConv",
    "build_hetero_gnn_from_graph",
    "HeteroGNNTrainer",
    "HeteroGNNMetrics",
    # Sampling
    "NeighborSampler",
    "EVRSampler",
    "MiniBatchLoader",
    "TrainTestSplit",
    # Training
    "GNNTrainer",
    "GNNTrainingMetrics",
    "train_gnn",
]
