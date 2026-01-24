"""
Expected Variance Reduction (EVR) Sampler.

Implements variance-reduced neighbor sampling for scalable GNN training.
Reduces the neighbor size during embedding aggregation while maintaining
accuracy through importance sampling.

Reference:
    Zhang et al. "Linear-Time Graph Neural Networks for Scalable Recommendations"
    WWW 2024
"""
import logging
from typing import Optional, Tuple, List

import torch
import numpy as np
from torch import Tensor

from src.config.settings import GNNConfig, get_config

logger = logging.getLogger(__name__)


class NeighborSampler:
    """
    Basic neighbor sampler for mini-batch training.
    """

    def __init__(
        self,
        edge_index: Tensor,
        num_nodes: int,
        sizes: List[int],
        edge_weight: Optional[Tensor] = None,
    ):
        """
        Initialize the sampler.

        Args:
            edge_index: Edge indices [2, num_edges].
            num_nodes: Total number of nodes.
            sizes: Number of neighbors to sample at each layer.
            edge_weight: Optional edge weights for weighted sampling.
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.sizes = sizes
        self.edge_weight = edge_weight

        # Build adjacency list for efficient sampling
        self._build_adj_list()

    def _build_adj_list(self):
        """Build adjacency list from edge index."""
        self.adj_list = [[] for _ in range(self.num_nodes)]
        self.weight_list = [[] for _ in range(self.num_nodes)]

        edge_index = self.edge_index.cpu().numpy()
        weights = self.edge_weight.cpu().numpy() if self.edge_weight is not None else None

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            self.adj_list[dst].append(src)  # For message passing to dst
            if weights is not None:
                self.weight_list[dst].append(weights[i])

        # Convert to numpy for faster indexing
        self.adj_list = [np.array(neighbors) for neighbors in self.adj_list]
        self.weight_list = [np.array(w) if w else None for w in self.weight_list]

    def sample(self, node_indices: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Sample neighbors for given nodes.

        Args:
            node_indices: Target node indices [batch_size].

        Returns:
            Tuple of:
                - n_id: All sampled node indices
                - edge_index: Sampled edge indices
                - adjs: List of (edge_index, size) tuples per layer
        """
        node_indices = node_indices.cpu().numpy()
        batch_size = len(node_indices)

        n_ids = [node_indices]
        adjs = []

        for size in self.sizes:
            # Get current frontier
            frontier = n_ids[-1]

            # Sample neighbors for each node
            sampled_neighbors = []
            sampled_sources = []
            sampled_targets = []

            for i, node in enumerate(frontier):
                neighbors = self.adj_list[node]

                if len(neighbors) == 0:
                    continue

                if len(neighbors) <= size:
                    # Take all neighbors
                    selected = neighbors
                else:
                    # Sample uniformly
                    idx = np.random.choice(len(neighbors), size, replace=False)
                    selected = neighbors[idx]

                sampled_neighbors.extend(selected)
                sampled_sources.extend(selected)
                sampled_targets.extend([i] * len(selected))

            # Add sampled neighbors to n_ids
            unique_neighbors = np.unique(sampled_neighbors)
            n_ids.append(unique_neighbors)

            # Create edge index for this layer
            if sampled_sources:
                # Map sources to their position in the combined n_id
                combined = np.concatenate(n_ids[:-1])
                source_map = {node: idx for idx, node in enumerate(combined)}

                mapped_sources = [source_map.get(s, s) for s in sampled_sources]
                edge_index = torch.tensor([mapped_sources, sampled_targets], dtype=torch.long)
            else:
                edge_index = torch.tensor([[], []], dtype=torch.long)

            adjs.append((edge_index, (len(unique_neighbors), batch_size)))

        # Combine all sampled nodes
        all_n_ids = np.concatenate(n_ids)
        unique_n_ids = np.unique(all_n_ids)

        return torch.tensor(unique_n_ids, dtype=torch.long), adjs


class EVRSampler:
    """
    Expected Variance Reduction Sampler.

    Uses importance sampling to reduce variance while maintaining
    unbiased gradients. Samples neighbors with probability proportional
    to their importance (based on edge weights or degree).
    """

    def __init__(
        self,
        edge_index: Tensor,
        num_nodes: int,
        sample_size: int = 20,
        edge_weight: Optional[Tensor] = None,
        use_degree_importance: bool = True,
    ):
        """
        Initialize the EVR sampler.

        Args:
            edge_index: Edge indices [2, num_edges].
            num_nodes: Total number of nodes.
            sample_size: Number of neighbors to sample per node.
            edge_weight: Optional edge weights.
            use_degree_importance: Use node degree for importance weights.
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.sample_size = sample_size
        self.edge_weight = edge_weight
        self.use_degree_importance = use_degree_importance

        # Build adjacency structures
        self._build_structures()

    def _build_structures(self):
        """Build adjacency list and importance weights."""
        self.adj_list = [[] for _ in range(self.num_nodes)]
        self.edge_list = [[] for _ in range(self.num_nodes)]

        edge_index = self.edge_index.cpu().numpy()

        for edge_idx in range(edge_index.shape[1]):
            src, dst = edge_index[0, edge_idx], edge_index[1, edge_idx]
            self.adj_list[dst].append(src)
            self.edge_list[dst].append(edge_idx)

        # Convert to numpy
        self.adj_list = [np.array(n) for n in self.adj_list]
        self.edge_list = [np.array(e) for e in self.edge_list]

        # Compute importance weights
        self._compute_importance()

    def _compute_importance(self):
        """Compute importance sampling weights for each edge."""
        self.importance = []

        for node in range(self.num_nodes):
            neighbors = self.adj_list[node]
            edge_indices = self.edge_list[node]

            if len(neighbors) == 0:
                self.importance.append(np.array([]))
                continue

            if self.edge_weight is not None:
                # Use edge weights as importance
                weights = self.edge_weight[edge_indices].cpu().numpy()
            elif self.use_degree_importance:
                # Use inverse degree as importance
                degrees = np.array([len(self.adj_list[n]) for n in neighbors])
                weights = 1.0 / (degrees + 1)
            else:
                # Uniform importance
                weights = np.ones(len(neighbors))

            # Normalize to probabilities
            weights = weights / (weights.sum() + 1e-8)
            self.importance.append(weights)

    def sample(
        self,
        node_indices: Tensor,
        compute_weights: bool = True,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Sample neighbors with importance weighting.

        Args:
            node_indices: Target node indices [batch_size].
            compute_weights: Whether to compute importance weights for loss.

        Returns:
            Tuple of:
                - sampled_edge_index: Sampled edges [2, num_sampled_edges]
                - sampled_nodes: All involved node indices
                - importance_weights: Weights for unbiased gradients (if compute_weights)
        """
        node_indices = node_indices.cpu().numpy()

        sampled_sources = []
        sampled_targets = []
        sampled_weights = []

        # Map from original node index to position in result
        node_set = set(node_indices)
        node_map = {node: idx for idx, node in enumerate(node_indices)}

        all_nodes = list(node_indices)

        for target_pos, target_node in enumerate(node_indices):
            neighbors = self.adj_list[target_node]
            importance = self.importance[target_node]

            if len(neighbors) == 0:
                continue

            # Determine sample size
            k = min(self.sample_size, len(neighbors))

            if k == len(neighbors):
                # Take all neighbors
                selected_idx = np.arange(len(neighbors))
                selected_weights = np.ones(k) if compute_weights else None
            else:
                # Importance sampling
                selected_idx = np.random.choice(
                    len(neighbors),
                    size=k,
                    replace=False,
                    p=importance,
                )

                if compute_weights:
                    # Compute importance weights for unbiased gradients
                    # w_i = 1 / (k * p_i) where p_i is sampling probability
                    selected_probs = importance[selected_idx]
                    selected_weights = 1.0 / (k * selected_probs + 1e-8)
                    # Normalize weights
                    selected_weights = selected_weights / selected_weights.mean()

            # Add sampled neighbors
            for i, idx in enumerate(selected_idx):
                neighbor = neighbors[idx]

                # Add neighbor to node set if not present
                if neighbor not in node_set:
                    node_map[neighbor] = len(all_nodes)
                    all_nodes.append(neighbor)
                    node_set.add(neighbor)

                sampled_sources.append(node_map[neighbor])
                sampled_targets.append(target_pos)

                if compute_weights and selected_weights is not None:
                    sampled_weights.append(selected_weights[i])

        # Create tensors
        if sampled_sources:
            sampled_edge_index = torch.tensor(
                [sampled_sources, sampled_targets],
                dtype=torch.long,
            )
        else:
            sampled_edge_index = torch.zeros((2, 0), dtype=torch.long)

        sampled_nodes = torch.tensor(all_nodes, dtype=torch.long)

        if compute_weights and sampled_weights:
            importance_weights = torch.tensor(sampled_weights, dtype=torch.float32)
        else:
            importance_weights = None

        return sampled_edge_index, sampled_nodes, importance_weights


class MiniBatchLoader:
    """
    Mini-batch loader for GNN training with neighbor sampling.
    """

    def __init__(
        self,
        edge_index: Tensor,
        num_users: int,
        num_venues: int,
        batch_size: int = 1024,
        num_negatives: int = 4,
        sampler: Optional[EVRSampler] = None,
        shuffle: bool = True,
    ):
        """
        Initialize the loader.

        Args:
            edge_index: User-venue edge indices [2, num_edges].
            num_users: Number of user nodes.
            num_venues: Number of venue nodes.
            batch_size: Batch size.
            num_negatives: Number of negative samples per positive.
            sampler: Optional EVR sampler for neighbor sampling.
            shuffle: Whether to shuffle edges.
        """
        self.edge_index = edge_index
        self.num_users = num_users
        self.num_venues = num_venues
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.sampler = sampler
        self.shuffle = shuffle

        self.num_edges = edge_index.size(1)

        # Build user->venues mapping for negative sampling
        self._build_positive_map()

    def _build_positive_map(self):
        """Build mapping of users to their positive venues."""
        self.user_positives = {}

        edge_index = self.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            user, venue = edge_index[0, i], edge_index[1, i]
            if user not in self.user_positives:
                self.user_positives[user] = set()
            self.user_positives[user].add(venue)

    def _sample_negatives(self, users: np.ndarray) -> np.ndarray:
        """Sample negative venues for given users."""
        negatives = []

        for user in users:
            user_pos = self.user_positives.get(user, set())
            user_negs = []

            while len(user_negs) < self.num_negatives:
                neg = np.random.randint(0, self.num_venues)
                if neg not in user_pos:
                    user_negs.append(neg)

            negatives.append(user_negs)

        return np.array(negatives)

    def __len__(self) -> int:
        return (self.num_edges + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over mini-batches."""
        # Create edge permutation
        if self.shuffle:
            perm = torch.randperm(self.num_edges)
        else:
            perm = torch.arange(self.num_edges)

        # Iterate over batches
        for start in range(0, self.num_edges, self.batch_size):
            end = min(start + self.batch_size, self.num_edges)
            batch_perm = perm[start:end]

            # Get positive edges
            pos_users = self.edge_index[0, batch_perm]
            pos_venues = self.edge_index[1, batch_perm]

            # Sample negatives
            neg_venues = self._sample_negatives(pos_users.numpy())
            neg_venues = torch.tensor(neg_venues, dtype=torch.long)

            batch = {
                "pos_users": pos_users,
                "pos_venues": pos_venues,
                "neg_venues": neg_venues,
                "batch_size": end - start,
            }

            # Add sampled subgraph if sampler provided
            if self.sampler is not None:
                # Sample neighbors for batch users
                unique_users = torch.unique(pos_users)
                sampled_edge_index, sampled_nodes, importance_weights = \
                    self.sampler.sample(unique_users)

                batch["sampled_edge_index"] = sampled_edge_index
                batch["sampled_nodes"] = sampled_nodes
                batch["importance_weights"] = importance_weights

            yield batch


class TrainTestSplit:
    """Split edges into train/val/test sets."""

    def __init__(
        self,
        edge_index: Tensor,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize the splitter.

        Args:
            edge_index: Edge indices [2, num_edges].
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            seed: Random seed.
        """
        self.edge_index = edge_index
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        self._split()

    def _split(self):
        """Perform the split."""
        num_edges = self.edge_index.size(1)

        # Create random permutation
        np.random.seed(self.seed)
        perm = np.random.permutation(num_edges)

        # Calculate split points
        train_end = int(num_edges * self.train_ratio)
        val_end = int(num_edges * (self.train_ratio + self.val_ratio))

        # Create masks
        self.train_idx = torch.tensor(perm[:train_end], dtype=torch.long)
        self.val_idx = torch.tensor(perm[train_end:val_end], dtype=torch.long)
        self.test_idx = torch.tensor(perm[val_end:], dtype=torch.long)

        logger.info(
            f"Split edges: train={len(self.train_idx)}, "
            f"val={len(self.val_idx)}, test={len(self.test_idx)}"
        )

    def get_train_edges(self) -> Tensor:
        return self.edge_index[:, self.train_idx]

    def get_val_edges(self) -> Tensor:
        return self.edge_index[:, self.val_idx]

    def get_test_edges(self) -> Tensor:
        return self.edge_index[:, self.test_idx]
