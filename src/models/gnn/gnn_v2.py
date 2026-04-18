"""
GNN V2: User-User Edge Augmentation for Cold-Start Mitigation.

The original LTGNN uses a bipartite graph (user-venue edges only).
With 0.0175% density and median 8 reviews/user, cold-start users
get weak embeddings because they have too few edges for meaningful
message passing.

This module adds User-User similarity edges (based on MBTI profile
cosine similarity) so that users with similar personalities share
information even before visiting the same venues.

The augmented adjacency matrix is:
    [U-U block | U-V block]
    [V-U block |   0     ]

where U-U block = thresholded cosine similarity between MBTI embeddings.
"""
import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .ltgnn import build_sparse_adj

logger = logging.getLogger(__name__)


def build_user_user_edges(
    user_embeddings: np.ndarray,
    threshold: float = 0.8,
    max_neighbors: int = 10,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build User-User edges from MBTI embedding cosine similarity.

    Memory-safe: processes users in batches to avoid creating a
    [235K, 235K] similarity matrix (220 GB).

    Args:
        user_embeddings: MBTI embeddings [n_users, dim].
        threshold: Minimum cosine similarity to create an edge.
        max_neighbors: Maximum neighbors per user (prevents hubs).
        batch_size: Users processed per batch.

    Returns:
        Tuple of (src_indices, dst_indices) as numpy arrays.
    """
    from sklearn.preprocessing import normalize

    n_users = user_embeddings.shape[0]
    user_emb_norm = normalize(user_embeddings, axis=1)

    src_list = []
    dst_list = []

    logger.info(
        f"Building User-User edges: {n_users} users, "
        f"threshold={threshold}, max_neighbors={max_neighbors}"
    )

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        batch = user_emb_norm[start:end]  # [batch, dim]

        # Cosine similarity between batch and ALL users
        # [batch, n_users] - each row is sim(batch_user, all_users)
        sim = batch @ user_emb_norm.T

        for local_i in range(batch.shape[0]):
            global_i = start + local_i

            # Zero out self-similarity
            sim[local_i, global_i] = -1.0

            # Find top-k neighbors above threshold
            row = sim[local_i]
            above_thresh = np.where(row >= threshold)[0]

            if len(above_thresh) == 0:
                continue

            # Cap to max_neighbors (keep highest similarity)
            if len(above_thresh) > max_neighbors:
                top_idx = np.argpartition(
                    row[above_thresh], -max_neighbors
                )[-max_neighbors:]
                above_thresh = above_thresh[top_idx]

            for j in above_thresh:
                src_list.append(global_i)
                dst_list.append(int(j))

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)

    # Make symmetric (if A->B, add B->A)
    full_src = np.concatenate([src, dst])
    full_dst = np.concatenate([dst, src])

    # Deduplicate
    edges = np.stack([full_src, full_dst])
    edges = np.unique(edges, axis=1)

    logger.info(f"Built {edges.shape[1]} User-User edges")
    return edges[0], edges[1]


def build_augmented_sparse_adj(
    user_venue_edge_index: Tensor,
    num_users: int,
    num_venues: int,
    user_user_edges: Optional[Tuple[Tensor, Tensor]] = None,
    user_user_weight: float = 0.5,
    edge_weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Build sparse adjacency with User-User edges augmented.

    The node ordering is: [users (0..num_users-1), venues (num_users..num_nodes-1)]

    The adjacency has three blocks:
      - User-Venue (bipartite, from original graph)
      - Venue-User (reverse of above)
      - User-User (from MBTI similarity)

    User-User edges are down-weighted to prevent them from
    dominating the message passing signal.

    Args:
        user_venue_edge_index: [2, E] user-venue edges (raw, not offset).
        num_users: Number of user nodes.
        num_venues: Number of venue nodes.
        user_user_edges: Tuple of (src, dst) tensors for user-user edges.
        user_user_weight: Weight for user-user edges (0-1).
        edge_weight: Optional weights for user-venue edges.

    Returns:
        Sparse COO adjacency tensor [num_nodes, num_nodes].
    """
    num_nodes = num_users + num_venues
    device = user_venue_edge_index.device

    # Build user-venue bidirectional edges (offset venues)
    uv = user_venue_edge_index.clone()
    uv[1] = uv[1] + num_users  # Offset venue indices
    vu = uv.flip(0)
    all_edges = [uv, vu]

    if edge_weight is not None:
        all_weights = [edge_weight, edge_weight]
    else:
        all_weights = [
            torch.ones(uv.size(1), device=device),
            torch.ones(vu.size(1), device=device),
        ]

    # Add user-user edges (no offset needed, both are in user space)
    if user_user_edges is not None:
        uu_src, uu_dst = user_user_edges
        uu_edge_index = torch.stack([uu_src, uu_dst]).to(device)
        all_edges.append(uu_edge_index)
        # Down-weight user-user edges
        uu_weights = torch.full(
            (uu_edge_index.size(1),), user_user_weight, device=device
        )
        all_weights.append(uu_weights)

    full_edge_index = torch.cat(all_edges, dim=1)
    full_weights = torch.cat(all_weights)

    # Symmetric normalization: D^{-0.5} A D^{-0.5}
    row, col = full_edge_index
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, row, full_weights)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    norm_weights = deg_inv_sqrt[row] * full_weights * deg_inv_sqrt[col]

    adj = torch.sparse_coo_tensor(
        full_edge_index, norm_weights, (num_nodes, num_nodes)
    ).coalesce()

    return adj


class AugmentedLTGNN(torch.nn.Module):
    """
    LTGNN with User-User edge augmentation.

    Wraps the existing LTGNN architecture but replaces the adjacency
    construction to include User-User similarity edges. All other
    components (projections, FixedPointLayer, link predictor) are
    inherited from the base LTGNN.

    Usage:
        model = AugmentedLTGNN.from_base(base_ltgnn, user_user_edges)
        user_emb, venue_emb = model.encode(user_x, venue_x, edge_index)
    """

    def __init__(
        self,
        base_model: "LTGNN",
        user_user_src: Optional[Tensor] = None,
        user_user_dst: Optional[Tensor] = None,
        user_user_weight: float = 0.5,
    ):
        super().__init__()
        # Share all parameters with the base model
        self.user_projection = base_model.user_projection
        self.venue_projection = base_model.venue_projection
        self.gnn = base_model.gnn
        self.output_projection = base_model.output_projection
        self.link_predictor = base_model.link_predictor

        self.hidden_dim = base_model.hidden_dim
        self.embedding_dim = base_model.embedding_dim

        # User-User edges
        if user_user_src is not None and user_user_dst is not None:
            self.register_buffer("uu_src", user_user_src)
            self.register_buffer("uu_dst", user_user_dst)
        else:
            self.uu_src = None
            self.uu_dst = None

        self.user_user_weight = user_user_weight
        self._cached_adj = None
        self._cached_edge_id = None

    @classmethod
    def from_base(
        cls,
        base_model: "LTGNN",
        user_embeddings: np.ndarray,
        similarity_threshold: float = 0.8,
        max_neighbors: int = 10,
        user_user_weight: float = 0.5,
    ) -> "AugmentedLTGNN":
        """
        Create an AugmentedLTGNN from a trained base LTGNN.

        Args:
            base_model: Trained LTGNN model.
            user_embeddings: MBTI embeddings for user-user similarity.
            similarity_threshold: Cosine sim threshold for edges.
            max_neighbors: Max user-user neighbors per user.
            user_user_weight: Weight for user-user edges in adj matrix.

        Returns:
            AugmentedLTGNN instance sharing base model's parameters.
        """
        src, dst = build_user_user_edges(
            user_embeddings,
            threshold=similarity_threshold,
            max_neighbors=max_neighbors,
        )

        return cls(
            base_model=base_model,
            user_user_src=torch.from_numpy(src).long(),
            user_user_dst=torch.from_numpy(dst).long(),
            user_user_weight=user_user_weight,
        )

    def encode(
        self,
        user_x: Tensor,
        venue_x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode with augmented adjacency (user-venue + user-user).

        Same signature as LTGNN.encode() for drop-in replacement.
        """
        num_users = user_x.size(0)
        num_venues = venue_x.size(0)

        # Project to hidden dim
        user_h = self.user_projection(user_x)
        venue_h = self.venue_projection(venue_x)
        x = torch.cat([user_h, venue_h], dim=0)

        # Build augmented adjacency (cached after first call)
        if self._cached_adj is None:
            uu_edges = None
            if self.uu_src is not None:
                uu_edges = (self.uu_src, self.uu_dst)

            self._cached_adj = build_augmented_sparse_adj(
                edge_index, num_users, num_venues,
                user_user_edges=uu_edges,
                user_user_weight=self.user_user_weight,
                edge_weight=edge_weight,
            )

        # GNN message passing
        h = self.gnn(x, self._cached_adj)

        # Output projection
        out = self.output_projection(h)

        user_emb = out[:num_users]
        venue_emb = out[num_users:]
        return user_emb, venue_emb

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
        """Memory-safe encoding for inference, identical contract to LTGNN.encode_inference."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        num_users = user_x.size(0)
        num_venues = venue_x.size(0)

        # Project user features in chunks
        user_h_parts = []
        for start in range(0, num_users, chunk_size):
            end = min(start + chunk_size, num_users)
            chunk = user_x[start:end].to(device)
            user_h_parts.append(self.user_projection(chunk).cpu())
            del chunk
        user_h = torch.cat(user_h_parts, dim=0)
        del user_h_parts

        # Project venue features in chunks
        venue_h_parts = []
        for start in range(0, num_venues, chunk_size):
            end = min(start + chunk_size, num_venues)
            chunk = venue_x[start:end].to(device)
            venue_h_parts.append(self.venue_projection(chunk).cpu())
            del chunk
        venue_h = torch.cat(venue_h_parts, dim=0)
        del venue_h_parts

        # Move compact hidden features to device for GNN
        x = torch.cat([user_h, venue_h], dim=0).to(device)
        del user_h, venue_h

        # Build augmented adjacency (cached after first call)
        if self._cached_adj is None:
            uu_edges = None
            if self.uu_src is not None:
                uu_edges = (self.uu_src, self.uu_dst)
            edge_index_dev = edge_index.to(device)
            ew_dev = edge_weight.to(device) if edge_weight is not None else None
            self._cached_adj = build_augmented_sparse_adj(
                edge_index_dev, num_users, num_venues,
                user_user_edges=uu_edges,
                user_user_weight=self.user_user_weight,
                edge_weight=ew_dev,
            )

        # GNN message passing
        h = self.gnn(x, self._cached_adj)
        del x

        # Output projection in chunks, results to CPU
        user_emb_parts = []
        for start in range(0, num_users, chunk_size):
            end = min(start + chunk_size, num_users)
            user_emb_parts.append(self.output_projection(h[start:end]).cpu())
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

    def recommend(
        self,
        user_x: Tensor,
        venue_x: Tensor,
        edge_index: Tensor,
        user_idx: int,
        k: int = 10,
        exclude_venues=None,
        edge_weight=None,
        cached_embeddings=None,
    ) -> Tuple[Tensor, Tensor]:
        """Top-k venue recommendations for a user (same contract as LTGNN.recommend)."""
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
            scores = torch.sigmoid(self.link_predictor(combined).squeeze(-1))

            if exclude_venues:
                for vid in exclude_venues:
                    scores[vid] = -float("inf")

            top_scores, top_indices = torch.topk(scores, k)

        return top_indices, top_scores

    def predict_from_embeddings(
        self,
        user_emb: Tensor,
        venue_emb: Tensor,
        user_indices: Tensor,
        venue_indices: Tensor,
    ) -> Tensor:
        """Predict from pre-computed embeddings (same as base LTGNN)."""
        u = user_emb[user_indices]
        v = venue_emb[venue_indices]
        combined = torch.cat([u, v], dim=1)
        return self.link_predictor(combined).squeeze(-1)

    def invalidate_cache(self):
        """Invalidate the cached adjacency matrix."""
        self._cached_adj = None
        self._cached_edge_id = None
