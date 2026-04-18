#!/usr/bin/env python3
"""
Train Graph Neural Network for Recommendations.

Usage:
    python scripts/train_gnn.py --epochs 100 --hidden-dim 128
    python scripts/train_gnn.py --use-gbce --gbce-t 0.8
"""
import argparse
import gc
import logging
import os
from pathlib import Path

# Prevent CUDA memory fragmentation on Windows (must be set before torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch

from src.config import get_config
from src.config.settings import PROCESSED_DATA_DIR, MODEL_DIR, CHECKPOINT_DIR
from src.models.gnn import (
    HeteroGraphBuilder,
    LTGNN,
    GNNTrainer,
    TrainTestSplit,
    build_recommendation_graph,
    AugmentedLTGNN,
    build_user_user_edges,
)
from src.utils.helpers import setup_logging, set_seed, get_device

logger = logging.getLogger(__name__)


def load_embeddings(model_dir: Path) -> dict:
    """Load pre-computed embeddings from previous phases."""
    embeddings = {}

    # User MBTI embeddings (from Phase 2)
    user_emb_path = PROCESSED_DATA_DIR / "user_mbti_embeddings.npy"
    if user_emb_path.exists():
        embeddings["user"] = np.load(user_emb_path)
        logger.info(f"Loaded user embeddings: {embeddings['user'].shape}")
    else:
        logger.warning("User embeddings not found, will use random initialization")

    # Venue embeddings (from Phase 3)
    venue_emb_path = model_dir / "bertopic" / "venue_embeddings.npy"
    if venue_emb_path.exists():
        embeddings["venue"] = np.load(venue_emb_path)
        logger.info(f"Loaded venue embeddings: {embeddings['venue'].shape}")
    else:
        logger.warning("Venue embeddings not found, will use random initialization")

    return embeddings


def _reconcile_embeddings(
    embeddings: dict, key: str, n_nodes: int, default_dim: int
) -> dict:
    """Ensure embeddings[key] has exactly n_nodes rows.

    Three cases:
    - Missing: create random init with default_dim.
    - Too few rows: pad with random init (preserves loaded dim).
    - Too many rows: truncate (assumes row order matches node order).
    """
    if key not in embeddings:
        logger.info(f"Creating random {key} embeddings ({n_nodes}, {default_dim})")
        embeddings[key] = np.random.randn(
            n_nodes, default_dim
        ).astype(np.float32) * 0.1
    else:
        loaded = embeddings[key]
        dim = loaded.shape[1]
        if loaded.shape[0] < n_nodes:
            pad = np.random.randn(
                n_nodes - loaded.shape[0], dim
            ).astype(np.float32) * 0.1
            embeddings[key] = np.vstack([loaded, pad])
            logger.info(f"Padded {key} embeddings: {loaded.shape[0]} -> {n_nodes}")
        elif loaded.shape[0] > n_nodes:
            embeddings[key] = loaded[:n_nodes]
            logger.info(f"Truncated {key} embeddings: {loaded.shape[0]} -> {n_nodes}")
    return embeddings


def main(args):
    """Main training function."""
    setup_logging(level=logging.INFO)
    config = get_config()
    set_seed(42)

    # Get device
    device = get_device(args.device)

    # Load data
    logger.info("Loading data...")

    # Load reviews for interactions
    train_reviews = pd.read_parquet(PROCESSED_DATA_DIR / "train_reviews.parquet")
    val_reviews = pd.read_parquet(PROCESSED_DATA_DIR / "val_reviews.parquet")

    logger.info(f"Train interactions: {len(train_reviews)}")
    logger.info(f"Val interactions: {len(val_reviews)}")

    # Load user profiles
    user_profiles_path = PROCESSED_DATA_DIR / "user_mbti_profiles.parquet"
    if user_profiles_path.exists():
        user_profiles = pd.read_parquet(user_profiles_path)
    else:
        # Create from reviews
        user_profiles = pd.DataFrame({
            "user_id": train_reviews["user_id"].unique()
        })
    logger.info(f"Users: {len(user_profiles)}")

    # Load venue topics
    venue_topics_path = MODEL_DIR / "bertopic" / "venue_topics.parquet"
    if venue_topics_path.exists():
        venue_topics = pd.read_parquet(venue_topics_path)
    else:
        # Create from reviews
        venue_topics = pd.DataFrame({
            "venue_id": train_reviews["business_id"].unique()
        })
        venue_topics["topic"] = 0
    logger.info(f"Venues: {len(venue_topics)}")

    # Load embeddings
    embeddings = load_embeddings(MODEL_DIR)

    # Reconcile embeddings with actual node counts
    n_users = len(user_profiles)
    n_venues = len(venue_topics)

    embeddings = _reconcile_embeddings(
        embeddings, "user", n_users, args.embedding_dim
    )
    embeddings = _reconcile_embeddings(
        embeddings, "venue", n_venues, args.embedding_dim
    )

    logger.info(f"User embeddings: {embeddings['user'].shape}")
    logger.info(f"Venue embeddings: {embeddings['venue'].shape}")

    # Build graph
    logger.info("Building graph...")

    builder = HeteroGraphBuilder(device=device)

    # Add nodes
    builder.add_user_nodes(
        user_profiles["user_id"].tolist(),
        embeddings["user"],
    )

    builder.add_venue_nodes(
        venue_topics["venue_id"].tolist(),
        embeddings["venue"],
    )

    # Add edges from training interactions
    builder.add_user_venue_edges(
        train_reviews["user_id"].tolist(),
        train_reviews["business_id"].tolist(),
        train_reviews["stars"].tolist() if "stars" in train_reviews.columns else None,
    )

    # Get statistics
    stats = builder.get_statistics()
    logger.info(f"Graph statistics: {stats}")

    # Get train/val edges
    train_edge_index = builder.edges[("user", "visits", "venue")].edge_index

    # Create validation edges from val_reviews (vectorized)
    user_node = builder.nodes["user"]
    venue_node = builder.nodes["venue"]

    val_user_idx = np.array([
        user_node.id_to_idx.get(uid, -1)
        for uid in val_reviews["user_id"].values
    ])
    val_venue_idx = np.array([
        venue_node.id_to_idx.get(vid, -1)
        for vid in val_reviews["business_id"].values
    ])
    valid = (val_user_idx >= 0) & (val_venue_idx >= 0)
    val_edge_index = torch.tensor(
        [val_user_idx[valid], val_venue_idx[valid]], dtype=torch.long
    )

    logger.info(f"Train edges: {train_edge_index.size(1)}")
    logger.info(f"Val edges: {val_edge_index.size(1)}")

    # Create model
    logger.info("Creating LTGNN model...")

    base_model = LTGNN(
        user_input_dim=embeddings["user"].shape[1],
        venue_input_dim=embeddings["venue"].shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_iterations=args.num_iterations,
        dropout=args.dropout,
    )

    # V2: Optionally augment with User-User edges for cold-start mitigation
    if args.use_user_edges:
        logger.info("Building User-User edges from MBTI similarity...")
        model = AugmentedLTGNN.from_base(
            base_model,
            user_embeddings=embeddings["user"],
            similarity_threshold=args.user_edge_threshold,
            max_neighbors=args.user_edge_max_neighbors,
            user_user_weight=args.user_edge_weight,
        )
        logger.info("Using AugmentedLTGNN with User-User edges")
    else:
        model = base_model

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Configure training
    config.gnn.num_epochs = args.epochs
    config.gnn.batch_size = args.batch_size
    config.gnn.learning_rate = args.learning_rate
    config.gnn.hidden_dim = args.hidden_dim
    config.gnn.embedding_dim = args.embedding_dim

    # Create trainer
    user_features = torch.tensor(embeddings["user"], dtype=torch.float32)
    venue_features = torch.tensor(embeddings["venue"], dtype=torch.float32)

    trainer = GNNTrainer(
        model=model,
        user_features=user_features,
        venue_features=venue_features,
        train_edge_index=train_edge_index,
        val_edge_index=val_edge_index,
        config=config.gnn,
        device=device,
        use_gbce=args.use_gbce,
        gbce_t=args.gbce_t,
    )

    # Free stale allocations before training
    gc.collect()
    torch.cuda.empty_cache()

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    # Get final embeddings (memory-safe: chunked projection, CPU output)
    logger.info("Extracting final embeddings for ALL nodes...")
    user_emb, venue_emb = trainer.get_embeddings()

    # Validate shapes BEFORE saving - this is the critical gate that
    # prevents Phase 5 from crashing with truncated embeddings
    n_users = user_features.size(0)
    n_venues = venue_features.size(0)
    assert user_emb.size(0) == n_users, (
        f"FATAL: user embeddings have {user_emb.size(0)} rows, "
        f"expected {n_users}"
    )
    assert venue_emb.size(0) == n_venues, (
        f"FATAL: venue embeddings have {venue_emb.size(0)} rows, "
        f"expected {n_venues}"
    )

    logger.info(
        f"User embeddings:  {tuple(user_emb.shape)} "
        f"(expected ({n_users}, {args.embedding_dim}))"
    )
    logger.info(
        f"Venue embeddings: {tuple(venue_emb.shape)} "
        f"(expected ({n_venues}, {args.embedding_dim}))"
    )

    # Save embeddings (already on CPU from get_embeddings)
    output_dir = MODEL_DIR / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "user_gnn_embeddings.npy", user_emb.numpy())
    np.save(output_dir / "venue_gnn_embeddings.npy", venue_emb.numpy())

    # Also save to gnn_hetero/ so evaluate_hybrid.py can find them
    hetero_dir = MODEL_DIR / "gnn_hetero"
    hetero_dir.mkdir(parents=True, exist_ok=True)
    np.save(hetero_dir / "user_embeddings.npy", user_emb.numpy())
    np.save(hetero_dir / "venue_embeddings.npy", venue_emb.numpy())

    # Save graph
    builder.save(output_dir / "graph")

    logger.info(f"\nTraining complete!")
    logger.info(f"Best NDCG@10: {history['best_metric']:.4f}")
    logger.info(f"Embeddings saved to: {output_dir}")
    logger.info(f"Also saved to: {hetero_dir} (for evaluate_hybrid.py)")
    logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN model")

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (GNNs converge fast)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Training batch size (larger = fewer full-graph passes)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden layer dimension",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Output embedding dimension",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Fixed-point iterations (5 is sufficient, 10 is overkill)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate",
    )
    parser.add_argument(
        "--use-gbce",
        action="store_true",
        default=True,
        help="Use gBCE loss for calibration",
    )
    parser.add_argument(
        "--no-gbce",
        action="store_false",
        dest="use_gbce",
        help="Use standard BCE loss",
    )
    parser.add_argument(
        "--gbce-t",
        type=float,
        default=0.8,
        help="gBCE temperature parameter",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device for training",
    )
    # V2: User-User edge augmentation for cold-start
    parser.add_argument(
        "--use-user-edges",
        action="store_true",
        default=False,
        help="Add User-User edges from MBTI similarity (cold-start fix)",
    )
    parser.add_argument(
        "--user-edge-threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for User-User edges",
    )
    parser.add_argument(
        "--user-edge-max-neighbors",
        type=int,
        default=10,
        help="Max User-User neighbors per user",
    )
    parser.add_argument(
        "--user-edge-weight",
        type=float,
        default=0.5,
        help="Weight for User-User edges in adjacency matrix (0-1)",
    )

    args = parser.parse_args()
    main(args)
