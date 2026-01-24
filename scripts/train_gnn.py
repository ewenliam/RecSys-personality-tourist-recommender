#!/usr/bin/env python3
"""
Train Graph Neural Network for Recommendations.

Usage:
    python scripts/train_gnn.py --epochs 100 --hidden-dim 128
    python scripts/train_gnn.py --use-gbce --gbce-t 0.8
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import get_config, PROCESSED_DATA_DIR, MODEL_DIR, CHECKPOINT_DIR
from src.models.gnn import (
    HeteroGraphBuilder,
    LTGNN,
    GNNTrainer,
    TrainTestSplit,
    build_recommendation_graph,
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

    # Create embeddings if not available
    if "user" not in embeddings:
        logger.info("Creating random user embeddings")
        embeddings["user"] = np.random.randn(
            len(user_profiles), args.embedding_dim
        ).astype(np.float32) * 0.1

    if "venue" not in embeddings:
        logger.info("Creating random venue embeddings")
        embeddings["venue"] = np.random.randn(
            len(venue_topics), args.embedding_dim
        ).astype(np.float32) * 0.1

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

    # Create validation edges from val_reviews
    val_user_idx = []
    val_venue_idx = []
    user_node = builder.nodes["user"]
    venue_node = builder.nodes["venue"]

    for _, row in val_reviews.iterrows():
        user_idx = user_node.get_idx(row["user_id"])
        venue_idx = venue_node.get_idx(row["business_id"])
        if user_idx is not None and venue_idx is not None:
            val_user_idx.append(user_idx)
            val_venue_idx.append(venue_idx)

    val_edge_index = torch.tensor([val_user_idx, val_venue_idx], dtype=torch.long)

    logger.info(f"Train edges: {train_edge_index.size(1)}")
    logger.info(f"Val edges: {val_edge_index.size(1)}")

    # Create model
    logger.info("Creating LTGNN model...")

    model = LTGNN(
        user_input_dim=embeddings["user"].shape[1],
        venue_input_dim=embeddings["venue"].shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_iterations=args.num_iterations,
        dropout=args.dropout,
    )

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

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    # Get final embeddings
    logger.info("Extracting final embeddings...")
    user_emb, venue_emb = trainer.get_embeddings()

    # Save embeddings
    output_dir = MODEL_DIR / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "user_gnn_embeddings.npy", user_emb.cpu().numpy())
    np.save(output_dir / "venue_gnn_embeddings.npy", venue_emb.cpu().numpy())

    # Save graph
    builder.save(output_dir / "graph")

    logger.info(f"\nTraining complete!")
    logger.info(f"Best NDCG@10: {history['best_metric']:.4f}")
    logger.info(f"Embeddings saved to: {output_dir}")
    logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN model")

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size",
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
        default=10,
        help="Fixed-point iterations",
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

    args = parser.parse_args()
    main(args)
