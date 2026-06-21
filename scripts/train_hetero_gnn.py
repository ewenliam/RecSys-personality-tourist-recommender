#!/usr/bin/env python3
"""
Phase 3: Heterogeneous GNN Training.

Trains a HeteroGNN on the user-venue interaction graph with:
  - User nodes initialized with BERT-MBTI CLS embeddings
  - Venue nodes initialized with BERTopic topic distributions
  - GraphSAGE-based heterogeneous message passing
  - gBCE loss for calibrated link prediction

Usage:
    python scripts/train_hetero_gnn.py
    python scripts/train_hetero_gnn.py --epochs 50 --hidden-dim 128 --quick
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import (
    get_config,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    PROJECT_ROOT,
)
from src.models.gnn.graph_builder import HeteroGraphBuilder
from src.models.gnn.hetero_gnn import HeteroGNN
from src.models.gnn.hetero_trainer import HeteroGNNTrainer
from src.utils.helpers import setup_logging, set_seed, get_device

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"


def load_embeddings(model_dir: Path) -> dict:
    """Load pre-computed embeddings from previous phases."""
    embeddings = {}

    # User MBTI embeddings (from Phase 2 or inference)
    user_emb_path = model_dir / "gnn" / "user_gnn_embeddings.npy"
    if user_emb_path.exists():
        embeddings["user"] = np.load(user_emb_path)
        logger.info(f"Loaded user embeddings: {embeddings['user'].shape}")
    else:
        logger.warning(f"User embeddings not found at {user_emb_path}")

    # Venue topic embeddings (from Phase 2 BERTopic)
    venue_emb_path = model_dir / "bertopic_mbti" / "venue_embeddings.npy"
    if not venue_emb_path.exists():
        venue_emb_path = model_dir / "bertopic" / "venue_embeddings.npy"
    if venue_emb_path.exists():
        embeddings["venue"] = np.load(venue_emb_path)
        logger.info(f"Loaded venue embeddings: {embeddings['venue'].shape}")
    else:
        logger.warning(f"Venue embeddings not found")

    # Venue topics mapping
    topics_path = model_dir / "bertopic_mbti" / "venue_topics.parquet"
    if not topics_path.exists():
        topics_path = model_dir / "bertopic" / "venue_topics.parquet"
    if topics_path.exists():
        embeddings["venue_topics"] = pd.read_parquet(topics_path)
        logger.info(f"Loaded venue topics: {len(embeddings['venue_topics'])} venues")

    return embeddings


def load_interactions(data_dir: Path) -> pd.DataFrame:
    """Load user-venue interaction data."""
    train_path = data_dir / "train_reviews.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {train_path}\n"
            "Run 'python scripts/prepare_data.py' first."
        )

    dfs = []
    for split in ["train_reviews.parquet", "val_reviews.parquet", "test_reviews.parquet"]:
        split_path = data_dir / split
        if split_path.exists():
            dfs.append(pd.read_parquet(split_path))

    df = pd.concat(dfs, ignore_index=True)

    user_col = "user_id" if "user_id" in df.columns else df.columns[0]
    biz_col = "business_id" if "business_id" in df.columns else "venue_id"
    rating_col = "stars" if "stars" in df.columns else None

    logger.info(
        f"Loaded {len(df)} interactions: "
        f"{df[user_col].nunique()} users, {df[biz_col].nunique()} venues"
    )
    return df, user_col, biz_col, rating_col


def build_graph(
    interactions: pd.DataFrame,
    user_col: str,
    biz_col: str,
    rating_col: str,
    embeddings: dict,
) -> tuple[HeteroGraphBuilder, dict]:
    """
    Build the heterogeneous graph from interactions and embeddings.

    If pre-computed embeddings are available, uses them.
    Otherwise generates random embeddings for demonstration.
    """
    # Get unique users and venues
    unique_users = interactions[user_col].unique()
    unique_venues = interactions[biz_col].unique()

    user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
    venue_id_map = {vid: idx for idx, vid in enumerate(unique_venues)}

    # User features: review statistics provide behavioral signal that
    # differentiates users beyond graph structure alone.
    logger.info("Building user features from review statistics")
    rating_col_name = rating_col if rating_col else None

    user_stats = interactions.groupby(user_col).agg(
        n_reviews=(biz_col, "count"),
        **({"mean_stars": (rating_col_name, "mean"),
            "std_stars": (rating_col_name, "std")} if rating_col_name else {}),
    ).reindex(unique_users).fillna(0)

    feat_cols = [np.log1p(user_stats["n_reviews"].values).reshape(-1, 1).astype(np.float32)]
    if rating_col_name:
        feat_cols.append(
            ((user_stats["mean_stars"].values - 3.0) / 2.0).reshape(-1, 1).astype(np.float32)
        )
        feat_cols.append(
            user_stats["std_stars"].fillna(0).values.reshape(-1, 1).astype(np.float32)
        )

    user_features = np.hstack(feat_cols)
    logger.info(f"User features: {user_features.shape} (review stats)")

    # Venue features: reindex BERTopic embeddings to match pd.unique() ordering.
    # The BERTopic venue_topics.parquet maps BERTopic row indices to string IDs.
    # We scatter those rows into the correct positions according to venue_id_map.
    if "venue" in embeddings and "venue_topics" in embeddings:
        raw_venue_embs = embeddings["venue"]          # [n_btopic, 768]
        btopic_venue_ids = embeddings["venue_topics"]["venue_id"].tolist()
        d = raw_venue_embs.shape[1]
        venue_features = np.zeros((len(unique_venues), d), dtype=np.float32)
        matched = 0
        for b_idx, vid in enumerate(btopic_venue_ids):
            gnn_idx = venue_id_map.get(vid)
            if gnn_idx is not None and b_idx < len(raw_venue_embs):
                venue_features[gnn_idx] = raw_venue_embs[b_idx]
                matched += 1
        logger.info(f"Venue features reindexed: {matched}/{len(btopic_venue_ids)} matched, shape={venue_features.shape}")
    else:
        logger.info("Generating random venue features for demonstration")
        venue_features = np.random.randn(len(unique_venues), 768).astype(np.float32)
        venue_features = venue_features / (np.linalg.norm(venue_features, axis=1, keepdims=True) + 1e-8)

    # Build graph
    builder = HeteroGraphBuilder()
    builder.add_user_nodes(list(unique_users), user_features)
    builder.add_venue_nodes(list(unique_venues), venue_features)

    ratings = interactions[rating_col].tolist() if rating_col else None
    builder.add_user_venue_edges(
        interactions[user_col].tolist(),
        interactions[biz_col].tolist(),
        ratings=ratings,
    )

    stats = builder.get_statistics()
    logger.info(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges")

    mappings = {
        "user_id_map": user_id_map,
        "venue_id_map": venue_id_map,
    }

    return builder, mappings


def split_edges(
    builder: HeteroGraphBuilder,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split user-venue edges into train/val/test."""
    edge_type = HeteroGraphBuilder.USER_VISITS_VENUE
    edge_data = builder.edges[edge_type]
    edge_index = edge_data.edge_index

    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)

    n_train = int(num_edges * train_ratio)
    n_val = int(num_edges * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_edges = edge_index[:, train_idx]
    val_edges = edge_index[:, val_idx]
    test_edges = edge_index[:, test_idx]

    logger.info(
        f"Edge split: train={train_edges.size(1)}, "
        f"val={val_edges.size(1)}, test={test_edges.size(1)}"
    )

    return train_edges, val_edges, test_edges


def main(args):
    """Train the heterogeneous GNN."""
    setup_logging(level=logging.INFO)
    set_seed(42)

    config = get_config()
    if args.epochs:
        config.gnn.num_epochs = args.epochs
    if args.hidden_dim:
        config.gnn.hidden_dim = args.hidden_dim
    if args.quick:
        config.gnn.num_epochs = min(config.gnn.num_epochs, 10)
        config.gnn.batch_size = 512

    device = get_device(args.device)

    # Load data
    logger.info("Loading embeddings and interactions...")
    embeddings = load_embeddings(MODEL_DIR)
    interactions, user_col, biz_col, rating_col = load_interactions(PROCESSED_DATA_DIR)

    # Build graph
    logger.info("Building heterogeneous graph...")
    builder, mappings = build_graph(interactions, user_col, biz_col, rating_col, embeddings)

    # Prepare node features and edge indices for HeteroGNN
    x_dict = {
        "user": builder.nodes["user"].features,
        "venue": builder.nodes["venue"].features,
    }

    # Collect all edge types (for message passing)
    edge_index_dict = {}
    for edge_type, edge_data in builder.edges.items():
        edge_index_dict[edge_type] = edge_data.edge_index

    # Split edges
    train_edges, val_edges, test_edges = split_edges(builder)

    # Determine input dimensions
    node_input_dims = {
        "user": x_dict["user"].size(1),
        "venue": x_dict["venue"].size(1),
    }

    edge_types = list(edge_index_dict.keys())

    # Create model
    # 1 layer prevents over-smoothing on this dense bipartite graph.
    # With mean degree 41.7, a 2-layer SAGEConv covers ~1740 venues per user
    # (41.7^2), mixing embeddings across essentially the whole graph and
    # causing all embeddings to collapse to the global mean.
    num_layers = args.num_layers if args.num_layers else 1
    model = HeteroGNN(
        node_input_dims=node_input_dims,
        edge_types=edge_types,
        hidden_dim=config.gnn.hidden_dim,
        embedding_dim=config.gnn.embedding_dim,
        num_layers=num_layers,
        dropout=config.gnn.dropout,
        config=config.gnn,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"HeteroGNN: {n_params:,} trainable parameters")

    # Create trainer
    num_venues = len(builder.nodes["venue"])
    trainer = HeteroGNNTrainer(
        model=model,
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        train_user_venue_edges=train_edges,
        val_user_venue_edges=val_edges,
        num_venues=num_venues,
        config=config.gnn,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR / "gnn_hetero",
        use_gbce=True,
        gbce_t=config.hybrid.gbce_calibration_t,
        num_negatives=8,  # 4 hard (popularity-weighted) + 4 easy (uniform)
    )

    # Train
    logger.info("=" * 60)
    logger.info("Training Heterogeneous GNN")
    logger.info("=" * 60)

    history = trainer.train()

    # Extract and save embeddings
    logger.info("Extracting learned embeddings...")
    learned_embs = trainer.get_embeddings()

    save_dir = MODEL_DIR / "gnn_hetero"
    save_dir.mkdir(parents=True, exist_ok=True)

    for node_type, emb in learned_embs.items():
        emb_np = emb.cpu().numpy()
        np.save(save_dir / f"{node_type}_embeddings.npy", emb_np)
        logger.info(f"Saved {node_type} embeddings: {emb_np.shape}")

    # Save mappings
    torch.save(mappings, save_dir / "id_mappings.pt")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(edge_index=test_edges.to(device))
    logger.info(
        f"Test Results: AUC={test_metrics.auc:.4f}, "
        f"P@10={test_metrics.precision_at_10:.4f}, "
        f"R@10={test_metrics.recall_at_10:.4f}, "
        f"NDCG@10={test_metrics.ndcg_at_10:.4f}"
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame([{
        "model": "HeteroGNN",
        "auc": test_metrics.auc,
        "precision@10": test_metrics.precision_at_10,
        "recall@10": test_metrics.recall_at_10,
        "ndcg@10": test_metrics.ndcg_at_10,
        "best_val_ndcg": history["best_ndcg"],
        "num_params": n_params,
        "num_layers": num_layers,
        "hidden_dim": config.gnn.hidden_dim,
        "embedding_dim": config.gnn.embedding_dim,
    }])
    results_df.to_csv(RESULTS_DIR / "phase3_gnn_results.csv", index=False)

    logger.info(f"\nResults saved to {RESULTS_DIR / 'phase3_gnn_results.csv'}")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Heterogeneous GNN for venue recommendation"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced epochs")

    args = parser.parse_args()
    main(args)
