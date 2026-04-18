#!/usr/bin/env python3
"""
V3 GNN Training Pipeline - Feature-Rich, Index-Aligned.

Key improvements over previous versions:
  1. Real feature initialization (no random noise):
     - Users: behavioral stats from users.parquet (review_count,
       average_stars, useful, funny, cool, fans) log-scaled
     - Venues: BERTopic MBTI embeddings (768-dim) PCA'd to 64-dim
  2. Index-aligned with evaluate_hybrid.py: loads ALL three parquet
     splits to build the global node universe (720,485 users,
     92,037 venues) so saved embeddings load without pad/truncate
  3. Cold-start nodes get mean-embedding initialization rather than
     zeros or random noise
  4. AUC inversion fix, double-sigmoid fix, gBCE loss

Usage:
    python scripts/train_gnn_v3.py
    python scripts/train_gnn_v3.py --epochs 30 --hidden-dim 128
"""
import argparse
import gc
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src.config import get_config
from src.config.settings import PROCESSED_DATA_DIR, MODEL_DIR, CHECKPOINT_DIR
from src.models.gnn import LTGNN, GNNTrainer
from src.utils.helpers import setup_logging, set_seed, get_device

logger = logging.getLogger(__name__)

# Behavioral columns from users.parquet that serve as real user features.
# Count-based features are log1p-scaled; average_stars is left raw (1-5).
USER_BEHAVIOR_COLS = [
    "review_count",
    "average_stars",
    "useful",
    "funny",
    "cool",
    "fans",
]
LOG_SCALE_COLS = {"review_count", "useful", "funny", "cool", "fans"}


def build_global_id_maps(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Load all three review splits and build global ID maps.

    This replicates the EXACT logic in evaluate_hybrid.py so that
    the GNN embedding indices match the hybrid evaluator's indices:
        unique_X = concat(train, val, test)["X"].unique()
        id_map = {xid: idx for idx, xid in enumerate(unique_X)}

    Returns:
        train_df, val_df, test_df, user_id_map, venue_id_map
    """
    train = pd.read_parquet(data_dir / "train_reviews.parquet")
    val = pd.read_parquet(data_dir / "val_reviews.parquet")
    test = pd.read_parquet(data_dir / "test_reviews.parquet")

    all_reviews = pd.concat([train, val, test], ignore_index=True)
    unique_users = all_reviews["user_id"].unique()
    unique_venues = all_reviews["business_id"].unique()

    user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
    venue_id_map = {vid: idx for idx, vid in enumerate(unique_venues)}

    logger.info(
        f"Global node universe: {len(user_id_map)} users, "
        f"{len(venue_id_map)} venues"
    )

    return train, val, test, user_id_map, venue_id_map


def build_user_features(
    users_parquet: Path,
    user_id_map: dict,
    n_users: int,
) -> np.ndarray:
    """
    Build user feature matrix from Yelp profile behavioral stats.

    Uses real signals (review_count, average_stars, useful, funny,
    cool, fans) instead of random MBTI noise. Count features are
    log1p-scaled to reduce skew. Cold-start users (in reviews but
    missing from users.parquet) get the column-mean embedding.

    Args:
        users_parquet: Path to users.parquet.
        user_id_map: Global user_id -> index mapping.
        n_users: Total number of users.

    Returns:
        Feature matrix [n_users, n_features] float32.
    """
    n_features = len(USER_BEHAVIOR_COLS)
    features = np.full((n_users, n_features), np.nan, dtype=np.float32)

    users_df = pd.read_parquet(users_parquet, columns=["user_id"] + USER_BEHAVIOR_COLS)

    # Map to global indices
    mapped_idx = users_df["user_id"].map(user_id_map)
    valid = mapped_idx.notna()
    idx = mapped_idx[valid].astype(int).values

    for i, col in enumerate(USER_BEHAVIOR_COLS):
        vals = users_df.loc[valid, col].values.astype(np.float32)
        if col in LOG_SCALE_COLS:
            vals = np.log1p(vals)
        features[idx, i] = vals

    # Cold-start imputation: replace NaN rows with column means
    col_means = np.nanmean(features, axis=0)
    nan_mask = np.isnan(features[:, 0])
    n_cold = nan_mask.sum()
    features[nan_mask] = col_means

    # Standardize (zero mean, unit variance) for stable GNN projection
    mu = features.mean(axis=0)
    sigma = features.std(axis=0) + 1e-8
    features = (features - mu) / sigma

    logger.info(
        f"User features: [{n_users}, {n_features}] "
        f"({n_users - n_cold} from profiles, {n_cold} cold-start mean-filled)"
    )

    return features.astype(np.float32)


def build_venue_features(
    bertopic_dir: Path,
    venue_id_map: dict,
    n_venues: int,
    target_dim: int = 64,
) -> np.ndarray:
    """
    Build venue feature matrix from BERTopic MBTI embeddings.

    Loads the 768-dim MBTI-informed embeddings, applies PCA to
    target_dim, and maps them to the global venue_id_map. Cold-start
    venues (in reviews but not in BERTopic) get the mean embedding.

    Args:
        bertopic_dir: Path to bertopic_mbti/ directory.
        venue_id_map: Global venue_id -> index mapping.
        n_venues: Total number of venues.
        target_dim: Output dimensionality (PCA components).

    Returns:
        Feature matrix [n_venues, target_dim] float32.
    """
    emb_path = bertopic_dir / "venue_embeddings.npy"
    topics_path = bertopic_dir / "venue_topics.parquet"

    raw_embs = np.load(emb_path)  # (85857, 768)
    logger.info(f"Loaded BERTopic venue embeddings: {raw_embs.shape}")

    # PCA reduction
    if raw_embs.shape[1] > target_dim:
        pca = PCA(n_components=target_dim, random_state=42)
        reduced = pca.fit_transform(raw_embs).astype(np.float32)
        explained = pca.explained_variance_ratio_.sum()
        logger.info(
            f"PCA: {raw_embs.shape[1]}-dim -> {target_dim}-dim "
            f"(explained variance: {explained:.3f})"
        )
    else:
        reduced = raw_embs.astype(np.float32)

    # Load venue_id ordering from BERTopic output
    topics_df = pd.read_parquet(topics_path, columns=["venue_id"])
    bertopic_venue_ids = topics_df["venue_id"].values

    assert len(bertopic_venue_ids) == reduced.shape[0], (
        f"venue_topics ({len(bertopic_venue_ids)}) != "
        f"venue_embeddings ({reduced.shape[0]})"
    )

    # Map to global indices with mean imputation for cold-start
    mean_emb = reduced.mean(axis=0)
    features = np.tile(mean_emb, (n_venues, 1)).astype(np.float32)

    n_mapped = 0
    for i, vid in enumerate(bertopic_venue_ids):
        global_idx = venue_id_map.get(vid, -1)
        if global_idx >= 0:
            features[global_idx] = reduced[i]
            n_mapped += 1

    n_cold = n_venues - n_mapped
    logger.info(
        f"Venue features: [{n_venues}, {target_dim}] "
        f"({n_mapped} from BERTopic, {n_cold} cold-start mean-filled)"
    )

    return features


def build_edge_index(
    df: pd.DataFrame,
    user_id_map: dict,
    venue_id_map: dict,
    label: str = "edges",
) -> torch.Tensor:
    """
    Map a review DataFrame to a [2, E] edge index tensor.

    Args:
        df: DataFrame with user_id and business_id columns.
        user_id_map: User string ID -> integer index.
        venue_id_map: Venue string ID -> integer index.
        label: Label for logging.

    Returns:
        Edge index tensor [2, n_valid_edges] int64.
    """
    user_idx = df["user_id"].map(user_id_map)
    venue_idx = df["business_id"].map(venue_id_map)

    # Drop any unmapped (should be 0 since maps come from same data)
    valid = user_idx.notna() & venue_idx.notna()
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        logger.warning(f"{label}: dropped {n_dropped} unmapped edges")

    edge_index = torch.tensor(
        np.array([
            user_idx[valid].astype(int).values,
            venue_idx[valid].astype(int).values,
        ]),
        dtype=torch.long,
    )
    logger.info(f"{label}: {edge_index.size(1)} edges")

    return edge_index


def main(args):
    """V3 training pipeline with real features and index alignment."""
    setup_logging(level=logging.INFO)
    config = get_config()
    set_seed(42)
    device = get_device(args.device)

    # -- 1. Build global ID maps (same logic as evaluate_hybrid.py) --
    logger.info("Loading all review splits for global ID alignment...")
    train_df, val_df, test_df, user_id_map, venue_id_map = (
        build_global_id_maps(PROCESSED_DATA_DIR)
    )
    n_users = len(user_id_map)
    n_venues = len(venue_id_map)

    logger.info(
        f"Train interactions: {len(train_df)}, "
        f"Val interactions: {len(val_df)}, "
        f"Test interactions: {len(test_df)}"
    )

    # Free test_df (not used for GNN training)
    del test_df
    gc.collect()

    # -- 2. Build REAL feature matrices --
    logger.info("Building user features from behavioral profiles...")
    user_feat_np = build_user_features(
        PROCESSED_DATA_DIR / "users.parquet",
        user_id_map,
        n_users,
    )

    logger.info("Building venue features from BERTopic MBTI embeddings...")
    bertopic_dir = MODEL_DIR / "bertopic_mbti"
    if not (bertopic_dir / "venue_embeddings.npy").exists():
        # Fallback to non-MBTI BERTopic
        bertopic_dir = MODEL_DIR / "bertopic"
        logger.warning(
            f"bertopic_mbti not found, falling back to {bertopic_dir}"
        )

    venue_feat_np = build_venue_features(
        bertopic_dir,
        venue_id_map,
        n_venues,
        target_dim=args.venue_feature_dim,
    )

    logger.info(
        f"Feature dimensions: "
        f"user={user_feat_np.shape[1]}, "
        f"venue={venue_feat_np.shape[1]}"
    )

    # -- 3. Build edge indices --
    train_edge_index = build_edge_index(
        train_df, user_id_map, venue_id_map, "Train"
    )
    val_edge_index = build_edge_index(
        val_df, user_id_map, venue_id_map, "Val"
    )

    # Free DataFrames
    del train_df, val_df
    gc.collect()

    # -- 4. Create model --
    logger.info("Creating LTGNN model...")
    model = LTGNN(
        user_input_dim=user_feat_np.shape[1],
        venue_input_dim=venue_feat_np.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_iterations=args.num_iterations,
        dropout=args.dropout,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # -- 5. Configure and train --
    config.gnn.num_epochs = args.epochs
    config.gnn.batch_size = args.batch_size
    config.gnn.learning_rate = args.learning_rate

    user_features = torch.tensor(user_feat_np, dtype=torch.float32)
    venue_features = torch.tensor(venue_feat_np, dtype=torch.float32)
    del user_feat_np, venue_feat_np
    gc.collect()

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

    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Starting GNN training...")
    history = trainer.train()

    # -- 6. Extract and save final embeddings --
    logger.info("Extracting final 64-dim embeddings...")
    user_emb, venue_emb = trainer.get_embeddings()

    assert user_emb.size(0) == n_users, (
        f"user embeddings {user_emb.size(0)} != {n_users}"
    )
    assert venue_emb.size(0) == n_venues, (
        f"venue embeddings {venue_emb.size(0)} != {n_venues}"
    )

    user_emb_np = user_emb.cpu().numpy()
    venue_emb_np = venue_emb.cpu().numpy()

    # Save to both directories (gnn/ for general use, gnn_hetero/ for
    # evaluate_hybrid.py which loads from there)
    for out_dir, u_name, v_name in [
        (MODEL_DIR / "gnn", "user_gnn_embeddings.npy", "venue_gnn_embeddings.npy"),
        (MODEL_DIR / "gnn_hetero", "user_embeddings.npy", "venue_embeddings.npy"),
    ]:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / u_name, user_emb_np)
        np.save(out_dir / v_name, venue_emb_np)

    # Save ID mappings for reproducibility verification
    id_maps = {
        "n_users": n_users,
        "n_venues": n_venues,
        "user_id_map_sample": dict(list(user_id_map.items())[:5]),
        "venue_id_map_sample": dict(list(venue_id_map.items())[:5]),
    }
    maps_path = MODEL_DIR / "gnn_hetero" / "id_mappings.json"
    with open(maps_path, "w") as f:
        json.dump(id_maps, f, indent=2)

    # Also save the full maps as torch file for evaluate_hybrid.py
    torch.save(
        {"user_id_map": user_id_map, "venue_id_map": venue_id_map},
        MODEL_DIR / "gnn_hetero" / "id_mappings.pt",
    )

    logger.info(f"Training complete!")
    logger.info(f"Best NDCG@10: {history['best_metric']:.4f}")
    logger.info(
        f"Saved embeddings: "
        f"users [{n_users}, {args.embedding_dim}], "
        f"venues [{n_venues}, {args.embedding_dim}]"
    )
    logger.info(f"Output: {MODEL_DIR / 'gnn_hetero'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GNN V3 with real features and index alignment"
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--venue-feature-dim", type=int, default=64,
                        help="PCA target dim for BERTopic venue embeddings")
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use-gbce", action="store_true", default=True)
    parser.add_argument("--no-gbce", action="store_false", dest="use_gbce")
    parser.add_argument("--gbce-t", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "cpu"])

    args = parser.parse_args()
    main(args)
