#!/usr/bin/env python3
"""
Train Hybrid Recommendation Model.

Combines GNN embeddings with XGBoost ranking and DCI patterns.

Usage:
    python scripts/train_hybrid.py --num-rounds 100
    python scripts/train_hybrid.py --calibration-t 0.8 --num-negatives 4
"""
import argparse
import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.config import get_config, PROCESSED_DATA_DIR, MODEL_DIR
from src.models.hybrid import (
    HybridRecommender,
    UserProfileMiner,
    CalibrationMetrics,
)
from src.utils.helpers import setup_logging, set_seed
from src.utils.metrics import RecommendationMetrics

logger = logging.getLogger(__name__)


def main(args):
    """Main training function."""
    setup_logging(level=logging.INFO)
    set_seed(42)
    config = get_config()

    # Load data
    logger.info("Loading data...")

    # Load interactions
    train_reviews = pd.read_parquet(PROCESSED_DATA_DIR / "train_reviews.parquet")
    val_reviews = pd.read_parquet(PROCESSED_DATA_DIR / "val_reviews.parquet")
    test_reviews = pd.read_parquet(PROCESSED_DATA_DIR / "test_reviews.parquet")

    logger.info(f"Train: {len(train_reviews)}, Val: {len(val_reviews)}, Test: {len(test_reviews)}")

    # Load GNN embeddings
    gnn_dir = MODEL_DIR / "gnn"
    user_emb_path = gnn_dir / "user_gnn_embeddings.npy"
    venue_emb_path = gnn_dir / "venue_gnn_embeddings.npy"
    id_map_path = gnn_dir / "id_mappings.json"

    if user_emb_path.exists():
        user_embeddings = np.load(user_emb_path)
        venue_embeddings = np.load(venue_emb_path)
        with open(id_map_path) as f:
            id_mappings = json.load(f)
        user_ids = id_mappings["user_ids"]
        venue_ids = id_mappings["venue_ids"]
        logger.info(f"Loaded GNN embeddings: users {user_embeddings.shape}, venues {venue_embeddings.shape}")
        logger.info(f"ID mappings: {len(user_ids)} users, {len(venue_ids)} venues")

        # Reconcile: ID list MUST match embedding rows.
        # They can diverge if embeddings and id_mappings were saved by different runs.
        if len(user_ids) != user_embeddings.shape[0]:
            n = min(len(user_ids), user_embeddings.shape[0])
            logger.warning(
                f"user_ids ({len(user_ids)}) != user_embeddings "
                f"({user_embeddings.shape[0]}), truncating to {n}"
            )
            user_ids = user_ids[:n]
            user_embeddings = user_embeddings[:n]

        if len(venue_ids) != venue_embeddings.shape[0]:
            n = min(len(venue_ids), venue_embeddings.shape[0])
            logger.warning(
                f"venue_ids ({len(venue_ids)}) != venue_embeddings "
                f"({venue_embeddings.shape[0]}), truncating to {n}"
            )
            venue_ids = venue_ids[:n]
            venue_embeddings = venue_embeddings[:n]
    else:
        logger.warning("GNN embeddings not found, using random initialization")
        user_ids = train_reviews["user_id"].unique().tolist()
        venue_ids = train_reviews["business_id"].unique().tolist()
        user_embeddings = np.random.randn(len(user_ids), 64).astype(np.float32) * 0.1
        venue_embeddings = np.random.randn(len(venue_ids), 64).astype(np.float32) * 0.1

    # Create ID mappings (indices are now guaranteed to be < embedding rows)
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    venue_id_map = {vid: idx for idx, vid in enumerate(venue_ids)}
    logger.info(
        f"ID maps: {len(user_id_map)} users, {len(venue_id_map)} venues, "
        f"embeddings: users {user_embeddings.shape}, venues {venue_embeddings.shape}"
    )

    # Mine user patterns with DCI Closed
    logger.info("Mining user patterns with DCI Closed...")
    miner = UserProfileMiner(
        min_support=args.min_support,
        max_itemset_size=args.max_itemset_size,
    )
    miner.fit(train_reviews, user_col="user_id", venue_col="business_id")

    # Get itemset features for users
    user_profiles_df = miner.get_all_profiles_df(itemset_features=args.itemset_features)
    logger.info(f"Mined patterns for {len(user_profiles_df)} users")
    logger.info(f"Found {len(miner.dci.closed_itemsets)} closed itemsets")

    # Create user extra features from itemset patterns (vectorized)
    itemset_cols = [f"itemset_{i}" for i in range(args.itemset_features)]
    user_extra = np.zeros((len(user_ids), args.itemset_features), dtype=np.float32)

    # Map profile user_ids to indices (vectorized, no iterrows)
    profile_user_ids = user_profiles_df["user_id"].values
    profile_indices = np.array([
        user_id_map.get(uid, -1) for uid in profile_user_ids
    ])
    valid = profile_indices >= 0
    if valid.any():
        valid_indices = profile_indices[valid]
        itemset_values = user_profiles_df[itemset_cols].values[valid]
        user_extra[valid_indices] = itemset_values.astype(np.float32)

    logger.info(f"User extra features shape: {user_extra.shape}")

    # Convert edges to indices (vectorized, not iterrows)
    def reviews_to_edges(reviews_df):
        u_idx = np.array([user_id_map.get(uid, -1) for uid in reviews_df["user_id"].values])
        v_idx = np.array([venue_id_map.get(vid, -1) for vid in reviews_df["business_id"].values])
        valid = (u_idx >= 0) & (v_idx >= 0)
        return np.array([u_idx[valid], v_idx[valid]])

    train_edges = reviews_to_edges(train_reviews)
    val_edges = reviews_to_edges(val_reviews)
    test_edges = reviews_to_edges(test_reviews)

    logger.info(f"Train edges: {train_edges.shape[1]}")
    logger.info(f"Val edges: {val_edges.shape[1]}")
    logger.info(f"Test edges: {test_edges.shape[1]}")

    # Configure hybrid model
    config.hybrid.xgb_n_estimators = args.num_rounds
    config.hybrid.gbce_calibration_t = args.calibration_t

    # Create and train hybrid recommender
    logger.info("Training hybrid recommender...")
    recommender = HybridRecommender(config=config.hybrid)

    history = recommender.fit(
        train_edges=train_edges,
        user_embeddings=user_embeddings,
        venue_embeddings=venue_embeddings,
        user_ids=user_ids,
        venue_ids=venue_ids,
        user_extra=user_extra,
        val_edges=val_edges,
        num_negatives=args.num_negatives,
        num_rounds=args.num_rounds,
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")

    # Sample users for evaluation
    test_users = test_reviews["user_id"].unique()[:100]

    # Get ground truth
    user_ground_truth = {}
    for uid in test_users:
        user_venues = test_reviews[test_reviews["user_id"] == uid]["business_id"].tolist()
        user_ground_truth[uid] = set(user_venues)

    # Get recommendations
    user_recommendations = {}
    for uid in test_users:
        recs = recommender.recommend(uid, k=20, exclude_visited=True)
        user_recommendations[uid] = [vid for vid, _ in recs]

    # Compute metrics
    metrics = RecommendationMetrics()
    results = metrics.evaluate_all(
        user_recommendations,
        user_ground_truth,
        k_values=[5, 10, 20],
    )

    logger.info("\nTest Set Results:")
    for metric, value in sorted(results.items()):
        logger.info(f"  {metric}: {value:.4f}")

    # Compute calibration metrics
    logger.info("\nCalibration Analysis...")

    # Get predictions for test edges
    test_user_idx = test_edges[0]
    test_venue_idx = test_edges[1]

    pos_scores = recommender.ranker.predict(
        test_user_idx,
        test_venue_idx,
        user_embeddings,
        venue_embeddings,
        user_extra,
    )

    # Sample negatives
    neg_venue_idx = np.random.randint(0, len(venue_ids), len(test_user_idx))
    neg_scores = recommender.ranker.predict(
        test_user_idx,
        neg_venue_idx,
        user_embeddings,
        venue_embeddings,
        user_extra,
    )

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    ece = CalibrationMetrics.expected_calibration_error(all_scores, all_labels)
    mce = CalibrationMetrics.maximum_calibration_error(all_scores, all_labels)

    logger.info(f"  Expected Calibration Error (ECE): {ece:.4f}")
    logger.info(f"  Maximum Calibration Error (MCE): {mce:.4f}")

    # Save model
    output_dir = MODEL_DIR / "hybrid"
    output_dir.mkdir(parents=True, exist_ok=True)

    recommender.ranker.save(output_dir / "xgboost_ranker.json")

    # Save feature importance
    importance = recommender.ranker.get_feature_importance()
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_dir / "test_results.csv", index=False)

    logger.info(f"\nModel saved to {output_dir}")
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid recommender")

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=100,
        help="XGBoost boosting rounds",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=4,
        help="Negative samples per positive",
    )
    parser.add_argument(
        "--calibration-t",
        type=float,
        default=0.8,
        help="gBCE calibration temperature",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.005,
        help="Minimum support for DCI Closed (0.005 = 0.5%% of users, "
             "~1178 users for 235K dataset)",
    )
    parser.add_argument(
        "--max-itemset-size",
        type=int,
        default=3,
        help="Maximum itemset size (3 is practical for Yelp scale)",
    )
    parser.add_argument(
        "--itemset-features",
        type=int,
        default=10,
        help="Number of itemset features",
    )

    args = parser.parse_args()
    main(args)
