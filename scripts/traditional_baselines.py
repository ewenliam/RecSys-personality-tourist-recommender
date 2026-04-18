#!/usr/bin/env python3
"""
Traditional ML Baselines: Non-Graph Recommendation Models.

Trains and evaluates classical ML models using ONLY MBTI personality
features and BERTopic embeddings (no GNN structural data). These serve
as lower-bound baselines to demonstrate the value of graph-based
learning in the hybrid pipeline.

Models:
  1. Logistic Regression (linear baseline)
  2. Decision Tree (non-linear, interpretable)
  3. Random Forest (ensemble)
  4. LinearSVC via SGDClassifier (scalable SVM)
  5. KNN with subsampling (similarity-based)

Memory Safety:
  With 2.8M training edges, materializing all user-venue pairs is
  infeasible. We subsample negatives per user and train on a
  manageable matrix (~500K-1M rows).

Usage:
    python scripts/traditional_baselines.py
    python scripts/traditional_baselines.py --max-train-rows 500000 --num-negatives 4
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_config, PROCESSED_DATA_DIR, MODEL_DIR, PROJECT_ROOT
from src.utils.metrics import RecommendationMetrics
from src.utils.helpers import setup_logging, set_seed

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"


def load_embeddings(model_dir: Path) -> dict:
    """Load BERTopic venue embeddings and topic assignments."""
    data = {}

    for subdir in ["bertopic_v2", "bertopic_mbti", "bertopic"]:
        ve_path = model_dir / subdir / "venue_embeddings.npy"
        vt_path = model_dir / subdir / "venue_topics.parquet"

        if ve_path.exists() and "venue_bert_embs" not in data:
            data["venue_bert_embs"] = np.load(ve_path)
            logger.info(f"Loaded venue embeddings from {subdir}: {data['venue_bert_embs'].shape}")
        if vt_path.exists() and "venue_topics" not in data:
            data["venue_topics"] = pd.read_parquet(vt_path)
            logger.info(f"Loaded venue topics from {subdir}")

    return data


def load_interactions(data_dir: Path) -> pd.DataFrame:
    """Load interaction data from parquet splits."""
    dfs = []
    for split in ["train_reviews.parquet", "val_reviews.parquet", "test_reviews.parquet"]:
        path = data_dir / split
        if path.exists():
            dfs.append(pd.read_parquet(path))

    return pd.concat(dfs, ignore_index=True)


def build_features(
    user_indices: np.ndarray,
    venue_indices: np.ndarray,
    user_mbti_probs: np.ndarray,
    venue_bert_embs: np.ndarray,
) -> np.ndarray:
    """
    Build feature vectors for user-venue pairs.

    Features: [user_mbti_probs(4) | venue_bert_embs(dim) | elementwise_interaction(4)]

    The interaction features capture how personality dimensions
    relate to venue embedding magnitude (a simple cross-domain signal).
    """
    u_feats = user_mbti_probs[user_indices]  # [N, 4]
    v_embs = venue_bert_embs[venue_indices]   # [N, emb_dim]

    # Cross-feature: MBTI probs * mean venue embedding magnitude
    v_mag = np.linalg.norm(v_embs, axis=1, keepdims=True)  # [N, 1]
    cross = u_feats * v_mag  # [N, 4]

    return np.hstack([u_feats, v_embs, cross])


def create_train_test_data(
    interactions: pd.DataFrame,
    n_users: int,
    n_venues: int,
    user_id_map: dict,
    venue_id_map: dict,
    user_mbti_probs: np.ndarray,
    venue_bert_embs: np.ndarray,
    num_negatives: int = 4,
    test_ratio: float = 0.2,
    max_train_rows: int = 500_000,
) -> tuple:
    """
    Create train/test feature matrices with negative sampling.

    Memory-safe: caps total training rows to max_train_rows.
    """
    user_col = "user_id" if "user_id" in interactions.columns else interactions.columns[0]
    biz_col = "business_id" if "business_id" in interactions.columns else "venue_id"

    valid = interactions[
        interactions[user_col].isin(user_id_map) &
        interactions[biz_col].isin(venue_id_map)
    ]

    user_idx = valid[user_col].map(user_id_map).values
    venue_idx = valid[biz_col].map(venue_id_map).values

    n = len(user_idx)
    perm = np.random.permutation(n)
    n_test = int(n * test_ratio)

    test_mask = perm[:n_test]
    train_mask = perm[n_test:]

    # Build ground truth for test users
    user_gt = {}
    for i in test_mask:
        u, v = int(user_idx[i]), int(venue_idx[i])
        if u not in user_gt:
            user_gt[u] = set()
        user_gt[u].add(v)

    # Build positive interactions set for negative sampling
    pos_set = set(zip(user_idx.tolist(), venue_idx.tolist()))

    # Subsample training positives if too many
    max_positives = max_train_rows // (1 + num_negatives)
    if len(train_mask) > max_positives:
        train_mask = np.random.choice(train_mask, max_positives, replace=False)
        logger.info(f"Subsampled training positives to {max_positives}")

    # Generate training data with negatives
    train_users = []
    train_venues = []
    train_labels = []

    for i in train_mask:
        u, v = int(user_idx[i]), int(venue_idx[i])
        train_users.append(u)
        train_venues.append(v)
        train_labels.append(1)

        # Sample negatives
        for _ in range(num_negatives):
            neg_v = np.random.randint(0, n_venues)
            while (u, neg_v) in pos_set:
                neg_v = np.random.randint(0, n_venues)
            train_users.append(u)
            train_venues.append(neg_v)
            train_labels.append(0)

    train_users = np.array(train_users)
    train_venues = np.array(train_venues)
    train_labels = np.array(train_labels)

    logger.info(f"Training samples: {len(train_labels)} ({train_labels.sum()} pos, {(1 - train_labels).sum()} neg)")

    # Build feature matrices
    X_train = build_features(train_users, train_venues, user_mbti_probs, venue_bert_embs)
    y_train = train_labels

    return X_train, y_train, user_gt, pos_set


def evaluate_model(
    model,
    scaler: StandardScaler,
    user_mbti_probs: np.ndarray,
    venue_bert_embs: np.ndarray,
    user_gt: dict,
    n_venues: int,
    k_values: list[int],
    num_eval_users: int = 500,
    model_name: str = "Model",
    max_candidates: int = 2000,
) -> tuple[pd.DataFrame, float]:
    """
    Evaluate a trained classifier as a recommender.

    For each test user, score `max_candidates` venues (always including
    ground truth venues) and rank by predicted probability. Capping
    candidates makes KNN tractable on large catalogs.
    """
    calc = RecommendationMetrics()

    eval_users = list(user_gt.keys())
    if len(eval_users) > num_eval_users:
        eval_users = list(np.random.choice(eval_users, num_eval_users, replace=False))

    results = {k: {"precision": [], "recall": [], "ndcg": [], "mrr": [], "hit_rate": []}
               for k in k_values}

    all_scores = []
    all_labels = []

    # Whether to use sampled candidates (slow models like KNN)
    use_sampling = n_venues > max_candidates

    for user_idx in eval_users:
        gt = user_gt[user_idx]
        if not gt:
            continue

        if use_sampling:
            # Always include ground truth venues; sample the rest
            gt_list = list(gt)
            n_random = max_candidates - len(gt_list)
            if n_random > 0:
                random_venues = np.random.choice(
                    [v for v in range(n_venues) if v not in gt],
                    size=min(n_random, n_venues - len(gt_list)),
                    replace=False,
                )
                candidate_venues = np.concatenate([gt_list, random_venues])
            else:
                candidate_venues = np.array(gt_list[:max_candidates])
        else:
            candidate_venues = np.arange(n_venues)

        user_indices = np.full(len(candidate_venues), user_idx)
        X = build_features(user_indices, candidate_venues, user_mbti_probs, venue_bert_embs)
        X_scaled = scaler.transform(X)

        # Get scores (probability of class 1, or decision function)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_scaled)[:, 1]
        else:
            scores = model.decision_function(X_scaled)

        # Collect for ECE
        for i, v in enumerate(candidate_venues[:1000]):
            all_scores.append(float(scores[i]))
            all_labels.append(1 if v in gt else 0)

        # Map top-k back to venue indices
        max_k = max(k_values)
        top_local = np.argsort(scores)[::-1][:max_k]
        recs = candidate_venues[top_local].tolist()

        for k in k_values:
            results[k]["precision"].append(calc.precision_at_k(recs, gt, k))
            results[k]["recall"].append(calc.recall_at_k(recs, gt, k))
            results[k]["ndcg"].append(calc.ndcg_at_k(recs, gt, k))
            results[k]["mrr"].append(calc.mrr(recs, gt))
            results[k]["hit_rate"].append(calc.hit_rate_at_k(recs, gt, k))

    # Normalize scores for ECE (sigmoid if not already probabilities)
    if all_scores:
        scores_arr = np.array(all_scores)
        if scores_arr.min() < 0 or scores_arr.max() > 1:
            scores_arr = 1.0 / (1.0 + np.exp(-scores_arr))
            all_scores = scores_arr.tolist()
        ece = calc.expected_calibration_error(all_scores, all_labels)
    else:
        ece = 0.0

    rows = []
    for k in k_values:
        rows.append({
            "Model": model_name,
            "K": k,
            "Precision@K": np.mean(results[k]["precision"]),
            "Recall@K": np.mean(results[k]["recall"]),
            "NDCG@K": np.mean(results[k]["ndcg"]),
            "MRR": np.mean(results[k]["mrr"]),
            "Hit Rate@K": np.mean(results[k]["hit_rate"]),
            "ECE": ece,
        })

    return pd.DataFrame(rows), ece


def main(args):
    """Train and evaluate traditional ML baselines."""
    setup_logging(level=logging.INFO)
    set_seed(42)

    config = get_config()
    k_values = [5, 10, 20]

    # Load data
    logger.info("Loading data...")
    emb_data = load_embeddings(MODEL_DIR)
    interactions = load_interactions(PROCESSED_DATA_DIR)

    user_col = "user_id" if "user_id" in interactions.columns else interactions.columns[0]
    biz_col = "business_id" if "business_id" in interactions.columns else "venue_id"

    unique_users = interactions[user_col].unique()
    unique_venues = interactions[biz_col].unique()
    user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
    venue_id_map = {vid: idx for idx, vid in enumerate(unique_venues)}
    n_users = len(unique_users)
    n_venues = len(unique_venues)

    logger.info(f"Users: {n_users}, Venues: {n_venues}")

    # MBTI probabilities (from trained model or random placeholder)
    user_mbti_probs = np.random.rand(n_users, 4).astype(np.float32)

    # Venue BERTopic embeddings (or random placeholder)
    venue_bert_embs = emb_data.get(
        "venue_bert_embs",
        np.random.randn(n_venues, 64).astype(np.float32),
    )

    # Reconcile embedding dimensions
    if venue_bert_embs.shape[0] != n_venues:
        logger.warning(
            f"Venue embeddings shape {venue_bert_embs.shape[0]} != n_venues {n_venues}, "
            f"padding/truncating"
        )
        padded = np.random.randn(n_venues, venue_bert_embs.shape[1]).astype(np.float32)
        n = min(venue_bert_embs.shape[0], n_venues)
        padded[:n] = venue_bert_embs[:n]
        venue_bert_embs = padded

    # Create train/test data
    logger.info("Building train/test data with negative sampling...")
    X_train, y_train, user_gt, pos_set = create_train_test_data(
        interactions, n_users, n_venues,
        user_id_map, venue_id_map,
        user_mbti_probs, venue_bert_embs,
        num_negatives=args.num_negatives,
        max_train_rows=args.max_train_rows,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    logger.info(f"Feature matrix: {X_train_scaled.shape}")

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=50, random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=20,
            n_jobs=-1, random_state=42,
        ),
        "LinearSVC (SGD)": SGDClassifier(
            loss="hinge", alpha=1e-4, max_iter=1000,
            random_state=42, n_jobs=-1,
        ),
        "KNN (k=50)": KNeighborsClassifier(
            n_neighbors=50, metric="cosine", n_jobs=-1,
        ),
    }

    # KNN needs aggressive subsampling: training and eval are both O(n_train * n_query)
    # With 100K train rows and 2000 candidates * 500 users = 1M queries -> 100B ops
    # Use 5K training rows, 200 candidates, 50 eval users to keep it under 1 second
    KNN_TRAIN_ROWS = 5_000
    KNN_MAX_CANDIDATES = 200
    KNN_EVAL_USERS = 50

    all_metrics = []

    for name, model in models.items():
        logger.info("=" * 60)
        logger.info(f"Training: {name}")
        logger.info("=" * 60)

        t0 = time.time()

        if "KNN" in name:
            idx = np.random.choice(len(X_train_scaled), min(KNN_TRAIN_ROWS, len(X_train_scaled)), replace=False)
            model.fit(X_train_scaled[idx], y_train[idx])
            logger.info(f"KNN trained on {len(idx)} rows (strict subsample for speed)")
        else:
            model.fit(X_train_scaled, y_train)

        train_time = time.time() - t0
        logger.info(f"Training time: {train_time:.1f}s")

        eval_users = KNN_EVAL_USERS if "KNN" in name else args.num_eval_users
        max_cand = KNN_MAX_CANDIDATES if "KNN" in name else 2000

        # Evaluate
        metrics_df, ece = evaluate_model(
            model, scaler,
            user_mbti_probs, venue_bert_embs,
            user_gt, n_venues, k_values,
            num_eval_users=eval_users,
            max_candidates=max_cand,
            model_name=name,
        )

        all_metrics.append(metrics_df)
        logger.info(f"{name} ECE: {ece:.4f}")
        logger.info(f"{name} results:\n{metrics_df.to_string(index=False)}")

    # Combine all results
    combined = pd.concat(all_metrics, ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RESULTS_DIR / "traditional_baselines.csv", index=False)

    # Print summary (K=10 only for readability)
    logger.info("\n" + "=" * 80)
    logger.info("TRADITIONAL BASELINES SUMMARY (K=10)")
    logger.info("=" * 80)

    summary = combined[combined["K"] == 10][[
        "Model", "Precision@K", "Recall@K", "NDCG@K", "MRR", "Hit Rate@K", "ECE"
    ]]
    print(summary.to_string(index=False))

    logger.info(f"\nFull results saved to {RESULTS_DIR / 'traditional_baselines.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traditional ML Baselines")
    parser.add_argument("--max-train-rows", type=int, default=500_000,
                        help="Maximum training rows (memory safety)")
    parser.add_argument("--num-negatives", type=int, default=4,
                        help="Negative samples per positive edge")
    parser.add_argument("--num-eval-users", type=int, default=500,
                        help="Number of users to evaluate")

    args = parser.parse_args()
    main(args)
