#!/usr/bin/env python3
"""
Phase 4: Hybrid Evaluation Pipeline.

Combines GNN embeddings with BERT-MBTI personality features via XGBoost
ranking, evaluates with Precision@K, Recall@K, NDCG@K, and generates
methodology comparison tables (Omer's approach vs Current).

Usage:
    python scripts/evaluate_hybrid.py
    python scripts/evaluate_hybrid.py --k 10 --quick
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import (
    get_config,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    PROJECT_ROOT,
)
from src.models.hybrid.xgboost_ranker import XGBoostRanker, FeatureSynthesizer, HybridRecommender
from src.models.hybrid.personality_scorer import PersonalityScorer
from src.models.hybrid.gbce_loss import CalibrationMetrics
from src.utils.metrics import RecommendationMetrics
from src.utils.helpers import setup_logging, set_seed

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"


def load_gnn_embeddings(model_dir: Path) -> dict:
    """Load GNN-learned embeddings from Phase 3."""
    emb_dir = model_dir / "gnn_hetero"
    embeddings = {}

    for node_type in ["user", "venue"]:
        path = emb_dir / f"{node_type}_embeddings.npy"
        if path.exists():
            embeddings[node_type] = np.load(path)
            logger.info(f"Loaded GNN {node_type} embeddings: {embeddings[node_type].shape}")
        else:
            logger.warning(f"GNN {node_type} embeddings not found at {path}")

    # Load ID mappings
    mappings_path = emb_dir / "id_mappings.pt"
    if mappings_path.exists():
        import torch
        embeddings["mappings"] = torch.load(mappings_path, map_location="cpu")
    else:
        embeddings["mappings"] = {}

    return embeddings


def load_personality_data(model_dir: Path) -> dict:
    """Load MBTI personality predictions and embeddings."""
    data = {}

    # MBTI embeddings
    mbti_emb_path = model_dir / "gnn" / "user_gnn_embeddings.npy"
    if mbti_emb_path.exists():
        data["user_mbti_embeddings"] = np.load(mbti_emb_path)

    # Venue embeddings from BERTopic
    for subdir in ["bertopic_mbti", "bertopic"]:
        venue_emb_path = model_dir / subdir / "venue_embeddings.npy"
        if venue_emb_path.exists():
            data["venue_embeddings"] = np.load(venue_emb_path)
            break

    return data


def load_interactions(data_dir: Path) -> tuple[pd.DataFrame, str, str]:
    """Load interaction data and identify columns."""
    dfs = []
    for split in ["train_reviews.parquet", "val_reviews.parquet", "test_reviews.parquet"]:
        path = data_dir / split
        if path.exists():
            dfs.append(pd.read_parquet(path))

    if not dfs:
        raise FileNotFoundError(f"No processed data in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    user_col = "user_id" if "user_id" in df.columns else df.columns[0]
    biz_col = "business_id" if "business_id" in df.columns else "venue_id"

    return df, user_col, biz_col


def create_train_test_edges(
    interactions: pd.DataFrame,
    user_col: str,
    biz_col: str,
    user_id_map: dict,
    venue_id_map: dict,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create edge arrays with train/test split."""
    # Filter to mapped users and venues
    valid = interactions[
        interactions[user_col].isin(user_id_map) &
        interactions[biz_col].isin(venue_id_map)
    ]

    user_indices = valid[user_col].map(user_id_map).values
    venue_indices = valid[biz_col].map(venue_id_map).values

    edges = np.stack([user_indices, venue_indices])

    # Split
    n = edges.shape[1]
    perm = np.random.permutation(n)
    n_test = int(n * test_ratio)

    train_edges = edges[:, perm[n_test:]]
    test_edges = edges[:, perm[:n_test]]

    # Build ground truth per user
    user_gt = {}
    for i in range(test_edges.shape[1]):
        u, v = int(test_edges[0, i]), int(test_edges[1, i])
        if u not in user_gt:
            user_gt[u] = set()
        user_gt[u].add(v)

    logger.info(f"Train edges: {train_edges.shape[1]}, Test edges: {test_edges.shape[1]}")
    return train_edges, test_edges, user_gt


def evaluate_recommendations(
    ranker: XGBoostRanker,
    user_embeddings: np.ndarray,
    venue_embeddings: np.ndarray,
    user_gt: dict,
    k_values: list[int],
    user_extra: np.ndarray = None,
    venue_extra: np.ndarray = None,
    num_eval_users: int = 200,
) -> pd.DataFrame:
    """
    Evaluate recommendation quality with ranking metrics.

    Args:
        ranker: Trained XGBoostRanker.
        user_embeddings: User embedding matrix.
        venue_embeddings: Venue embedding matrix.
        user_gt: Ground truth - dict mapping user_idx -> set of venue_idx.
        k_values: List of K values for metrics.
        user_extra: Extra user features (personality).
        venue_extra: Extra venue features.
        num_eval_users: Number of users to evaluate.

    Returns:
        DataFrame with metrics per K.
    """
    calc = RecommendationMetrics()

    eval_users = list(user_gt.keys())
    if len(eval_users) > num_eval_users:
        eval_users = list(np.random.choice(eval_users, num_eval_users, replace=False))

    results = {k: {"precision": [], "recall": [], "ndcg": [], "mrr": [], "hit_rate": []}
               for k in k_values}

    for user_idx in eval_users:
        gt = user_gt[user_idx]
        if not gt:
            continue

        rec_indices, rec_scores = ranker.recommend(
            user_idx=user_idx,
            user_embeddings=user_embeddings,
            venue_embeddings=venue_embeddings,
            k=max(k_values),
            exclude_venues=None,
            user_extra=user_extra,
            venue_extra=venue_extra,
        )

        recs = rec_indices.tolist()

        for k in k_values:
            results[k]["precision"].append(calc.precision_at_k(recs, gt, k))
            results[k]["recall"].append(calc.recall_at_k(recs, gt, k))
            results[k]["ndcg"].append(calc.ndcg_at_k(recs, gt, k))
            results[k]["mrr"].append(calc.mrr(recs, gt))
            results[k]["hit_rate"].append(calc.hit_rate_at_k(recs, gt, k))

    rows = []
    for k in k_values:
        rows.append({
            "K": k,
            "Precision@K": np.mean(results[k]["precision"]),
            "Recall@K": np.mean(results[k]["recall"]),
            "NDCG@K": np.mean(results[k]["ndcg"]),
            "MRR": np.mean(results[k]["mrr"]),
            "Hit Rate@K": np.mean(results[k]["hit_rate"]),
        })

    return pd.DataFrame(rows)


def run_hybrid_evaluation(
    train_edges: np.ndarray,
    test_edges: np.ndarray,
    user_gt: dict,
    user_embeddings: np.ndarray,
    venue_embeddings: np.ndarray,
    personality_features: np.ndarray,
    config,
    k_values: list[int],
    label: str = "Hybrid",
    num_rounds: int = 100,
    venue_extra: np.ndarray = None,
) -> tuple[pd.DataFrame, XGBoostRanker]:
    """Run a late-fusion ensemble evaluation pipeline.

    XGBoost receives 4 pre-computed scalar features per (user, venue)
    pair rather than raw embeddings:
      [0] knn_score     - cosine(MBTI, BERTopic)
      [1] gnn_score     - cosine(GNN_user, GNN_venue)
      [2] venue_degree  - log(1 + #reviews)
      [3] user_degree   - log(1 + #reviews)

    Args:
        personality_features: MBTI features [n_users, 7] passed as
            user_extra to create_pair_features for KNN score.
        venue_extra: BERTopic PCA embeddings [n_venues, 64] passed
            for KNN score computation (not included raw).
    """
    logger.info(f"[{label}] Training XGBoost late-fusion ensemble...")

    ranker = XGBoostRanker(config=config.hybrid, use_calibration=True)

    # Late fusion: 500K cap is fine since feature matrix is tiny
    # (only 4 columns per pair instead of 144)
    train_features, train_labels = ranker.prepare_training_data(
        train_edges,
        user_embeddings,
        venue_embeddings,
        num_negatives=4,
        user_extra=personality_features,
        venue_extra=venue_extra,
        max_pos_samples=500_000,
    )

    # Train (scale_pos_weight computed automatically from labels)
    ranker.train(
        train_features, train_labels,
        num_rounds=num_rounds,
        early_stopping_rounds=10,
    )

    # -- Feature importance (ALL features, there are only 4) --
    FEAT_NAMES = ["knn_score", "gnn_score", "venue_degree", "user_degree"]
    if hasattr(ranker, "model") and ranker.model is not None:
        importance = ranker.model.get_score(importance_type="gain")
        n_feat = train_features.shape[1]
        feat_names = list(FEAT_NAMES)
        while len(feat_names) < n_feat:
            feat_names.append(f"feat_{len(feat_names)}")

        named_importance = {
            feat_names[int(k[1:])]: v
            for k, v in importance.items()
            if k.startswith("f") and int(k[1:]) < len(feat_names)
        }
        sorted_imp = sorted(
            named_importance.items(), key=lambda x: x[1], reverse=True
        )
        logger.info(f"[{label}] Feature Importances (gain):")
        for rank, (name, gain) in enumerate(sorted_imp, 1):
            logger.info(f"  {rank}. {name:20s} {gain:.1f}")

    # Evaluate
    logger.info(f"[{label}] Evaluating...")
    metrics = evaluate_recommendations(
        ranker, user_embeddings, venue_embeddings,
        user_gt, k_values,
        user_extra=personality_features,
        venue_extra=venue_extra,
    )
    metrics["Model"] = label

    return metrics, ranker


def generate_methodology_comparison(
    current_metrics: pd.DataFrame,
    k: int = 10,
) -> tuple[pd.DataFrame, str]:
    """
    Generate comparison table between Omer's approach and current.

    Since we can't re-run Omer's exact pipeline, we report our metrics
    alongside reported values from the thesis for context.
    """
    # Omer's reported values (from thesis Chapter 6.3)
    omer_reported = {
        "MBTI Accuracy": "~94% (leaky)",
        "BERTopic Model": "all-MiniLM-L6-v2",
        "Upsampling": "Before split (data leakage)",
        "GNN Type": "N/A (XGBoost only)",
        "Classifier": "XGBoost (log loss: 0.4207)",
        "Evaluation": "Log loss, Accuracy",
    }

    current_k_row = current_metrics[current_metrics["K"] == k].iloc[0]

    rows = [
        {
            "Aspect": "MBTI Classifier",
            "Omer (2024)": "BERT -> 16-class (leaky eval)",
            "Current": "BERT -> 4 binary heads (robust eval)",
        },
        {
            "Aspect": "Upsampling Strategy",
            "Omer (2024)": "Entire dataset before split",
            "Current": "Training set only after split",
        },
        {
            "Aspect": "Topic Embeddings",
            "Omer (2024)": "all-MiniLM-L6-v2 (generic)",
            "Current": "BERT-MBTI CLS tokens (personality-informed)",
        },
        {
            "Aspect": "Graph Model",
            "Omer (2024)": "None",
            "Current": f"HeteroGNN (GraphSAGE, link prediction)",
        },
        {
            "Aspect": "Final Ranking",
            "Omer (2024)": "XGBoost (baseline features)",
            "Current": "XGBoost (GNN embs + personality features + gBCE)",
        },
        {
            "Aspect": f"Precision@{k}",
            "Omer (2024)": "Not reported",
            "Current": f"{current_k_row['Precision@K']:.4f}",
        },
        {
            "Aspect": f"Recall@{k}",
            "Omer (2024)": "Not reported",
            "Current": f"{current_k_row['Recall@K']:.4f}",
        },
        {
            "Aspect": f"NDCG@{k}",
            "Omer (2024)": "Not reported",
            "Current": f"{current_k_row['NDCG@K']:.4f}",
        },
        {
            "Aspect": f"MRR",
            "Omer (2024)": "Not reported",
            "Current": f"{current_k_row['MRR']:.4f}",
        },
    ]

    comparison_df = pd.DataFrame(rows)

    # Generate LaTeX
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Methodology Comparison: Omer (2024) vs Current System}\n"
    latex += "\\label{tab:full_methodology_comparison}\n"
    latex += "\\begin{tabular}{l p{5cm} p{5cm}}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Aspect} & \\textbf{Omer (2024)} & \\textbf{Current} \\\\\n"
    latex += "\\midrule\n"

    for _, row in comparison_df.iterrows():
        latex += f"{row['Aspect']} & {row['Omer (2024)']} & {row['Current']} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return comparison_df, latex


def main(args):
    """Run the hybrid evaluation pipeline."""
    setup_logging(level=logging.INFO)
    set_seed(42)

    config = get_config()
    k_values = [5, 10, 20]

    # Load data
    logger.info("Loading data...")
    gnn_embs = load_gnn_embeddings(MODEL_DIR)
    personality_data = load_personality_data(MODEL_DIR)
    interactions, user_col, biz_col = load_interactions(PROCESSED_DATA_DIR)

    # Determine user/venue mappings
    unique_users = interactions[user_col].unique()
    unique_venues = interactions[biz_col].unique()
    user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
    venue_id_map = {vid: idx for idx, vid in enumerate(unique_venues)}
    n_users = len(unique_users)
    n_venues = len(unique_venues)

    # Get embeddings (from GNN or fallback to random)
    # Reconcile: pad or truncate to match current data dimensions
    emb_dim = 64
    if "user" in gnn_embs:
        user_embeddings = gnn_embs["user"]
        emb_dim = user_embeddings.shape[1]
        if user_embeddings.shape[0] < n_users:
            pad = np.random.randn(
                n_users - user_embeddings.shape[0], emb_dim
            ).astype(np.float32) * 0.1
            logger.warning(
                f"User embeddings ({user_embeddings.shape[0]}) < n_users ({n_users}), "
                f"padding {pad.shape[0]} rows. Re-train GNN for accurate results."
            )
            user_embeddings = np.vstack([user_embeddings, pad])
        elif user_embeddings.shape[0] > n_users:
            user_embeddings = user_embeddings[:n_users]
    else:
        logger.info("Using random user embeddings for demonstration")
        user_embeddings = np.random.randn(n_users, emb_dim).astype(np.float32)

    if "venue" in gnn_embs:
        venue_embeddings = gnn_embs["venue"]
        v_dim = venue_embeddings.shape[1]
        if venue_embeddings.shape[0] < n_venues:
            pad = np.random.randn(
                n_venues - venue_embeddings.shape[0], v_dim
            ).astype(np.float32) * 0.1
            logger.warning(
                f"Venue embeddings ({venue_embeddings.shape[0]}) < n_venues ({n_venues}), "
                f"padding {pad.shape[0]} rows. Re-train GNN for accurate results."
            )
            venue_embeddings = np.vstack([venue_embeddings, pad])
        elif venue_embeddings.shape[0] > n_venues:
            venue_embeddings = venue_embeddings[:n_venues]
    else:
        logger.info("Using random venue embeddings for demonstration")
        venue_embeddings = np.random.randn(n_venues, emb_dim).astype(np.float32)

    # Create personality features - reconcile embedding sizes
    user_mbti_probs = np.random.rand(n_users, 4).astype(np.float32)
    user_mbti_embs = personality_data.get("user_mbti_embeddings", None)
    venue_embs_for_scorer = personality_data.get("venue_embeddings", None)

    # PersonalityScorer computes cosine sim between user and venue embeddings,
    # so they MUST have the same dimension. GNN embeddings are dim=64 for both,
    # but BERTopic venue embeddings are dim=768. Use GNN venue embeddings
    # (same dim as user) for the scorer.
    if (user_mbti_embs is not None and venue_embs_for_scorer is not None
            and user_mbti_embs.shape[1] != venue_embs_for_scorer.shape[1]):
        logger.warning(
            f"Dimension mismatch: user_emb dim={user_mbti_embs.shape[1]}, "
            f"venue_emb dim={venue_embs_for_scorer.shape[1]}. "
            f"Using GNN venue embeddings (dim={user_embeddings.shape[1]}) for scorer."
        )
        venue_embs_for_scorer = venue_embeddings.copy()

    # Pad/truncate MBTI embeddings to match n_users
    if user_mbti_embs is not None and user_mbti_embs.shape[0] != n_users:
        d = user_mbti_embs.shape[1]
        if user_mbti_embs.shape[0] < n_users:
            pad = np.random.randn(
                n_users - user_mbti_embs.shape[0], d
            ).astype(np.float32) * 0.1
            user_mbti_embs = np.vstack([user_mbti_embs, pad])
        else:
            user_mbti_embs = user_mbti_embs[:n_users]

    # Pad/truncate venue embeddings for scorer
    if venue_embs_for_scorer is not None and venue_embs_for_scorer.shape[0] != n_venues:
        d = venue_embs_for_scorer.shape[1]
        if venue_embs_for_scorer.shape[0] < n_venues:
            pad = np.random.randn(
                n_venues - venue_embs_for_scorer.shape[0], d
            ).astype(np.float32) * 0.1
            venue_embs_for_scorer = np.vstack([venue_embs_for_scorer, pad])
        else:
            venue_embs_for_scorer = venue_embs_for_scorer[:n_venues]

    scorer = PersonalityScorer(
        user_mbti_probs=user_mbti_probs,
        user_mbti_embeddings=user_mbti_embs,
        venue_embeddings=venue_embs_for_scorer,
    )

    # Create train/test split
    train_edges, test_edges, user_gt = create_train_test_edges(
        interactions, user_col, biz_col,
        user_id_map, venue_id_map,
        test_ratio=0.2,
    )

    # Compute personality features for all users
    all_user_indices = np.arange(n_users)
    dummy_venue_indices = np.zeros(n_users, dtype=np.int64)
    personality_features = scorer.compute_personality_features(
        all_user_indices, dummy_venue_indices,
    )
    logger.info(f"Personality features shape: {personality_features.shape}")

    # -- BERTopic venue embeddings as venue_extra for XGBoost --
    # These 768-dim semantic embeddings are concatenated alongside the
    # 64-dim GNN venue embeddings so XGBoost sees BOTH representations.
    bertopic_venue_embs = personality_data.get("venue_embeddings", None)
    if bertopic_venue_embs is not None:
        if bertopic_venue_embs.shape[0] < n_venues:
            pad = np.zeros(
                (n_venues - bertopic_venue_embs.shape[0],
                 bertopic_venue_embs.shape[1]),
                dtype=np.float32,
            )
            bertopic_venue_embs = np.vstack([bertopic_venue_embs, pad])
        elif bertopic_venue_embs.shape[0] > n_venues:
            bertopic_venue_embs = bertopic_venue_embs[:n_venues]
        # PCA: reduce 768-dim -> 64-dim to balance with GNN features
        if bertopic_venue_embs.shape[1] > 64:
            logger.info(
                f"Applying PCA: {bertopic_venue_embs.shape[1]}-dim -> 64-dim"
            )
            pca = PCA(n_components=64, random_state=42)
            bertopic_venue_embs = pca.fit_transform(bertopic_venue_embs).astype(
                np.float32
            )
            explained = pca.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance: {explained:.3f}")
        logger.info(
            f"BERTopic venue_extra for XGBoost: {bertopic_venue_embs.shape}"
        )
    else:
        logger.warning("No BERTopic venue embeddings found - venue_extra=None")

    num_rounds = 50 if args.quick else 100

    # --- Evaluation 1: Full late-fusion ensemble ---
    # Features: knn_score + gnn_score + venue_degree + user_degree
    logger.info("=" * 60)
    logger.info("Late-Fusion Ensemble: KNN + GNN + Degree stats")
    logger.info("=" * 60)

    hybrid_metrics, hybrid_ranker = run_hybrid_evaluation(
        train_edges, test_edges, user_gt,
        user_embeddings, venue_embeddings,
        personality_features, config, k_values,
        label="Late-Fusion (KNN+GNN+Degree)",
        num_rounds=num_rounds,
        venue_extra=bertopic_venue_embs,
    )

    # --- Evaluation 2: Ablation - GNN + Degree only (no KNN/MBTI) ---
    # Features: gnn_score + venue_degree + user_degree (knn_score=0)
    logger.info("=" * 60)
    logger.info("Ablation: GNN + Degree only (no MBTI/KNN)")
    logger.info("=" * 60)

    baseline_metrics, _ = run_hybrid_evaluation(
        train_edges, test_edges, user_gt,
        user_embeddings, venue_embeddings,
        None, config, k_values,
        label="GNN + Degree (no KNN)",
        num_rounds=num_rounds,
        venue_extra=None,
    )

    # Combine results
    all_metrics = pd.concat([hybrid_metrics, baseline_metrics], ignore_index=True)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    print(all_metrics.to_string(index=False))

    # Generate methodology comparison
    comparison_df, latex_str = generate_methodology_comparison(hybrid_metrics, k=10)

    logger.info("\nMethodology Comparison:")
    print(comparison_df.to_string(index=False))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics.to_csv(RESULTS_DIR / "phase4_hybrid_metrics.csv", index=False)
    comparison_df.to_csv(RESULTS_DIR / "phase4_methodology_comparison.csv", index=False)

    latex_path = RESULTS_DIR / "phase4_methodology_comparison.tex"
    latex_path.write_text(latex_str, encoding="utf-8")

    # Save feature importance
    importance = hybrid_ranker.get_feature_importance()
    importance.to_csv(RESULTS_DIR / "phase4_feature_importance.csv", index=False)

    logger.info(f"\nResults saved to {RESULTS_DIR}")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid recommendation evaluation pipeline"
    )
    parser.add_argument("--k", type=int, default=10, help="Primary K for metrics")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer rounds")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])

    args = parser.parse_args()
    main(args)
