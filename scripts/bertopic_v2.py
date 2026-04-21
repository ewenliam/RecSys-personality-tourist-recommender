#!/usr/bin/env python3
"""
BERTopic V2: Sweet-Spot Clustering with Soft Topic Assignment.

Solves two problems from V1:
  1. 40.4% outlier rate (HDBSCAN too aggressive) -> Aim for 5-15%
  2. Hard 0/1 topic assignment -> Soft probability via approximate_distribution

Strategy:
  - Grid search over HDBSCAN min_cluster_size to find the sweet spot
  - Two-stage outlier reduction (embedding + c-TF-IDF)
  - approximate_distribution for soft topic probabilities
  - Export topic_confidence per venue for XGBoost feature engineering

Usage:
    python scripts/bertopic_v2.py
    python scripts/bertopic_v2.py --target-outlier-pct 10 --n-clusters 50
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_config, PROCESSED_DATA_DIR, MODEL_DIR, PROJECT_ROOT
from src.utils.helpers import setup_logging, set_seed

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"


def load_venue_reviews(data_dir: Path) -> tuple[dict, list[str]]:
    """Load reviews grouped by venue."""
    dfs = []
    for split in ["train_reviews.parquet", "val_reviews.parquet"]:
        path = data_dir / split
        if path.exists():
            dfs.append(pd.read_parquet(path))

    if not dfs:
        raise FileNotFoundError(f"No processed data in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    biz_col = "business_id" if "business_id" in df.columns else "venue_id"

    venue_reviews = df.groupby(biz_col)[text_col].apply(list).to_dict()
    venue_ids = list(venue_reviews.keys())

    logger.info(f"Loaded reviews for {len(venue_ids)} venues")
    return venue_reviews, venue_ids


def build_documents(venue_reviews: dict, venue_ids: list[str]) -> list[str]:
    """Build one document per venue from concatenated reviews."""
    documents = []
    for vid in venue_ids:
        reviews = venue_reviews[vid][:5]
        doc = " ".join(str(r)[:500] for r in reviews)
        documents.append(doc)
    return documents


def find_sweet_spot(
    documents: list[str],
    embeddings: np.ndarray,
    target_outlier_pct: float = 10.0,
    candidates: list[int] = None,
) -> tuple[int, float]:
    """
    Grid search over HDBSCAN min_cluster_size to hit target outlier rate.

    Returns the min_cluster_size that gets closest to the target outlier %
    while staying in the 5-15% band if possible.

    Args:
        documents: Pre-built venue documents.
        embeddings: Pre-computed document embeddings.
        target_outlier_pct: Desired outlier percentage.
        candidates: List of min_cluster_size values to try.

    Returns:
        Tuple of (best_min_cluster_size, actual_outlier_pct).
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
    from sklearn.feature_extraction.text import CountVectorizer

    if candidates is None:
        candidates = [5, 8, 10, 15, 20, 30, 50]

    logger.info(f"Searching for sweet spot (target: {target_outlier_pct}% outliers)")
    logger.info(f"Candidates: {candidates}")

    # Shared UMAP reduction (expensive, do once)
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        low_memory=True,
        random_state=42,
    )
    reduced = umap_model.fit_transform(embeddings)

    results = []
    for mcs in candidates:
        hdbscan_model = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=max(1, mcs // 3),
            metric="euclidean",
            prediction_data=True,
        )
        labels = hdbscan_model.fit_predict(reduced)
        n_outliers = (labels == -1).sum()
        outlier_pct = 100.0 * n_outliers / len(labels)
        n_topics = len(set(labels) - {-1})

        results.append({
            "min_cluster_size": mcs,
            "outlier_pct": outlier_pct,
            "n_topics": n_topics,
            "n_outliers": n_outliers,
        })
        logger.info(
            f"  min_cluster_size={mcs:3d}: "
            f"{outlier_pct:5.1f}% outliers, {n_topics} topics"
        )

    # Pick the one closest to target, preferring 5-15% band
    results_df = pd.DataFrame(results)
    in_band = results_df[
        (results_df["outlier_pct"] >= 5) & (results_df["outlier_pct"] <= 15)
    ]

    if len(in_band) > 0:
        best_idx = (in_band["outlier_pct"] - target_outlier_pct).abs().idxmin()
    else:
        best_idx = (results_df["outlier_pct"] - target_outlier_pct).abs().idxmin()

    best = results_df.loc[best_idx]
    logger.info(
        f"Selected min_cluster_size={int(best['min_cluster_size'])}: "
        f"{best['outlier_pct']:.1f}% outliers, {int(best['n_topics'])} topics"
    )

    return int(best["min_cluster_size"]), float(best["outlier_pct"])


def fit_bertopic_v2(
    documents: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int = 15,
    reduce_outliers: bool = True,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> tuple[list[int], np.ndarray, list[bool], "BERTopic"]:
    """
    Fit BERTopic with tuned HDBSCAN and two-stage outlier reduction.

    Returns:
        topics: Topic assignments per document.
        topic_distr: Soft topic distribution [n_docs, n_topics].
        was_outlier: Boolean mask of documents originally assigned to -1.
        model: Fitted BERTopic model.
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from hdbscan import HDBSCAN
    from umap import UMAP
    from sklearn.feature_extraction.text import CountVectorizer

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        low_memory=True,
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, min_cluster_size // 3),
        metric="euclidean",
        prediction_data=True,  # Required for approximate_predict
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=10,
        max_features=5000,
    )

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    representation_model = [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.3),
    ]

    model = BERTopic(
        embedding_model=embedding_model_name,  # Required by KeyBERTInspired for repr docs
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        verbose=True,
    )

    # Initial fit
    logger.info("Fitting BERTopic...")
    topics, probs = model.fit_transform(documents, embeddings=embeddings)

    # Track original outliers BEFORE reduction
    was_outlier = [t == -1 for t in topics]
    n_outliers_before = sum(was_outlier)
    logger.info(
        f"Initial: {n_outliers_before}/{len(topics)} outliers "
        f"({100 * n_outliers_before / len(topics):.1f}%)"
    )

    # Two-stage outlier reduction
    if reduce_outliers and n_outliers_before > 0:
        # Stage 1: Embedding-based
        try:
            topics = model.reduce_outliers(
                documents, topics,
                strategy="embeddings",
                embeddings=embeddings,
                threshold=0.0,
            )
            n_after_emb = sum(1 for t in topics if t == -1)
            logger.info(f"After embedding reduction: {n_after_emb} outliers")
        except Exception as e:
            logger.warning(f"Embedding reduction failed: {e}")

        # Stage 2: c-TF-IDF-based
        n_still = sum(1 for t in topics if t == -1)
        if n_still > 0:
            try:
                topics = model.reduce_outliers(
                    documents, topics,
                    strategy="c-tf-idf",
                    threshold=0.0,
                )
                n_after_ctfidf = sum(1 for t in topics if t == -1)
                logger.info(f"After c-TF-IDF reduction: {n_after_ctfidf} outliers")
            except Exception as e:
                logger.warning(f"c-TF-IDF reduction failed: {e}")

        # Update internal state
        model.update_topics(documents, topics=topics)

    n_final = sum(1 for t in topics if t == -1)
    logger.info(
        f"Final: {n_final}/{len(topics)} outliers "
        f"({100 * n_final / len(topics):.1f}%), "
        f"{len(set(topics) - {-1})} topics"
    )

    # Soft topic distribution via approximate_distribution
    logger.info("Computing soft topic distribution via approximate_distribution...")
    try:
        topic_distr, _ = model.approximate_distribution(
            documents, min_similarity=0.0
        )
        logger.info(f"Topic distribution shape: {topic_distr.shape}")
    except Exception as e:
        logger.warning(f"approximate_distribution failed ({e}), using hard assignments")
        n_topics = len(set(topics) - {-1})
        topic_distr = np.zeros((len(topics), max(n_topics, 1)))
        for i, t in enumerate(topics):
            if t >= 0 and t < topic_distr.shape[1]:
                topic_distr[i, t] = 1.0

    return topics, topic_distr, was_outlier, model


def build_topic_confidence(
    topics: list[int],
    topic_distr: np.ndarray,
    was_outlier: list[bool],
    outlier_penalty: float = 0.1,
) -> np.ndarray:
    """
    Build per-venue topic confidence score.

    Venues with strong topic assignment get high confidence.
    Ex-outlier venues get penalized so XGBoost learns to rely
    on GNN embeddings instead of topic features for those venues.

    Args:
        topics: Topic assignments.
        topic_distr: Soft topic distribution [n_docs, n_topics].
        was_outlier: Whether the venue was originally topic -1.
        outlier_penalty: Max confidence for ex-outlier venues.

    Returns:
        Confidence array [n_venues, 1].
    """
    n = len(topics)
    confidence = np.ones((n, 1), dtype=np.float32)

    for i in range(n):
        if was_outlier[i]:
            # Ex-outlier: use the max topic probability (capped)
            max_prob = topic_distr[i].max() if topic_distr.shape[1] > 0 else 0.0
            confidence[i, 0] = min(float(max_prob), outlier_penalty)
        else:
            # Original topic: use the assigned topic's probability
            t = topics[i]
            if 0 <= t < topic_distr.shape[1]:
                confidence[i, 0] = float(topic_distr[i, t])
            else:
                confidence[i, 0] = 1.0

    return confidence


def main(args):
    """Run BERTopic V2 pipeline."""
    setup_logging(level=logging.INFO)
    set_seed(42)

    config = get_config()

    # Load venue reviews
    venue_reviews, venue_ids = load_venue_reviews(PROCESSED_DATA_DIR)
    documents = build_documents(venue_reviews, venue_ids)
    logger.info(f"Built {len(documents)} venue documents")

    # Compute embeddings
    logger.info("Computing embeddings...")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(
        config.bertopic.embedding_model, device=args.device
    )
    embeddings = st_model.encode(
        documents, show_progress_bar=True, batch_size=64
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Find sweet spot min_cluster_size
    if args.auto_tune:
        best_mcs, actual_pct = find_sweet_spot(
            documents, embeddings,
            target_outlier_pct=args.target_outlier_pct,
        )
    else:
        best_mcs = args.min_cluster_size

    # Fit BERTopic V2
    topics, topic_distr, was_outlier, model = fit_bertopic_v2(
        documents, embeddings,
        min_cluster_size=best_mcs,
        reduce_outliers=True,
        embedding_model_name=config.bertopic.embedding_model,
    )

    # Build topic confidence
    confidence = build_topic_confidence(
        topics, topic_distr, was_outlier,
        outlier_penalty=config.bertopic.outlier_confidence,
    )

    # Build output DataFrame
    topic_info = model.get_topic_info()
    topic_names = {}
    for _, row in topic_info.iterrows():
        topic_names[row["Topic"]] = row["Name"]

    venue_topics = pd.DataFrame({
        "venue_id": venue_ids,
        "topic": topics,
        "topic_prob": [
            float(topic_distr[i, t]) if 0 <= t < topic_distr.shape[1] else 0.0
            for i, t in enumerate(topics)
        ],
        "topic_confidence": confidence.flatten(),
        "was_outlier": was_outlier,
        "topic_name": [topic_names.get(t, "Outlier") for t in topics],
    })

    # Save everything
    save_dir = MODEL_DIR / "bertopic_v2"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(save_dir / "topic_model"))
    np.save(save_dir / "venue_embeddings.npy", embeddings)
    np.save(save_dir / "topic_distribution.npy", topic_distr)
    np.save(save_dir / "topic_confidence.npy", confidence)
    venue_topics.to_parquet(save_dir / "venue_topics.parquet", index=False)

    # Save search results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Summary statistics
    n_topics = len(set(topics) - {-1})
    n_outliers = sum(1 for t in topics if t == -1)
    n_ex_outliers = sum(was_outlier)

    summary = pd.DataFrame([{
        "Metric": "Total venues",
        "Value": len(venue_ids),
    }, {
        "Metric": "Topics found",
        "Value": n_topics,
    }, {
        "Metric": "Final outliers",
        "Value": f"{n_outliers} ({100 * n_outliers / len(venue_ids):.1f}%)",
    }, {
        "Metric": "Originally outliers (before reduction)",
        "Value": f"{n_ex_outliers} ({100 * n_ex_outliers / len(venue_ids):.1f}%)",
    }, {
        "Metric": "Mean topic confidence (non-outlier)",
        "Value": f"{confidence[~np.array(was_outlier)].mean():.4f}",
    }, {
        "Metric": "Mean topic confidence (ex-outlier)",
        "Value": f"{confidence[np.array(was_outlier)].mean():.4f}" if any(was_outlier) else "N/A",
    }, {
        "Metric": "min_cluster_size used",
        "Value": best_mcs,
    }])

    summary.to_csv(RESULTS_DIR / "bertopic_v2_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    logger.info(f"\nSaved to {save_dir}")
    logger.info(f"Summary saved to {RESULTS_DIR / 'bertopic_v2_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERTopic V2: Sweet-Spot Clustering")
    parser.add_argument("--target-outlier-pct", type=float, default=10.0,
                        help="Target outlier percentage for grid search")
    parser.add_argument("--min-cluster-size", type=int, default=15,
                        help="HDBSCAN min_cluster_size (used if --no-auto-tune)")
    parser.add_argument("--auto-tune", action="store_true", default=True,
                        help="Auto-tune min_cluster_size via grid search")
    parser.add_argument("--no-auto-tune", dest="auto_tune", action="store_false")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"])

    args = parser.parse_args()
    main(args)
