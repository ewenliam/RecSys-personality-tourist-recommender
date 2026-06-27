#!/usr/bin/env python3
"""
Phase 2: BERTopic with BERT-MBTI Embeddings.

Extracts venue topics using the fine-tuned BERT-MBTI model as the
embedding backbone instead of generic sentence-transformers. The
personality-informed embeddings produce semantically richer topic
clusters aligned with user behavioral patterns.

Usage:
    python scripts/extract_topics_mbti.py
    python scripts/extract_topics_mbti.py --n-topics 30 --checkpoint path/to/model.pt
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# UMAP -> numba tries to link Intel SVML symbols (e.g. __svml_sqrtf16) that are
# not present in this environment, causing a hard "LLVM ERROR: Symbol not
# found" crash.  Disabling SVML makes numba fall back to standard math.  Must
# be set BEFORE numba is first imported (it is imported lazily by UMAP).
os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import (
    get_config,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    PROJECT_ROOT,
)
from src.models.bertopic.mbti_embedder import MBTIEmbedder
from src.models.bertopic.topic_extractor import VenueTopicExtractor
from src.utils.helpers import setup_logging, set_seed

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"


def load_reviews(data_dir: Path) -> pd.DataFrame:
    """Load review data from processed parquet files."""
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

    text_col = "clean_text" if "clean_text" in df.columns else "text"
    biz_col = "business_id" if "business_id" in df.columns else "venue_id"

    logger.info(f"Loaded {len(df)} reviews for {df[biz_col].nunique()} venues")
    return df, text_col, biz_col


def extract_topics_with_mbti(
    reviews_df: pd.DataFrame,
    text_col: str,
    biz_col: str,
    mbti_checkpoint: Path,
    n_clusters: int = 50,
    use_kmeans: bool = True,
) -> tuple[pd.DataFrame, VenueTopicExtractor, np.ndarray]:
    """
    Extract venue topics using BERT-MBTI embeddings.

    Args:
        reviews_df: DataFrame with review text and business IDs.
        text_col: Column name for review text.
        biz_col: Column name for business/venue ID.
        mbti_checkpoint: Path to trained MBTI model checkpoint.
        n_clusters: Number of topic clusters.
        use_kmeans: Use KMeans instead of HDBSCAN.

    Returns:
        Tuple of (venue_topics DataFrame, extractor, venue_embeddings).
    """
    config = get_config()

    # Create MBTI-informed embedding model
    logger.info("Loading BERT-MBTI embedding model...")
    mbti_embedder = MBTIEmbedder(
        model_path=mbti_checkpoint,
        config=config.bert,
    )

    # Group reviews by venue
    venue_reviews = (
        reviews_df.groupby(biz_col)[text_col]
        .apply(list)
        .to_dict()
    )
    logger.info(f"Grouped reviews for {len(venue_reviews)} venues")

    # Pre-compute MBTI-informed venue embeddings
    logger.info("Computing MBTI-informed venue embeddings...")
    venue_ids, venue_embeddings = mbti_embedder.encode_venues(
        venue_reviews,
        max_reviews_per_venue=5,
        max_chars_per_review=500,
        batch_size=config.bert.batch_size,
    )

    # Build BERTopic model with MBTI embedder as the embedding model
    logger.info("Building BERTopic model with MBTI embeddings...")
    topic_config = config.bertopic
    topic_config.kmeans_n_clusters = n_clusters

    extractor = VenueTopicExtractor(config=topic_config)
    extractor.build_model(use_kmeans=use_kmeans)

    # Override the embedding model with our MBTI embedder
    extractor.topic_model.embedding_model = mbti_embedder

    # Create documents from venue reviews (matching order of venue_ids)
    documents = []
    for vid in venue_ids:
        reviews = venue_reviews[vid][:5]
        doc = " ".join(r[:500] for r in reviews)
        documents.append(doc)

    # Fit with pre-computed embeddings
    logger.info("Fitting BERTopic with MBTI embeddings...")
    topics, probs = extractor.fit(documents, embeddings=venue_embeddings)

    if probs is None:
        probs = [1.0] * len(topics)

    # Build result DataFrame
    result = pd.DataFrame({
        "venue_id": venue_ids,
        "topic": topics,
        "topic_prob": [p.max() if isinstance(p, np.ndarray) else p for p in probs],
    })

    # Add topic names
    topic_info = extractor.get_topic_info()
    topic_names = {t.topic_id: t.name for t in topic_info}
    result["topic_name"] = result["topic"].map(topic_names)

    n_topics = len(set(topics) - {-1})
    logger.info(f"Extracted {n_topics} topics from {len(venue_ids)} venues")

    return result, extractor, venue_embeddings


def extract_topics_generic(
    reviews_df: pd.DataFrame,
    text_col: str,
    biz_col: str,
    n_clusters: int = 50,
    use_kmeans: bool = True,
) -> tuple[pd.DataFrame, VenueTopicExtractor, np.ndarray]:
    """
    Extract venue topics using generic sentence-transformers (baseline).

    For comparison with the MBTI-informed approach.
    """
    config = get_config()
    topic_config = config.bertopic
    topic_config.kmeans_n_clusters = n_clusters

    venue_reviews = (
        reviews_df.groupby(biz_col)[text_col]
        .apply(list)
        .to_dict()
    )

    extractor = VenueTopicExtractor(config=topic_config)
    extractor.build_model(use_kmeans=use_kmeans)

    result = extractor.fit_transform_venues(venue_reviews)

    venue_embeddings = extractor.embeddings
    return result, extractor, venue_embeddings


def compare_topic_quality(
    mbti_topics: pd.DataFrame,
    generic_topics: pd.DataFrame,
) -> pd.DataFrame:
    """Compare topic quality between MBTI and generic approaches."""
    rows = []

    for label, df in [("MBTI-Informed", mbti_topics), ("Generic (all-MiniLM)", generic_topics)]:
        n_topics = len(set(df["topic"]) - {-1})
        n_outliers = (df["topic"] == -1).sum()
        avg_prob = df[df["topic"] != -1]["topic_prob"].mean()
        topic_sizes = df[df["topic"] != -1].groupby("topic").size()

        rows.append({
            "Embedding Model": label,
            "Num Topics": n_topics,
            "Outlier Venues": n_outliers,
            "Outlier %": f"{100 * n_outliers / len(df):.1f}%",
            "Avg Topic Confidence": f"{avg_prob:.4f}",
            "Avg Topic Size": f"{topic_sizes.mean():.1f}",
            "Topic Size Std": f"{topic_sizes.std():.1f}",
        })

    return pd.DataFrame(rows)


def main(args):
    """Run MBTI-informed topic extraction."""
    setup_logging(level=logging.INFO)
    set_seed(42)

    # Load reviews
    reviews_df, text_col, biz_col = load_reviews(PROCESSED_DATA_DIR)

    # Determine checkpoint path
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        CHECKPOINT_DIR / "bert_mbti_robust" / "best_model.pt"
    )

    # Extract topics with MBTI embeddings
    logger.info("=" * 60)
    logger.info("MBTI-Informed BERTopic Extraction")
    logger.info("=" * 60)

    mbti_topics, mbti_extractor, mbti_venue_embs = extract_topics_with_mbti(
        reviews_df, text_col, biz_col,
        mbti_checkpoint=checkpoint,
        n_clusters=args.n_topics,
    )

    # Save MBTI topic model and embeddings
    save_dir = MODEL_DIR / "bertopic_mbti"
    save_dir.mkdir(parents=True, exist_ok=True)

    mbti_extractor.save(save_dir / "topic_model")
    np.save(save_dir / "venue_embeddings.npy", mbti_venue_embs)
    mbti_topics.to_parquet(save_dir / "venue_topics.parquet", index=False)

    # Save topic info
    topic_info = mbti_extractor.get_topic_info()
    topic_info_df = pd.DataFrame([
        {"topic_id": t.topic_id, "name": t.name, "count": t.count,
         "keywords": ", ".join(t.representation[:5])}
        for t in topic_info
    ])
    topic_info_df.to_csv(save_dir / "topic_info.csv", index=False)

    logger.info(f"Saved MBTI topic model to {save_dir}")

    # Optionally compare with generic baseline
    if args.compare:
        logger.info("=" * 60)
        logger.info("Generic Baseline BERTopic Extraction")
        logger.info("=" * 60)

        generic_topics, generic_extractor, _ = extract_topics_generic(
            reviews_df, text_col, biz_col,
            n_clusters=args.n_topics,
        )

        comparison = compare_topic_quality(mbti_topics, generic_topics)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(RESULTS_DIR / "phase2_topic_comparison.csv", index=False)

        logger.info("\nTopic Quality Comparison:")
        print(comparison.to_string(index=False))

    # Print topic summary
    logger.info("\nExtracted Topics:")
    for t in topic_info[:10]:
        logger.info(f"  Topic {t.topic_id}: {t.name} ({t.count} venues)")
        logger.info(f"    Keywords: {', '.join(t.representation[:5])}")

    logger.info(f"\nDone. Venue topics saved to {save_dir / 'venue_topics.parquet'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract venue topics using BERT-MBTI embeddings"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained MBTI model checkpoint",
    )
    parser.add_argument(
        "--n-topics", type=int, default=50,
        help="Number of topic clusters (default: 50)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also run generic baseline for comparison",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu", "mps"],
    )

    args = parser.parse_args()
    main(args)
