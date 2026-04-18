#!/usr/bin/env python3
"""
Extract Topics from Venue Reviews.

Usage:
    python scripts/extract_topics.py --n-topics 50
    python scripts/extract_topics.py --use-multimodal --n-topics 30
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import get_config
from src.config.settings import PROCESSED_DATA_DIR, MODEL_DIR
from src.models.bertopic import (
    VenueTopicExtractor,
    MultimodalEmbedder,
    cluster_venues_by_location,
)
from src.utils.helpers import setup_logging, set_seed

logger = logging.getLogger(__name__)


def main(args):
    """Main topic extraction function."""
    setup_logging(level=logging.INFO)
    config = get_config()
    set_seed(42)

    # Load data
    logger.info("Loading data...")

    reviews_path = PROCESSED_DATA_DIR / "train_reviews.parquet"
    businesses_path = PROCESSED_DATA_DIR / "businesses.parquet"

    if not reviews_path.exists():
        raise FileNotFoundError(
            f"Reviews not found: {reviews_path}\n"
            "Run 'python scripts/prepare_data.py' first."
        )

    reviews_df = pd.read_parquet(reviews_path)
    businesses_df = pd.read_parquet(businesses_path) if businesses_path.exists() else None

    logger.info(f"Loaded {len(reviews_df)} reviews")
    if businesses_df is not None:
        logger.info(f"Loaded {len(businesses_df)} businesses")

    # Determine text column
    text_col = "clean_text" if "clean_text" in reviews_df.columns else "text"

    # Configure
    config.bertopic.kmeans_n_clusters = args.n_topics

    # Create output directory
    output_dir = MODEL_DIR / "bertopic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract topics
    logger.info("Extracting venue topics...")

    if args.use_multimodal and businesses_df is not None:
        # Use multimodal embeddings
        logger.info("Using multimodal embeddings (text + images)")

        # Group reviews by venue
        venue_reviews = reviews_df.groupby("business_id")[text_col].apply(list).to_dict()

        # Create multimodal embedder
        embedder = MultimodalEmbedder(
            text_model=config.bertopic.embedding_model,
            device="cuda" if args.device == "cuda" else "cpu",
        )

        # Generate venue embeddings (text only for now, images would need photo paths)
        venue_ids = list(venue_reviews.keys())
        venue_texts = [venue_reviews[vid] for vid in venue_ids]

        logger.info("Generating venue embeddings...")
        embeddings = embedder.text_embedder.encode_aggregated(
            venue_texts,
            aggregation="mean",
            batch_size=args.batch_size,
        )

        # Create extractor and fit
        extractor = VenueTopicExtractor(config=config.bertopic)
        extractor.build_model(use_kmeans=True)

        # Aggregate reviews for topic modeling
        documents = [" ".join(reviews[:10]) for reviews in venue_texts]
        topics, probs = extractor.fit(documents, embeddings=embeddings)

        # Create result DataFrame
        venue_topics = pd.DataFrame({
            "venue_id": venue_ids,
            "topic": topics,
            "topic_prob": [p.max() if isinstance(p, np.ndarray) else p for p in probs],
        })

    else:
        # Text-only topic extraction
        logger.info("Using text-only embeddings")

        # Group reviews by venue
        venue_reviews = reviews_df.groupby("business_id")[text_col].apply(list).to_dict()

        # Create and fit extractor
        extractor = VenueTopicExtractor(config=config.bertopic)
        venue_topics = extractor.fit_transform_venues(venue_reviews)

    # Add topic names
    topic_info = extractor.get_topic_info()
    topic_names = {t.topic_id: t.name for t in topic_info}
    venue_topics["topic_name"] = venue_topics["topic"].map(topic_names)

    # Print topic summary
    logger.info(f"\nExtracted {len(topic_info)} topics:")
    for info in topic_info[:10]:
        logger.info(f"  Topic {info.topic_id}: {info.name} ({info.count} venues)")
        logger.info(f"    Keywords: {', '.join(info.representation[:5])}")

    # Save topic model and results
    extractor.save(output_dir / "topic_model")
    venue_topics.to_parquet(output_dir / "venue_topics.parquet", index=False)
    logger.info(f"Saved topic model to {output_dir}")

    # Step 2: Geographic clustering (if business data available)
    if businesses_df is not None and "latitude" in businesses_df.columns:
        logger.info("\nClustering venues by location...")

        venue_regions, geo_clusterer = cluster_venues_by_location(
            businesses_df,
            lat_col="latitude",
            lon_col="longitude",
            id_col="business_id",
            min_cluster_size=args.min_region_size,
            save_path=output_dir / "venue_regions.parquet",
        )

        logger.info(f"Found {len(geo_clusterer.regions)} geographic regions")
        for region in geo_clusterer.regions[:5]:
            logger.info(
                f"  {region.name}: {region.venue_count} venues, "
                f"radius {region.radius_km:.1f}km"
            )

    # Step 3: Generate visualizations
    if args.visualize:
        logger.info("\nGenerating visualizations...")

        try:
            extractor.visualize_topics(output_dir / "topic_visualization.html")
            extractor.visualize_barchart(
                top_n_topics=min(20, len(topic_info)),
                output_path=output_dir / "topic_barchart.html",
            )
            logger.info(f"Saved visualizations to {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Topic Extraction Complete!")
    logger.info(f"  Venues processed: {len(venue_topics)}")
    logger.info(f"  Topics extracted: {len(topic_info)}")
    logger.info(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract venue topics")

    parser.add_argument(
        "--n-topics",
        type=int,
        default=50,
        help="Number of topics to extract",
    )
    parser.add_argument(
        "--use-multimodal",
        action="store_true",
        help="Use multimodal embeddings (text + images)",
    )
    parser.add_argument(
        "--min-region-size",
        type=int,
        default=10,
        help="Minimum venues per geographic region",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device for computation",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Generate topic visualizations",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Skip visualizations",
    )

    args = parser.parse_args()
    main(args)
