#!/usr/bin/env python3
"""
Data Preparation Script.

Downloads and preprocesses the Yelp dataset for the recommendation system.
"""
import argparse
import logging
from pathlib import Path

from src.data import YelpDataLoader, TextPreprocessor, DataSampler
from src.config import get_config
from src.config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def main(args):
    """Main data preparation pipeline."""
    setup_logging(level=logging.INFO)
    config = get_config()

    # Check if raw data exists
    if not RAW_DATA_DIR.exists():
        logger.error(f"Raw data directory not found: {RAW_DATA_DIR}")
        logger.info(
            "Please download the Yelp dataset from: "
            "https://www.yelp.com/dataset"
        )
        logger.info(f"Extract the files to: {RAW_DATA_DIR}")
        return

    # Initialize components
    loader = YelpDataLoader()
    preprocessor = TextPreprocessor()
    sampler = DataSampler()

    # Define tourism-related categories
    tourism_categories = [
        "Restaurant", "Hotel", "Museum", "Park", "Bar",
        "Cafe", "Attraction", "Entertainment", "Shopping",
        "Nightlife", "Arts", "Tours", "Landmarks",
    ]

    logger.info("Loading Yelp dataset...")

    # Load data
    data = loader.load_all(
        limit=args.limit,
        categories_filter=tourism_categories if args.filter_tourism else None,
    )

    # Print stats
    stats = loader.get_dataset_stats(data)
    for name, info in stats.items():
        logger.info(f"{name}: {info['count']:,} records ({info['memory_mb']:.1f} MB)")

    # Preprocess reviews
    logger.info("Preprocessing reviews...")
    reviews = preprocessor.process_dataframe(
        data["reviews"],
        text_column="text",
        output_column="clean_text",
    )

    # Filter by activity (use CLI overrides if provided)
    min_user = args.min_user_reviews or config.data.min_reviews_per_user
    min_biz = args.min_business_reviews or config.data.min_reviews_per_business
    logger.info(
        f"Filtering by activity: min_user_reviews={min_user}, "
        f"min_business_reviews={min_biz}"
    )
    reviews = sampler.filter_by_activity(
        reviews,
        min_user_reviews=min_user,
        min_business_reviews=min_biz,
    )

    # Split data
    logger.info("Splitting into train/val/test...")
    train, val, test = sampler.train_val_test_split(reviews)

    # Save processed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed data to {PROCESSED_DATA_DIR}")
    train.to_parquet(PROCESSED_DATA_DIR / "train_reviews.parquet", index=False)
    val.to_parquet(PROCESSED_DATA_DIR / "val_reviews.parquet", index=False)
    test.to_parquet(PROCESSED_DATA_DIR / "test_reviews.parquet", index=False)
    data["businesses"].to_parquet(PROCESSED_DATA_DIR / "businesses.parquet", index=False)
    data["users"].to_parquet(PROCESSED_DATA_DIR / "users.parquet", index=False)

    logger.info("Data preparation complete!")
    logger.info(f"  Train: {len(train):,} reviews")
    logger.info(f"  Val: {len(val):,} reviews")
    logger.info(f"  Test: {len(test):,} reviews")
    logger.info(f"  Businesses: {len(data['businesses']):,}")
    logger.info(f"  Users: {len(data['users']):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Yelp dataset")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to load (for testing)",
    )
    parser.add_argument(
        "--filter-tourism",
        action="store_true",
        default=True,
        help="Filter to tourism-related categories",
    )
    parser.add_argument(
        "--no-filter-tourism",
        action="store_false",
        dest="filter_tourism",
        help="Do not filter by categories",
    )
    parser.add_argument(
        "--min-user-reviews",
        type=int,
        default=None,
        help="Override min reviews per user (default from config: 5)",
    )
    parser.add_argument(
        "--min-business-reviews",
        type=int,
        default=None,
        help="Override min reviews per business (default from config: 3)",
    )

    args = parser.parse_args()
    main(args)
