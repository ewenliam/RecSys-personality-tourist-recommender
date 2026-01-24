"""
Yelp Dataset Loader.

Handles loading and parsing of the Yelp Academic Dataset including:
- Business data (venues)
- Review data (user reviews)
- User data (user profiles)
- Photo metadata
"""
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from tqdm import tqdm

from src.config.settings import DataConfig, get_config

logger = logging.getLogger(__name__)


class YelpDataLoader:
    """Load and manage Yelp dataset files."""

    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the Yelp data loader.

        Args:
            config: Data configuration. Uses default if not provided.
        """
        self.config = config or get_config().data
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.config.business_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_json_lines(self, filepath: Path, limit: Optional[int] = None) -> Iterator[dict]:
        """
        Load JSON lines file lazily.

        Args:
            filepath: Path to the JSON lines file.
            limit: Maximum number of records to load.

        Yields:
            Parsed JSON objects.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    yield json.loads(line.strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {i}: {e}")
                    continue

    def load_businesses(
        self,
        limit: Optional[int] = None,
        categories_filter: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load business data.

        Args:
            limit: Maximum number of businesses to load.
            categories_filter: Filter by category keywords (e.g., ["Restaurant", "Hotel"]).

        Returns:
            DataFrame with business data.
        """
        logger.info(f"Loading businesses from {self.config.business_path}")

        records = []
        for record in tqdm(
            self._load_json_lines(self.config.business_path, limit),
            desc="Loading businesses",
        ):
            # Apply category filter if specified
            if categories_filter:
                categories = record.get("categories", "") or ""
                if not any(cat.lower() in categories.lower() for cat in categories_filter):
                    continue
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} businesses")
        return df

    def load_reviews(
        self,
        limit: Optional[int] = None,
        business_ids: Optional[set[str]] = None,
        user_ids: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        """
        Load review data.

        Args:
            limit: Maximum number of reviews to load.
            business_ids: Filter by business IDs.
            user_ids: Filter by user IDs.

        Returns:
            DataFrame with review data.
        """
        logger.info(f"Loading reviews from {self.config.review_path}")

        records = []
        for record in tqdm(
            self._load_json_lines(self.config.review_path, limit),
            desc="Loading reviews",
        ):
            # Apply filters
            if business_ids and record.get("business_id") not in business_ids:
                continue
            if user_ids and record.get("user_id") not in user_ids:
                continue

            # Filter by review length
            text = record.get("text", "")
            if len(text.split()) < self.config.min_review_length:
                continue

            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} reviews")
        return df

    def load_users(
        self,
        limit: Optional[int] = None,
        user_ids: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        """
        Load user data.

        Args:
            limit: Maximum number of users to load.
            user_ids: Filter by user IDs.

        Returns:
            DataFrame with user data.
        """
        logger.info(f"Loading users from {self.config.user_path}")

        records = []
        for record in tqdm(
            self._load_json_lines(self.config.user_path, limit),
            desc="Loading users",
        ):
            if user_ids and record.get("user_id") not in user_ids:
                continue
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} users")
        return df

    def load_photos(self, business_ids: Optional[set[str]] = None) -> pd.DataFrame:
        """
        Load photo metadata.

        Args:
            business_ids: Filter by business IDs.

        Returns:
            DataFrame with photo metadata.
        """
        photo_json = self.config.photos_path.parent / "photos.json"
        if not photo_json.exists():
            logger.warning(f"Photo metadata not found: {photo_json}")
            return pd.DataFrame()

        logger.info(f"Loading photos from {photo_json}")

        records = []
        for record in tqdm(
            self._load_json_lines(photo_json),
            desc="Loading photos",
        ):
            if business_ids and record.get("business_id") not in business_ids:
                continue
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} photos")
        return df

    def get_photo_path(self, photo_id: str) -> Path:
        """Get the file path for a photo."""
        return self.config.photos_path / f"{photo_id}.jpg"

    def load_all(
        self,
        limit: Optional[int] = None,
        categories_filter: Optional[list[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load all datasets with filtering.

        Args:
            limit: Limit records per dataset.
            categories_filter: Filter businesses by category.

        Returns:
            Dictionary with all DataFrames.
        """
        # Load businesses first to get IDs for filtering
        businesses = self.load_businesses(limit=limit, categories_filter=categories_filter)
        business_ids = set(businesses["business_id"].unique())

        # Load reviews for these businesses
        reviews = self.load_reviews(limit=limit, business_ids=business_ids)
        user_ids = set(reviews["user_id"].unique())

        # Load users who wrote reviews
        users = self.load_users(limit=limit, user_ids=user_ids)

        # Load photos for these businesses
        photos = self.load_photos(business_ids=business_ids)

        return {
            "businesses": businesses,
            "reviews": reviews,
            "users": users,
            "photos": photos,
        }

    def get_dataset_stats(self, data: dict[str, pd.DataFrame]) -> dict:
        """Get statistics about the loaded datasets."""
        stats = {}
        for name, df in data.items():
            stats[name] = {
                "count": len(df),
                "columns": list(df.columns),
                "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            }
        return stats
