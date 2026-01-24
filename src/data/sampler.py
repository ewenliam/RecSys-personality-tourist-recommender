"""
Data Sampling Utilities.

Handles train/val/test splitting, upsampling for class imbalance,
and negative sampling for training.
"""
import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import DataConfig, get_config

logger = logging.getLogger(__name__)


class DataSampler:
    """Sampling utilities for the recommendation system."""

    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the sampler.

        Args:
            config: Data configuration.
        """
        self.config = config or get_config().data

    def train_val_test_split(
        self,
        df: pd.DataFrame,
        stratify_column: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: DataFrame to split.
            stratify_column: Column to stratify by (for balanced splits).

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None

        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=self.config.test_split,
            random_state=self.config.random_seed,
            stratify=stratify,
        )

        # Update stratify for second split
        if stratify is not None:
            stratify = train_val[stratify_column]

        # Second split: train vs val
        val_ratio = self.config.val_split / (self.config.train_split + self.config.val_split)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=self.config.random_seed,
            stratify=stratify,
        )

        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test

    def upsample_minority_classes(
        self,
        df: pd.DataFrame,
        label_column: str,
        target_ratio: float = 1.0,
    ) -> pd.DataFrame:
        """
        Upsample minority classes to balance the dataset.

        Args:
            df: DataFrame with imbalanced classes.
            label_column: Column containing class labels.
            target_ratio: Target ratio relative to majority class (1.0 = fully balanced).

        Returns:
            DataFrame with upsampled minority classes.
        """
        if label_column not in df.columns:
            logger.warning(f"Label column '{label_column}' not found, skipping upsampling")
            return df

        # Count classes
        class_counts = Counter(df[label_column])
        max_count = max(class_counts.values())
        target_count = int(max_count * target_ratio)

        logger.info(f"Class distribution before upsampling: {dict(class_counts)}")

        # Upsample each minority class
        upsampled_dfs = []
        for label, count in class_counts.items():
            class_df = df[df[label_column] == label]

            if count < target_count:
                # Upsample with replacement
                additional_samples = target_count - count
                upsampled = class_df.sample(
                    n=additional_samples,
                    replace=True,
                    random_state=self.config.random_seed,
                )
                class_df = pd.concat([class_df, upsampled], ignore_index=True)

            upsampled_dfs.append(class_df)

        result = pd.concat(upsampled_dfs, ignore_index=True)
        result = result.sample(frac=1, random_state=self.config.random_seed).reset_index(drop=True)

        new_counts = Counter(result[label_column])
        logger.info(f"Class distribution after upsampling: {dict(new_counts)}")

        return result

    def filter_by_activity(
        self,
        reviews_df: pd.DataFrame,
        min_user_reviews: Optional[int] = None,
        min_business_reviews: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Filter reviews to keep only active users and popular businesses.

        Args:
            reviews_df: DataFrame with reviews.
            min_user_reviews: Minimum reviews per user.
            min_business_reviews: Minimum reviews per business.

        Returns:
            Filtered DataFrame.
        """
        min_user = min_user_reviews or self.config.min_reviews_per_user
        min_biz = min_business_reviews or self.config.min_reviews_per_business

        original_count = len(reviews_df)

        # Filter by user activity
        user_counts = reviews_df["user_id"].value_counts()
        active_users = user_counts[user_counts >= min_user].index
        reviews_df = reviews_df[reviews_df["user_id"].isin(active_users)]

        # Filter by business popularity
        biz_counts = reviews_df["business_id"].value_counts()
        popular_biz = biz_counts[biz_counts >= min_biz].index
        reviews_df = reviews_df[reviews_df["business_id"].isin(popular_biz)]

        logger.info(
            f"Filtered from {original_count} to {len(reviews_df)} reviews "
            f"({len(active_users)} users, {len(popular_biz)} businesses)"
        )

        return reviews_df


class NegativeSampler:
    """Negative sampling for training GNN and ranking models."""

    def __init__(
        self,
        user_ids: list,
        item_ids: list,
        interactions: set[tuple],
        num_negatives: int = 4,
        seed: int = 42,
    ):
        """
        Initialize the negative sampler.

        Args:
            user_ids: List of all user IDs.
            item_ids: List of all item IDs.
            interactions: Set of (user_id, item_id) positive pairs.
            num_negatives: Number of negatives per positive.
            seed: Random seed.
        """
        self.user_ids = list(user_ids)
        self.item_ids = list(item_ids)
        self.interactions = interactions
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

        # Build user->items mapping for hard negative mining
        self.user_items = {}
        for user_id, item_id in interactions:
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            self.user_items[user_id].add(item_id)

    def sample_negatives(self, user_id: str, num: Optional[int] = None) -> list:
        """
        Sample negative items for a user.

        Args:
            user_id: User to sample negatives for.
            num: Number of negatives (uses default if not specified).

        Returns:
            List of negative item IDs.
        """
        num = num or self.num_negatives
        positive_items = self.user_items.get(user_id, set())
        available_items = [i for i in self.item_ids if i not in positive_items]

        if len(available_items) < num:
            return available_items

        return list(self.rng.choice(available_items, size=num, replace=False))

    def sample_hard_negatives(
        self,
        user_id: str,
        item_embeddings: dict,
        positive_item: str,
        num: Optional[int] = None,
    ) -> list:
        """
        Sample hard negatives (semantically similar but not interacted).

        Args:
            user_id: User to sample negatives for.
            item_embeddings: Dictionary of item_id -> embedding.
            positive_item: The positive item to find similar negatives for.
            num: Number of negatives.

        Returns:
            List of hard negative item IDs.
        """
        num = num or self.num_negatives
        positive_items = self.user_items.get(user_id, set())

        if positive_item not in item_embeddings:
            return self.sample_negatives(user_id, num)

        # Get positive embedding
        pos_emb = np.array(item_embeddings[positive_item])

        # Calculate similarities to all non-positive items
        candidates = []
        for item_id in self.item_ids:
            if item_id in positive_items or item_id not in item_embeddings:
                continue

            item_emb = np.array(item_embeddings[item_id])
            similarity = np.dot(pos_emb, item_emb) / (
                np.linalg.norm(pos_emb) * np.linalg.norm(item_emb) + 1e-8
            )
            candidates.append((item_id, similarity))

        # Sort by similarity (descending) and take top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        hard_negatives = [c[0] for c in candidates[:num]]

        return hard_negatives

    def generate_training_pairs(
        self,
        interactions: list[tuple],
    ) -> list[dict]:
        """
        Generate training pairs with negative samples.

        Args:
            interactions: List of (user_id, item_id, label) tuples.

        Returns:
            List of training examples with negatives.
        """
        training_data = []

        for user_id, item_id, *extra in interactions:
            # Positive example
            training_data.append({
                "user_id": user_id,
                "item_id": item_id,
                "label": 1,
                **({"extra": extra} if extra else {}),
            })

            # Negative examples
            negatives = self.sample_negatives(user_id)
            for neg_item in negatives:
                training_data.append({
                    "user_id": user_id,
                    "item_id": neg_item,
                    "label": 0,
                })

        return training_data
