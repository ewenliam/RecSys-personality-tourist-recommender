"""
DCI Closed Algorithm for Frequent Closed Itemset Mining.

Mines frequent closed itemsets from user visit history to create
compact behavioral patterns. A closed itemset is one where no
superset has the same support.

Reference:
    Lucchese et al. "DCI_Closed: A Fast and Memory Efficient Algorithm
    for Mining Frequent Closed Itemsets"
"""
import logging
from typing import Optional, Set, Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.settings import HybridConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class ClosedItemset:
    """A closed itemset with its support."""
    items: frozenset
    support: float
    count: int


class DCIClosed:
    """
    DCI Closed algorithm for mining frequent closed itemsets.

    Optimized for user-venue interaction patterns.
    """

    def __init__(
        self,
        min_support: float = 0.01,
        max_itemset_size: int = 5,
        config: Optional[HybridConfig] = None,
    ):
        """
        Initialize the miner.

        Args:
            min_support: Minimum support threshold (fraction of transactions).
            max_itemset_size: Maximum itemset size to mine.
            config: Hybrid configuration.
        """
        self.config = config or get_config().hybrid
        self.min_support = min_support or self.config.min_support
        self.max_itemset_size = max_itemset_size or self.config.max_itemset_size

        self.transactions: List[Set] = []
        self.closed_itemsets: List[ClosedItemset] = []
        self.item_counts: Dict = {}

    def fit(self, transactions: List[Set]) -> List[ClosedItemset]:
        """
        Mine closed itemsets from transactions.

        Args:
            transactions: List of transaction sets (e.g., venues visited by each user).

        Returns:
            List of closed itemsets.
        """
        self.transactions = transactions
        n_transactions = len(transactions)

        if n_transactions == 0:
            return []

        min_count = int(self.min_support * n_transactions)
        logger.info(f"Mining closed itemsets from {n_transactions} transactions")
        logger.info(f"Min support: {self.min_support} ({min_count} transactions)")

        # Count individual items
        self.item_counts = defaultdict(int)
        for trans in transactions:
            for item in trans:
                self.item_counts[item] += 1

        # Filter frequent items
        frequent_items = {
            item for item, count in self.item_counts.items()
            if count >= min_count
        }
        logger.info(f"Frequent items: {len(frequent_items)}")

        # Mine closed itemsets using depth-first search
        self.closed_itemsets = []
        self._mine_closed(
            prefix=frozenset(),
            items=sorted(frequent_items, key=lambda x: self.item_counts[x]),
            transactions=transactions,
            min_count=min_count,
        )

        logger.info(f"Found {len(self.closed_itemsets)} closed itemsets")
        return self.closed_itemsets

    def _mine_closed(
        self,
        prefix: frozenset,
        items: List,
        transactions: List[Set],
        min_count: int,
    ) -> None:
        """Recursive mining of closed itemsets."""
        if len(prefix) >= self.max_itemset_size:
            return

        for i, item in enumerate(items):
            # Create new itemset
            new_prefix = prefix | {item}

            # Count support
            count = sum(1 for trans in transactions if new_prefix <= trans)

            if count < min_count:
                continue

            # Check if closed (no immediate superset has same support)
            is_closed = True
            for other_item in items[i + 1:]:
                extended = new_prefix | {other_item}
                extended_count = sum(1 for trans in transactions if extended <= trans)
                if extended_count == count:
                    is_closed = False
                    break

            if is_closed:
                support = count / len(transactions)
                self.closed_itemsets.append(ClosedItemset(
                    items=new_prefix,
                    support=support,
                    count=count,
                ))

            # Recurse with remaining items
            remaining_items = items[i + 1:]
            if remaining_items:
                # Filter transactions that contain new_prefix
                filtered_trans = [t for t in transactions if new_prefix <= t]
                if len(filtered_trans) >= min_count:
                    self._mine_closed(
                        prefix=new_prefix,
                        items=remaining_items,
                        transactions=filtered_trans,
                        min_count=min_count,
                    )

    def get_user_patterns(
        self,
        user_transactions: Dict[str, Set],
    ) -> Dict[str, List[ClosedItemset]]:
        """
        Get closed itemset patterns for each user.

        Args:
            user_transactions: Dict mapping user_id to their venue set.

        Returns:
            Dict mapping user_id to their matching closed itemsets.
        """
        user_patterns = {}

        for user_id, venues in user_transactions.items():
            matching = []
            for itemset in self.closed_itemsets:
                if itemset.items <= venues:
                    matching.append(itemset)

            # Sort by support (descending)
            matching.sort(key=lambda x: x.support, reverse=True)
            user_patterns[user_id] = matching

        return user_patterns

    def itemset_to_features(
        self,
        user_id: str,
        user_transactions: Dict[str, Set],
        top_k: int = 10,
    ) -> np.ndarray:
        """
        Convert user's itemset patterns to feature vector.

        Args:
            user_id: User ID.
            user_transactions: Dict mapping user_id to venue set.
            top_k: Number of top itemsets to use as features.

        Returns:
            Feature vector based on itemset membership.
        """
        if not self.closed_itemsets:
            return np.zeros(top_k)

        # Sort itemsets by support for consistent feature order
        sorted_itemsets = sorted(
            self.closed_itemsets,
            key=lambda x: (-x.support, -len(x.items)),
        )[:top_k]

        user_venues = user_transactions.get(user_id, set())
        features = np.zeros(top_k)

        for i, itemset in enumerate(sorted_itemsets):
            if itemset.items <= user_venues:
                features[i] = itemset.support

        return features


class UserProfileMiner:
    """
    Mine user profiles from interaction history.

    Combines closed itemset patterns with other behavioral features.
    """

    def __init__(
        self,
        min_support: float = 0.01,
        max_itemset_size: int = 5,
    ):
        """
        Initialize the miner.

        Args:
            min_support: Minimum support for closed itemsets.
            max_itemset_size: Maximum itemset size.
        """
        self.dci = DCIClosed(
            min_support=min_support,
            max_itemset_size=max_itemset_size,
        )
        self.user_transactions: Dict[str, Set] = {}
        self.venue_categories: Dict[str, Set[str]] = {}

    def fit(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = "user_id",
        venue_col: str = "business_id",
        category_col: Optional[str] = "categories",
    ) -> "UserProfileMiner":
        """
        Fit the miner on interaction data.

        Args:
            interactions_df: DataFrame with user-venue interactions.
            user_col: User ID column.
            venue_col: Venue ID column.
            category_col: Optional venue category column.

        Returns:
            Self for chaining.
        """
        # Build user transactions
        self.user_transactions = (
            interactions_df.groupby(user_col)[venue_col]
            .apply(set)
            .to_dict()
        )

        # Extract venue categories if available
        if category_col and category_col in interactions_df.columns:
            for _, row in interactions_df.drop_duplicates(venue_col).iterrows():
                venue = row[venue_col]
                cats = row[category_col]
                if pd.notna(cats):
                    self.venue_categories[venue] = set(
                        c.strip() for c in str(cats).split(",")
                    )

        # Mine closed itemsets from all transactions
        all_transactions = list(self.user_transactions.values())
        self.dci.fit(all_transactions)

        return self

    def get_user_profile(
        self,
        user_id: str,
        itemset_features: int = 10,
    ) -> Dict:
        """
        Get comprehensive user profile.

        Args:
            user_id: User to profile.
            itemset_features: Number of itemset features.

        Returns:
            Dict with profile features.
        """
        venues = self.user_transactions.get(user_id, set())

        profile = {
            "user_id": user_id,
            "num_venues": len(venues),
            "venues": list(venues),
        }

        # Itemset pattern features
        itemset_feats = self.dci.itemset_to_features(
            user_id,
            self.user_transactions,
            top_k=itemset_features,
        )
        for i, feat in enumerate(itemset_feats):
            profile[f"itemset_{i}"] = feat

        # Category distribution
        if self.venue_categories:
            category_counts = defaultdict(int)
            for venue in venues:
                for cat in self.venue_categories.get(venue, []):
                    category_counts[cat] += 1

            profile["top_categories"] = sorted(
                category_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

        # Matching patterns
        patterns = self.dci.get_user_patterns(self.user_transactions)
        user_patterns = patterns.get(user_id, [])
        profile["num_patterns"] = len(user_patterns)
        profile["top_pattern_support"] = (
            user_patterns[0].support if user_patterns else 0.0
        )

        return profile

    def get_all_profiles_df(
        self,
        itemset_features: int = 10,
    ) -> pd.DataFrame:
        """
        Get profiles for all users as DataFrame.

        Args:
            itemset_features: Number of itemset features.

        Returns:
            DataFrame with user profiles.
        """
        profiles = []

        for user_id in self.user_transactions.keys():
            profile = self.get_user_profile(user_id, itemset_features)

            # Flatten for DataFrame
            flat_profile = {
                "user_id": profile["user_id"],
                "num_venues": profile["num_venues"],
                "num_patterns": profile["num_patterns"],
                "top_pattern_support": profile["top_pattern_support"],
            }

            # Add itemset features
            for i in range(itemset_features):
                flat_profile[f"itemset_{i}"] = profile.get(f"itemset_{i}", 0.0)

            profiles.append(flat_profile)

        return pd.DataFrame(profiles)


def mine_user_patterns(
    interactions_df: pd.DataFrame,
    min_support: float = 0.01,
    max_itemset_size: int = 5,
    user_col: str = "user_id",
    venue_col: str = "business_id",
) -> Tuple[UserProfileMiner, pd.DataFrame]:
    """
    Mine user patterns from interactions.

    Args:
        interactions_df: Interaction DataFrame.
        min_support: Minimum support threshold.
        max_itemset_size: Maximum itemset size.
        user_col: User ID column.
        venue_col: Venue ID column.

    Returns:
        Tuple of (miner, profiles DataFrame).
    """
    miner = UserProfileMiner(
        min_support=min_support,
        max_itemset_size=max_itemset_size,
    )

    miner.fit(interactions_df, user_col=user_col, venue_col=venue_col)
    profiles_df = miner.get_all_profiles_df()

    return miner, profiles_df
