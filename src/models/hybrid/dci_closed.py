"""
DCI Closed Algorithm for Frequent Closed Itemset Mining.

Mines frequent closed itemsets from user visit history to create
compact behavioral patterns. A closed itemset is one where no
superset has the same support.

Optimized with inverted index for O(1) support counting instead
of scanning all transactions per candidate.

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

    Uses an inverted index (item -> transaction bitset) for fast
    support counting via set intersection instead of full scans.
    """

    def __init__(
        self,
        min_support: float = 0.05,
        max_itemset_size: int = 3,
        max_itemsets: int = 5000,
        config: Optional[HybridConfig] = None,
    ):
        """
        Initialize the miner.

        Args:
            min_support: Minimum support threshold (fraction of transactions).
            max_itemset_size: Maximum itemset size to mine.
            max_itemsets: Stop mining after this many itemsets found.
            config: Hybrid configuration.
        """
        self.config = config or get_config().hybrid
        self.min_support = min_support
        self.max_itemset_size = max_itemset_size
        self.max_itemsets = max_itemsets

        self.transactions: List[Set] = []
        self.closed_itemsets: List[ClosedItemset] = []
        self.item_counts: Dict = {}

        # Inverted index: item -> set of transaction indices
        self._item_tids: Dict = {}

    def fit(self, transactions: List[Set]) -> List[ClosedItemset]:
        """
        Mine closed itemsets from transactions.

        Args:
            transactions: List of transaction sets.

        Returns:
            List of closed itemsets.
        """
        self.transactions = transactions
        n_transactions = len(transactions)

        if n_transactions == 0:
            return []

        min_count = max(1, int(self.min_support * n_transactions))
        logger.info(f"Mining closed itemsets from {n_transactions} transactions")
        logger.info(f"Min support: {self.min_support} ({min_count} transactions)")

        # Build inverted index: item -> set of transaction IDs
        self._item_tids = defaultdict(set)
        for tid, trans in enumerate(transactions):
            for item in trans:
                self._item_tids[item].add(tid)

        # Filter to frequent items only
        frequent_items = {
            item for item, tids in self._item_tids.items()
            if len(tids) >= min_count
        }

        # Prune inverted index to frequent items
        self._item_tids = {
            item: tids for item, tids in self._item_tids.items()
            if item in frequent_items
        }

        logger.info(f"Frequent items: {len(frequent_items)}")

        if len(frequent_items) == 0:
            logger.warning("No frequent items found. Try lowering min_support.")
            return []

        if len(frequent_items) > 10000:
            logger.warning(
                f"Too many frequent items ({len(frequent_items)}). "
                f"Raising min_support is recommended."
            )

        # Sort by frequency (ascending) for better pruning
        sorted_items = sorted(
            frequent_items,
            key=lambda x: len(self._item_tids[x]),
        )

        # Mine closed itemsets using depth-first search
        self.closed_itemsets = []
        self._mine_closed(
            prefix=frozenset(),
            prefix_tids=set(range(n_transactions)),
            items=sorted_items,
            min_count=min_count,
        )

        logger.info(f"Found {len(self.closed_itemsets)} closed itemsets")
        return self.closed_itemsets

    def _mine_closed(
        self,
        prefix: frozenset,
        prefix_tids: Set[int],
        items: List,
        min_count: int,
    ) -> None:
        """
        Recursive mining of closed itemsets using TID-set intersection.

        Instead of scanning all transactions for each candidate,
        we intersect TID sets: O(min(|A|, |B|)) per candidate.
        """
        if len(prefix) >= self.max_itemset_size:
            return

        if len(self.closed_itemsets) >= self.max_itemsets:
            return

        n_transactions = len(self.transactions)

        for i, item in enumerate(items):
            if len(self.closed_itemsets) >= self.max_itemsets:
                return

            # TID-set intersection instead of full scan
            new_tids = prefix_tids & self._item_tids[item]
            count = len(new_tids)

            if count < min_count:
                continue

            new_prefix = prefix | {item}

            # Check if closed: no remaining item has the same support
            is_closed = True
            for other_item in items[i + 1:]:
                extended_tids = new_tids & self._item_tids[other_item]
                if len(extended_tids) == count:
                    is_closed = False
                    break

            if is_closed:
                support = count / n_transactions if n_transactions > 0 else 0.0
                self.closed_itemsets.append(ClosedItemset(
                    items=new_prefix,
                    support=support,
                    count=count,
                ))

            # Recurse with remaining items
            remaining = items[i + 1:]
            if remaining and count >= min_count:
                self._mine_closed(
                    prefix=new_prefix,
                    prefix_tids=new_tids,
                    items=remaining,
                    min_count=min_count,
                )

    def get_user_patterns(
        self,
        user_venues: Set,
    ) -> List[ClosedItemset]:
        """
        Get closed itemset patterns matching a single user's venues.

        Args:
            user_venues: Set of venues the user has visited.

        Returns:
            List of matching closed itemsets, sorted by support.
        """
        matching = [
            itemset for itemset in self.closed_itemsets
            if itemset.items <= user_venues
        ]
        matching.sort(key=lambda x: x.support, reverse=True)
        return matching

    def itemset_to_features(
        self,
        user_venues: Set,
        top_k: int = 10,
    ) -> np.ndarray:
        """
        Convert user's itemset patterns to feature vector.

        Args:
            user_venues: Set of venues the user visited.
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
        min_support: float = 0.05,
        max_itemset_size: int = 3,
        max_itemsets: int = 5000,
    ):
        self.dci = DCIClosed(
            min_support=min_support,
            max_itemset_size=max_itemset_size,
            max_itemsets=max_itemsets,
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
            venues,
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

        # Matching patterns (for this user only, not all users)
        user_patterns = self.dci.get_user_patterns(venues)
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
        # Pre-sort itemsets once for consistent feature ordering
        sorted_itemsets = sorted(
            self.dci.closed_itemsets,
            key=lambda x: (-x.support, -len(x.items)),
        )[:itemset_features]

        profiles = []
        for user_id, venues in self.user_transactions.items():
            flat = {
                "user_id": user_id,
                "num_venues": len(venues),
            }

            # Vectorized itemset features (no per-user full pattern search)
            for i, itemset in enumerate(sorted_itemsets):
                flat[f"itemset_{i}"] = (
                    itemset.support if itemset.items <= venues else 0.0
                )

            # Pad remaining features with zeros
            for i in range(len(sorted_itemsets), itemset_features):
                flat[f"itemset_{i}"] = 0.0

            # Pattern count
            n_patterns = sum(
                1 for itemset in self.dci.closed_itemsets
                if itemset.items <= venues
            )
            flat["num_patterns"] = n_patterns
            flat["top_pattern_support"] = (
                sorted_itemsets[0].support
                if sorted_itemsets and sorted_itemsets[0].items <= venues
                else 0.0
            )

            profiles.append(flat)

        return pd.DataFrame(profiles)


def mine_user_patterns(
    interactions_df: pd.DataFrame,
    min_support: float = 0.05,
    max_itemset_size: int = 3,
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
