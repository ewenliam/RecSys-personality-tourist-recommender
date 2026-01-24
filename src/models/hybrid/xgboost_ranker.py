"""
XGBoost-based Ranking Model.

Combines GNN embeddings with context features and DCI patterns
for final venue ranking. Uses custom calibrated loss to reduce
overconfidence.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.config.settings import HybridConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class RankingFeatures:
    """Container for ranking features."""
    user_features: np.ndarray
    venue_features: np.ndarray
    context_features: Optional[np.ndarray] = None
    itemset_features: Optional[np.ndarray] = None


class FeatureSynthesizer:
    """
    Synthesize features for the ranking model.

    Combines:
    - User GNN embeddings
    - Venue GNN embeddings
    - User MBTI features
    - Venue topic features
    - Context features (time, weather, location)
    - DCI itemset pattern features
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize the synthesizer.

        Args:
            normalize: Whether to normalize features.
        """
        self.normalize = normalize
        self.user_scaler = StandardScaler()
        self.venue_scaler = StandardScaler()
        self.context_scaler = StandardScaler()
        self.fitted = False

    def fit(
        self,
        user_embeddings: np.ndarray,
        venue_embeddings: np.ndarray,
        context_features: Optional[np.ndarray] = None,
    ) -> "FeatureSynthesizer":
        """
        Fit scalers on training data.

        Args:
            user_embeddings: User embedding matrix.
            venue_embeddings: Venue embedding matrix.
            context_features: Optional context feature matrix.

        Returns:
            Self for chaining.
        """
        if self.normalize:
            self.user_scaler.fit(user_embeddings)
            self.venue_scaler.fit(venue_embeddings)
            if context_features is not None:
                self.context_scaler.fit(context_features)

        self.fitted = True
        return self

    def transform(
        self,
        user_embeddings: np.ndarray,
        venue_embeddings: np.ndarray,
        context_features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Transform features using fitted scalers.

        Args:
            user_embeddings: User embeddings to transform.
            venue_embeddings: Venue embeddings to transform.
            context_features: Optional context features.

        Returns:
            Tuple of transformed features.
        """
        if not self.fitted:
            raise ValueError("Synthesizer not fitted. Call fit() first.")

        if self.normalize:
            user_emb = self.user_scaler.transform(user_embeddings)
            venue_emb = self.venue_scaler.transform(venue_embeddings)
            context = (
                self.context_scaler.transform(context_features)
                if context_features is not None else None
            )
        else:
            user_emb = user_embeddings
            venue_emb = venue_embeddings
            context = context_features

        return user_emb, venue_emb, context

    def create_pair_features(
        self,
        user_idx: np.ndarray,
        venue_idx: np.ndarray,
        user_embeddings: np.ndarray,
        venue_embeddings: np.ndarray,
        user_extra: Optional[np.ndarray] = None,
        venue_extra: Optional[np.ndarray] = None,
        context_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create feature matrix for user-venue pairs.

        Args:
            user_idx: User indices.
            venue_idx: Venue indices.
            user_embeddings: User embedding matrix.
            venue_embeddings: Venue embedding matrix.
            user_extra: Extra user features (e.g., itemset patterns).
            venue_extra: Extra venue features (e.g., topic vectors).
            context_features: Context features per pair.

        Returns:
            Feature matrix [num_pairs, feature_dim].
        """
        # Get embeddings for pairs
        user_emb = user_embeddings[user_idx]
        venue_emb = venue_embeddings[venue_idx]

        # Combine features
        features = [user_emb, venue_emb]

        # Add element-wise product (interaction)
        features.append(user_emb * venue_emb)

        # Add cosine similarity
        cos_sim = np.sum(user_emb * venue_emb, axis=1, keepdims=True)
        norms = (
            np.linalg.norm(user_emb, axis=1, keepdims=True) *
            np.linalg.norm(venue_emb, axis=1, keepdims=True) + 1e-8
        )
        cos_sim = cos_sim / norms
        features.append(cos_sim)

        # Add extra features
        if user_extra is not None:
            features.append(user_extra[user_idx])

        if venue_extra is not None:
            features.append(venue_extra[venue_idx])

        if context_features is not None:
            features.append(context_features)

        return np.hstack(features)


class XGBoostRanker:
    """
    XGBoost-based ranking model for recommendations.
    """

    def __init__(
        self,
        config: Optional[HybridConfig] = None,
        use_calibration: bool = True,
        calibration_t: float = 0.8,
    ):
        """
        Initialize the ranker.

        Args:
            config: Hybrid configuration.
            use_calibration: Apply calibration to predictions.
            calibration_t: Calibration temperature.
        """
        self.config = config or get_config().hybrid
        self.use_calibration = use_calibration
        self.calibration_t = calibration_t or self.config.gbce_calibration_t

        self.model: Optional[xgb.Booster] = None
        self.feature_synthesizer = FeatureSynthesizer()

        # XGBoost parameters
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "max_depth": self.config.xgb_max_depth,
            "learning_rate": self.config.xgb_learning_rate,
            "subsample": self.config.xgb_subsample,
            "colsample_bytree": self.config.xgb_colsample_bytree,
            "tree_method": "hist",
            "device": "cuda",  # Use GPU if available
            "seed": 42,
        }

    def prepare_training_data(
        self,
        train_edges: np.ndarray,
        user_embeddings: np.ndarray,
        venue_embeddings: np.ndarray,
        num_negatives: int = 4,
        user_extra: Optional[np.ndarray] = None,
        venue_extra: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with negative sampling.

        Args:
            train_edges: Positive edges [2, num_edges].
            user_embeddings: User embeddings.
            venue_embeddings: Venue embeddings.
            num_negatives: Negatives per positive.
            user_extra: Extra user features.
            venue_extra: Extra venue features.

        Returns:
            Tuple of (features, labels).
        """
        num_users = user_embeddings.shape[0]
        num_venues = venue_embeddings.shape[0]

        # Build positive set for each user
        user_positives = {}
        for i in range(train_edges.shape[1]):
            user = train_edges[0, i]
            venue = train_edges[1, i]
            if user not in user_positives:
                user_positives[user] = set()
            user_positives[user].add(venue)

        # Create training pairs
        pos_users = train_edges[0]
        pos_venues = train_edges[1]
        num_pos = len(pos_users)

        # Sample negatives
        neg_users = []
        neg_venues = []

        for user in pos_users:
            positives = user_positives.get(user, set())
            for _ in range(num_negatives):
                neg = np.random.randint(0, num_venues)
                while neg in positives:
                    neg = np.random.randint(0, num_venues)
                neg_users.append(user)
                neg_venues.append(neg)

        neg_users = np.array(neg_users)
        neg_venues = np.array(neg_venues)

        # Combine
        all_users = np.concatenate([pos_users, neg_users])
        all_venues = np.concatenate([pos_venues, neg_venues])
        labels = np.concatenate([
            np.ones(num_pos),
            np.zeros(len(neg_users)),
        ])

        # Fit synthesizer
        self.feature_synthesizer.fit(user_embeddings, venue_embeddings)

        # Create features
        user_emb, venue_emb, _ = self.feature_synthesizer.transform(
            user_embeddings, venue_embeddings
        )

        features = self.feature_synthesizer.create_pair_features(
            all_users, all_venues,
            user_emb, venue_emb,
            user_extra, venue_extra,
        )

        # Shuffle
        perm = np.random.permutation(len(labels))
        features = features[perm]
        labels = labels[perm]

        logger.info(f"Prepared {len(labels)} training samples "
                   f"({num_pos} pos, {len(neg_users)} neg)")

        return features, labels

    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        num_rounds: int = 100,
        early_stopping_rounds: int = 10,
    ) -> Dict:
        """
        Train the XGBoost model.

        Args:
            train_features: Training features.
            train_labels: Training labels.
            val_features: Validation features.
            val_labels: Validation labels.
            num_rounds: Number of boosting rounds.
            early_stopping_rounds: Early stopping patience.

        Returns:
            Training history.
        """
        num_rounds = num_rounds or self.config.xgb_n_estimators

        # Create DMatrix
        dtrain = xgb.DMatrix(train_features, label=train_labels)

        evals = [(dtrain, "train")]
        if val_features is not None:
            dval = xgb.DMatrix(val_features, label=val_labels)
            evals.append((dval, "val"))

        # Train
        logger.info("Training XGBoost ranker...")

        evals_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10,
        )

        logger.info(f"Best iteration: {self.model.best_iteration}")

        return evals_result

    def predict(
        self,
        user_idx: np.ndarray,
        venue_idx: np.ndarray,
        user_embeddings: np.ndarray,
        venue_embeddings: np.ndarray,
        user_extra: Optional[np.ndarray] = None,
        venue_extra: Optional[np.ndarray] = None,
        context_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict scores for user-venue pairs.

        Args:
            user_idx: User indices.
            venue_idx: Venue indices.
            user_embeddings: User embeddings.
            venue_embeddings: Venue embeddings.
            user_extra: Extra user features.
            venue_extra: Extra venue features.
            context_features: Context features.

        Returns:
            Predicted scores.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Transform embeddings
        user_emb, venue_emb, context = self.feature_synthesizer.transform(
            user_embeddings, venue_embeddings, context_features
        )

        # Create features
        features = self.feature_synthesizer.create_pair_features(
            user_idx, venue_idx,
            user_emb, venue_emb,
            user_extra, venue_extra,
            context,
        )

        # Predict
        dtest = xgb.DMatrix(features)
        scores = self.model.predict(dtest)

        # Apply calibration
        if self.use_calibration:
            scores = self._calibrate(scores)

        return scores

    def _calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply temperature calibration to reduce overconfidence.

        Args:
            scores: Raw prediction scores.

        Returns:
            Calibrated scores.
        """
        # Temperature scaling
        return np.power(scores, 1.0 / self.calibration_t)

    def recommend(
        self,
        user_idx: int,
        user_embeddings: np.ndarray,
        venue_embeddings: np.ndarray,
        k: int = 10,
        exclude_venues: Optional[set] = None,
        user_extra: Optional[np.ndarray] = None,
        venue_extra: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-k recommendations for a user.

        Args:
            user_idx: User index.
            user_embeddings: User embeddings.
            venue_embeddings: Venue embeddings.
            k: Number of recommendations.
            exclude_venues: Venues to exclude.
            user_extra: Extra user features.
            venue_extra: Extra venue features.

        Returns:
            Tuple of (venue_indices, scores).
        """
        num_venues = venue_embeddings.shape[0]

        # Score all venues
        user_indices = np.full(num_venues, user_idx)
        venue_indices = np.arange(num_venues)

        scores = self.predict(
            user_indices, venue_indices,
            user_embeddings, venue_embeddings,
            user_extra, venue_extra,
        )

        # Exclude specified venues
        if exclude_venues:
            for vid in exclude_venues:
                scores[vid] = -np.inf

        # Get top-k
        top_k_idx = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_idx]

        return top_k_idx, top_k_scores

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.get_score(importance_type="gain")
        df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in importance.items()
        ])
        df = df.sort_values("importance", ascending=False)

        return df

    def save(self, path: Union[str, Path]) -> None:
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path))
        logger.info(f"Saved model to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load a saved model."""
        path = Path(path)
        self.model = xgb.Booster()
        self.model.load_model(str(path))
        logger.info(f"Loaded model from {path}")


class HybridRecommender:
    """
    Complete hybrid recommendation system.

    Combines:
    - GNN embeddings
    - MBTI personality features
    - Topic embeddings
    - DCI closed itemset patterns
    - XGBoost ranking
    """

    def __init__(
        self,
        config: Optional[HybridConfig] = None,
    ):
        """
        Initialize the hybrid recommender.

        Args:
            config: Hybrid configuration.
        """
        self.config = config or get_config().hybrid
        self.ranker = XGBoostRanker(config=self.config)

        # Data storage
        self.user_embeddings: Optional[np.ndarray] = None
        self.venue_embeddings: Optional[np.ndarray] = None
        self.user_extra: Optional[np.ndarray] = None
        self.venue_extra: Optional[np.ndarray] = None
        self.user_id_map: Dict[str, int] = {}
        self.venue_id_map: Dict[str, int] = {}
        self.user_positives: Dict[int, set] = {}

    def fit(
        self,
        train_edges: np.ndarray,
        user_embeddings: np.ndarray,
        venue_embeddings: np.ndarray,
        user_ids: Optional[List[str]] = None,
        venue_ids: Optional[List[str]] = None,
        user_extra: Optional[np.ndarray] = None,
        venue_extra: Optional[np.ndarray] = None,
        val_edges: Optional[np.ndarray] = None,
        num_negatives: int = 4,
        num_rounds: int = 100,
    ) -> Dict:
        """
        Fit the hybrid recommender.

        Args:
            train_edges: Training edges [2, num_edges].
            user_embeddings: User embedding matrix.
            venue_embeddings: Venue embedding matrix.
            user_ids: User ID list.
            venue_ids: Venue ID list.
            user_extra: Extra user features.
            venue_extra: Extra venue features.
            val_edges: Validation edges.
            num_negatives: Negatives per positive.
            num_rounds: XGBoost rounds.

        Returns:
            Training history.
        """
        # Store data
        self.user_embeddings = user_embeddings
        self.venue_embeddings = venue_embeddings
        self.user_extra = user_extra
        self.venue_extra = venue_extra

        if user_ids:
            self.user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        if venue_ids:
            self.venue_id_map = {vid: idx for idx, vid in enumerate(venue_ids)}

        # Build positive set
        for i in range(train_edges.shape[1]):
            user = train_edges[0, i]
            venue = train_edges[1, i]
            if user not in self.user_positives:
                self.user_positives[user] = set()
            self.user_positives[user].add(venue)

        # Prepare training data
        train_features, train_labels = self.ranker.prepare_training_data(
            train_edges,
            user_embeddings,
            venue_embeddings,
            num_negatives=num_negatives,
            user_extra=user_extra,
            venue_extra=venue_extra,
        )

        # Prepare validation data
        val_features, val_labels = None, None
        if val_edges is not None:
            val_features, val_labels = self.ranker.prepare_training_data(
                val_edges,
                user_embeddings,
                venue_embeddings,
                num_negatives=num_negatives,
                user_extra=user_extra,
                venue_extra=venue_extra,
            )

        # Train
        history = self.ranker.train(
            train_features, train_labels,
            val_features, val_labels,
            num_rounds=num_rounds,
        )

        return history

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_visited: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations for a user.

        Args:
            user_id: User ID.
            k: Number of recommendations.
            exclude_visited: Exclude visited venues.

        Returns:
            List of (venue_id, score) tuples.
        """
        if user_id not in self.user_id_map:
            logger.warning(f"Unknown user: {user_id}")
            return []

        user_idx = self.user_id_map[user_id]

        # Get venues to exclude
        exclude = self.user_positives.get(user_idx, set()) if exclude_visited else set()

        # Get recommendations
        venue_indices, scores = self.ranker.recommend(
            user_idx,
            self.user_embeddings,
            self.venue_embeddings,
            k=k,
            exclude_venues=exclude,
            user_extra=self.user_extra,
            venue_extra=self.venue_extra,
        )

        # Convert to IDs
        idx_to_venue = {idx: vid for vid, idx in self.venue_id_map.items()}
        recommendations = [
            (idx_to_venue.get(idx, str(idx)), float(score))
            for idx, score in zip(venue_indices, scores)
        ]

        return recommendations
