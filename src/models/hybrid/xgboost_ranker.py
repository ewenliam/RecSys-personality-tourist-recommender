"""
XGBoost-based Ranking Model.

Combines GNN embeddings with context features and DCI patterns
for final venue ranking. Uses post-hoc calibration (Platt scaling
or isotonic regression) to produce well-calibrated probabilities.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

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
        personality_features: Optional[np.ndarray] = None,
        topic_confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Late-fusion ensemble feature matrix.

        Instead of feeding raw embeddings to XGBoost (early fusion),
        we pre-compute each sub-model's score for the (user, venue)
        pair and let XGBoost learn how to *weigh* them. This avoids
        the sparsity wall that killed early-fusion approaches.

        Feature layout:
          [0] knn_score      - cosine(MBTI_user, BERTopic_venue)
          [1] gnn_score      - cosine(GNN_user, GNN_venue)
          [2] venue_degree   - log(1 + #reviews for this venue)
          [3] user_degree    - log(1 + #reviews for this user)

        Only 4 features. Each is a dense, meaningful scalar that
        XGBoost can split on with max_depth=4 trees.

        Args:
            user_idx: User indices for each pair.
            venue_idx: Venue indices for each pair.
            user_embeddings: GNN user embeddings [n_users, d].
            venue_embeddings: GNN venue embeddings [n_venues, d].
            user_extra: MBTI personality features [n_users, 7].
            venue_extra: BERTopic PCA venue embeddings [n_venues, 64].
            context_features: Pre-computed [N, d] array of additional
                per-pair features (e.g., degree stats). If provided,
                these are appended directly.
            personality_features: Unused (legacy).
            topic_confidence: Unused (legacy).

        Returns:
            Feature matrix [num_pairs, num_features].
        """
        n = len(user_idx)
        features = []

        # --- Score 1: KNN score (content-based) ---
        # Cosine similarity between user's MBTI profile and venue's
        # BERTopic topic embedding. This replicates the KNN baseline
        # that achieved Hit Rate@10 = 0.200.
        if user_extra is not None and venue_extra is not None:
            pers = np.asarray(user_extra[user_idx], dtype=np.float32)
            topic = np.asarray(venue_extra[venue_idx], dtype=np.float32)
            # Project to common dimensionality for cosine sim
            d = min(pers.shape[1], topic.shape[1])
            p = pers[:, :d]
            t = topic[:, :d]
            dot = np.sum(p * t, axis=1, keepdims=True)
            p_norm = np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
            t_norm = np.linalg.norm(t, axis=1, keepdims=True) + 1e-8
            knn_score = (dot / (p_norm * t_norm)).astype(np.float32)
        else:
            knn_score = np.zeros((n, 1), dtype=np.float32)
        features.append(knn_score)

        # --- Score 2: GNN score (collaborative filtering) ---
        # Cosine similarity between GNN user and venue embeddings.
        # Captures structural neighbourhood affinity from the
        # interaction graph.
        user_emb = np.asarray(user_embeddings[user_idx], dtype=np.float32)
        venue_emb = np.asarray(venue_embeddings[venue_idx], dtype=np.float32)
        dot_gnn = np.sum(user_emb * venue_emb, axis=1, keepdims=True)
        u_norm = np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-8
        v_norm = np.linalg.norm(venue_emb, axis=1, keepdims=True) + 1e-8
        gnn_score = (dot_gnn / (u_norm * v_norm)).astype(np.float32)
        features.append(gnn_score)

        # --- Context features (degree stats, passed from caller) ---
        if context_features is not None:
            features.append(np.asarray(context_features, dtype=np.float32))

        return np.hstack(features)


def build_topic_confidence(
    venue_topics_df: pd.DataFrame,
    n_venues: int,
    venue_id_map: Optional[Dict[str, int]] = None,
    outlier_confidence: float = 0.1,
) -> np.ndarray:
    """
    Build a per-venue topic confidence vector from BERTopic output.

    Venues that were originally outliers (topic -1 before reduction)
    get a low confidence score. This helps XGBoost learn when to trust
    topic features vs. GNN embeddings.

    Args:
        venue_topics_df: DataFrame from VenueTopicExtractor with columns
            "venue_id", "topic_prob", and optionally "was_outlier".
        n_venues: Total number of venues in the embedding matrix.
        venue_id_map: Maps venue_id string -> integer index.
        outlier_confidence: Confidence assigned to ex-outlier venues.

    Returns:
        Array of shape [n_venues, 1] with confidence values.
    """
    confidence = np.ones((n_venues, 1), dtype=np.float32)

    for _, row in venue_topics_df.iterrows():
        vid = row["venue_id"]
        idx = venue_id_map.get(vid, None) if venue_id_map else None
        if idx is None:
            continue
        if idx >= n_venues:
            continue

        if row.get("was_outlier", False):
            confidence[idx, 0] = outlier_confidence
        else:
            # Use the topic probability as confidence
            prob = float(row.get("topic_prob", 1.0))
            confidence[idx, 0] = max(prob, outlier_confidence)

    return confidence


class XGBoostRanker:
    """
    XGBoost-based ranking model for recommendations.
    """

    def __init__(
        self,
        config: Optional[HybridConfig] = None,
        use_calibration: bool = True,
        calibration_method: str = "platt",
        calibration_t: float = 0.8,
    ):
        """
        Initialize the ranker.

        Args:
            config: Hybrid configuration.
            use_calibration: Apply calibration to predictions.
            calibration_method: One of "platt", "isotonic", "temperature".
                "platt" fits a logistic regression on raw XGBoost scores.
                "isotonic" fits an isotonic regression (non-parametric).
                "temperature" applies simple power-law scaling (legacy).
            calibration_t: Temperature for "temperature" method only.
        """
        self.config = config or get_config().hybrid
        self.use_calibration = use_calibration
        self.calibration_method = calibration_method
        self.calibration_t = calibration_t or self.config.gbce_calibration_t

        self.model: Optional[xgb.Booster] = None
        self.calibrator = None  # Fitted post-hoc calibrator
        self.feature_synthesizer = FeatureSynthesizer()

        # Degree arrays (set during prepare_training_data, used
        # during predict/recommend for late-fusion features)
        self.user_degree: Optional[np.ndarray] = None   # [n_users]
        self.venue_degree: Optional[np.ndarray] = None   # [n_venues]

        # Pointwise classification with class-weight balancing.
        # scale_pos_weight is computed dynamically in train() as
        # (n_neg / n_pos) to counteract the 4:1 negative sampling
        # ratio and prevent the model from defaulting to
        # "predict 0 for everything".
        #
        # With only 4 features (late fusion), we use shallow trees
        # (max_depth 3-4) and lower regularisation since each feature
        # is a strong, dense signal rather than a noisy embedding dim.
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "max_depth": self.config.xgb_max_depth,
            "learning_rate": self.config.xgb_learning_rate,
            "subsample": self.config.xgb_subsample,
            "colsample_bytree": 1.0,      # only 4 features, use all
            "min_child_weight": 5,         # less aggressive with 4 feats
            "gamma": 0.1,                  # lighter regularisation
            "tree_method": "hist",
            "device": "cuda",
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
        max_pos_samples: int = 500_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with negative sampling.

        Args:
            train_edges: Positive edges [2, num_edges].
            user_embeddings: User embeddings.
            venue_embeddings: Venue embeddings.
            num_negatives: Negatives per positive.
            user_extra: Extra user features (personality [n_users, 7]).
            venue_extra: Extra venue features (topic_pca [n_venues, 64]).
            max_pos_samples: Cap on positive edges to keep memory
                reasonable (500K * 5 = 2.5M pairs).

        Returns:
            Tuple of (features, labels).
        """
        num_venues = venue_embeddings.shape[0]
        num_edges = train_edges.shape[1]

        # Subsample if too many edges (XGBoost doesn't need millions)
        if num_edges > max_pos_samples:
            logger.info(
                f"Subsampling {num_edges} -> {max_pos_samples} positive edges"
            )
            perm = np.random.permutation(num_edges)[:max_pos_samples]
            train_edges = train_edges[:, perm]

        pos_users = train_edges[0]
        pos_venues = train_edges[1]
        num_pos = len(pos_users)

        # Vectorized negative sampling (no Python loop over 2.8M edges)
        total_neg = num_pos * num_negatives
        neg_users = np.repeat(pos_users, num_negatives)
        neg_venues = np.random.randint(0, num_venues, size=total_neg)

        # Re-sample collisions (negatives that hit a positive)
        pos_set = set(zip(pos_users.tolist(), pos_venues.tolist()))
        collision_idx = np.array([
            i for i in range(total_neg)
            if (neg_users[i], neg_venues[i]) in pos_set
        ])
        # Re-roll just the collisions (usually < 1% of total)
        for _ in range(10):
            if len(collision_idx) == 0:
                break
            neg_venues[collision_idx] = np.random.randint(
                0, num_venues, size=len(collision_idx)
            )
            still_bad = np.array([
                (neg_users[i], neg_venues[i]) in pos_set
                for i in collision_idx
            ])
            collision_idx = collision_idx[still_bad]

        # Combine
        all_users = np.concatenate([pos_users, neg_users])
        all_venues = np.concatenate([pos_venues, neg_venues])
        labels = np.concatenate([
            np.ones(num_pos),
            np.zeros(total_neg),
        ])

        # Compute and store degree arrays from the ORIGINAL training
        # edges (before subsampling). These are reused at inference.
        num_users = user_embeddings.shape[0]
        self.user_degree = np.zeros(num_users, dtype=np.float32)
        self.venue_degree = np.zeros(num_venues, dtype=np.float32)
        np.add.at(self.user_degree, pos_users, 1)
        np.add.at(self.venue_degree, pos_venues, 1)
        # Log-transform to compress heavy-tail distribution
        self.user_degree = np.log1p(self.user_degree)
        self.venue_degree = np.log1p(self.venue_degree)
        logger.info(
            f"Degree stats: user mean={self.user_degree.mean():.2f}, "
            f"venue mean={self.venue_degree.mean():.2f}"
        )

        # Build per-pair context: [venue_degree, user_degree]
        ctx = np.column_stack([
            self.venue_degree[all_venues],
            self.user_degree[all_users],
        ]).astype(np.float32)

        # Fit synthesizer
        self.feature_synthesizer.fit(user_embeddings, venue_embeddings)

        # Create features (late-fusion: knn_score + gnn_score + degrees)
        user_emb, venue_emb, _ = self.feature_synthesizer.transform(
            user_embeddings, venue_embeddings
        )

        features = self.feature_synthesizer.create_pair_features(
            all_users, all_venues,
            user_emb, venue_emb,
            user_extra, venue_extra,
            context_features=ctx,
        )

        # Shuffle (important for stochastic gradient descent)
        perm = np.random.permutation(len(labels))
        features = features[perm]
        labels = labels[perm]

        logger.info(
            f"Prepared {len(labels)} training samples "
            f"({num_pos} pos, {total_neg} neg, "
            f"neg:pos ratio = {num_negatives}:1, "
            f"features = {features.shape[1]})"
        )

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
        Train the XGBoost classifier with Platt calibration.

        Automatically computes ``scale_pos_weight`` from the label
        distribution (n_neg / n_pos) to counteract the negative
        sampling ratio and prevent the model from ignoring the
        minority positive class.

        Args:
            train_features: Training features [n_samples, n_features].
            train_labels: Binary labels (1=positive, 0=negative).
            val_features: Validation features (optional).
            val_labels: Validation labels (optional).
            num_rounds: Maximum boosting rounds.
            early_stopping_rounds: Early stopping patience.

        Returns:
            Training history dict.
        """
        num_rounds = num_rounds or self.config.xgb_n_estimators

        # Compute scale_pos_weight from actual label distribution
        n_pos = float(train_labels.sum())
        n_neg = float(len(train_labels) - n_pos)
        spw = n_neg / max(n_pos, 1.0)
        self.params["scale_pos_weight"] = spw
        logger.info(
            f"Class balance: {int(n_pos)} pos, {int(n_neg)} neg "
            f"-> scale_pos_weight={spw:.2f}"
        )

        # Carve out calibration set BEFORE training (never calibrate
        # on data the model has seen). 15% hold-out for Platt scaling.
        cal_features, cal_labels = val_features, val_labels
        if cal_features is None and self.use_calibration and self.calibration_method != "temperature":
            n = len(train_labels)
            perm = np.random.permutation(n)
            n_cal = max(int(n * 0.15), 1000)
            cal_idx = perm[:n_cal]
            train_idx = perm[n_cal:]

            cal_features = train_features[cal_idx]
            cal_labels = train_labels[cal_idx]
            train_features = train_features[train_idx]
            train_labels = train_labels[train_idx]
            logger.info(
                f"Carved {n_cal} calibration samples from training data"
            )

        # Create DMatrix
        dtrain = xgb.DMatrix(train_features, label=train_labels)

        evals = [(dtrain, "train")]
        if val_features is not None:
            dval = xgb.DMatrix(val_features, label=val_labels)
            evals.append((dval, "val"))

        # Train
        logger.info(
            f"Training XGBoost "
            f"(objective={self.params['objective']}, "
            f"max_depth={self.params['max_depth']}, "
            f"min_child_weight={self.params.get('min_child_weight', 1)}, "
            f"scale_pos_weight={spw:.2f})..."
        )

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

        # Fit Platt scaler on held-out calibration data.
        # binary:logistic outputs probabilities, so we convert to
        # logits first, then fit logistic regression for recalibration.
        if self.use_calibration and self.calibration_method != "temperature":
            self._fit_calibrator(cal_features, cal_labels)

        return evals_result

    def _fit_calibrator(
        self,
        cal_features: np.ndarray,
        cal_labels: np.ndarray,
    ) -> None:
        """
        Fit a post-hoc calibrator on held-out data.

        Platt scaling fits a logistic regression on the raw XGBoost
        output (logit of predicted probability). Isotonic regression
        fits a non-parametric monotonic function.

        Args:
            cal_features: Calibration feature matrix.
            cal_labels: Calibration binary labels.
        """
        if self.model is None:
            raise ValueError("XGBoost model must be trained first.")

        dcal = xgb.DMatrix(cal_features)
        raw_scores = self.model.predict(dcal)

        if self.calibration_method == "platt":
            # Platt scaling: logistic regression on logits
            logits = np.log(raw_scores / (1.0 - raw_scores + 1e-7))
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(logits.reshape(-1, 1), cal_labels)
            # Report calibration fit
            cal_probs = self.calibrator.predict_proba(
                logits.reshape(-1, 1)
            )[:, 1]
            logger.info(
                f"Platt calibrator fitted: "
                f"raw ECE={self._quick_ece(raw_scores, cal_labels):.4f}, "
                f"calibrated ECE={self._quick_ece(cal_probs, cal_labels):.4f}"
            )

        elif self.calibration_method == "isotonic":
            self.calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self.calibrator.fit(raw_scores, cal_labels)
            cal_probs = self.calibrator.predict(raw_scores)
            logger.info(
                f"Isotonic calibrator fitted: "
                f"raw ECE={self._quick_ece(raw_scores, cal_labels):.4f}, "
                f"calibrated ECE={self._quick_ece(cal_probs, cal_labels):.4f}"
            )
        else:
            logger.warning(
                f"Unknown calibration method '{self.calibration_method}', "
                f"falling back to temperature scaling"
            )

    @staticmethod
    def _quick_ece(
        preds: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> float:
        """Fast ECE computation for logging during calibrator fitting."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(preds)
        for i in range(n_bins):
            mask = (preds >= bin_boundaries[i]) & (preds < bin_boundaries[i + 1])
            count = mask.sum()
            if count > 0:
                ece += (count / n) * abs(
                    labels[mask].mean() - preds[mask].mean()
                )
        return ece

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
        user_emb, venue_emb, _ = self.feature_synthesizer.transform(
            user_embeddings, venue_embeddings
        )

        # Build per-pair context (degree stats) from stored arrays
        ctx = None
        if self.venue_degree is not None and self.user_degree is not None:
            ctx = np.column_stack([
                self.venue_degree[venue_idx],
                self.user_degree[user_idx],
            ]).astype(np.float32)

        # Create features (late-fusion scores + degrees)
        features = self.feature_synthesizer.create_pair_features(
            user_idx, venue_idx,
            user_emb, venue_emb,
            user_extra, venue_extra,
            context_features=ctx,
        )

        # Predict (binary:logistic outputs probabilities in [0, 1])
        dtest = xgb.DMatrix(features)
        scores = self.model.predict(dtest)

        # Apply Platt calibration to improve probability estimates
        if self.use_calibration:
            scores = self._calibrate(scores)

        return scores

    def _calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply post-hoc calibration to raw XGBoost scores.

        Uses the fitted calibrator (Platt or isotonic) if available,
        otherwise falls back to temperature scaling.

        Args:
            scores: Raw prediction scores from XGBoost.

        Returns:
            Calibrated probability scores.
        """
        if self.calibrator is not None:
            if self.calibration_method == "platt":
                logits = np.log(scores / (1.0 - scores + 1e-7))
                return self.calibrator.predict_proba(
                    logits.reshape(-1, 1)
                )[:, 1]
            elif self.calibration_method == "isotonic":
                return self.calibrator.predict(scores)

        # Fallback: temperature scaling (legacy)
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

        # Build positive set (vectorized with pandas groupby)
        import pandas as pd
        edge_df = pd.DataFrame({
            "user": train_edges[0], "venue": train_edges[1]
        })
        self.user_positives = (
            edge_df.groupby("user")["venue"].apply(set).to_dict()
        )

        # Prepare training data (capped at 500K positives to fit in RAM)
        train_features, train_labels = self.ranker.prepare_training_data(
            train_edges,
            user_embeddings,
            venue_embeddings,
            num_negatives=num_negatives,
            user_extra=user_extra,
            venue_extra=venue_extra,
            max_pos_samples=500_000,
        )

        # Prepare validation data (smaller cap for eval)
        val_features, val_labels = None, None
        if val_edges is not None:
            val_features, val_labels = self.ranker.prepare_training_data(
                val_edges,
                user_embeddings,
                venue_embeddings,
                num_negatives=num_negatives,
                user_extra=user_extra,
                venue_extra=venue_extra,
                max_pos_samples=100_000,
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
