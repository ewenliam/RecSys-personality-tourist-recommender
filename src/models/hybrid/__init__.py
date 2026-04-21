"""Hybrid recommendation engine module."""
from .dci_closed import (
    DCIClosed,
    ClosedItemset,
    UserProfileMiner,
    mine_user_patterns,
)
from .xgboost_ranker import (
    XGBoostRanker,
    FeatureSynthesizer,
    RankingFeatures,
    HybridRecommender,
    build_topic_confidence,
)
from .gbce_loss import (
    gBCELoss,
    FocalLoss,
    CalibrationMetrics,
    calibrate_predictions,
    compute_overconfidence_penalty,
)
from .personality_scorer import (
    PersonalityScorer,
    mbti_to_vector,
)

__all__ = [
    # DCI Closed
    "DCIClosed",
    "ClosedItemset",
    "UserProfileMiner",
    "mine_user_patterns",
    # XGBoost Ranker
    "XGBoostRanker",
    "FeatureSynthesizer",
    "RankingFeatures",
    "HybridRecommender",
    "build_topic_confidence",
    # Calibration
    "gBCELoss",
    "FocalLoss",
    "CalibrationMetrics",
    "calibrate_predictions",
    "compute_overconfidence_penalty",
    # Personality
    "PersonalityScorer",
    "mbti_to_vector",
]
