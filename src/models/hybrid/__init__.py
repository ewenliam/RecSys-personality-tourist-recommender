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
)
from .gbce_loss import (
    gBCELoss,
    FocalLoss,
    CalibrationMetrics,
    calibrate_predictions,
    compute_overconfidence_penalty,
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
    # Calibration
    "gBCELoss",
    "FocalLoss",
    "CalibrationMetrics",
    "calibrate_predictions",
    "compute_overconfidence_penalty",
]
