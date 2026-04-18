"""
Personality-Aware Scoring for Hybrid Recommendations.

Computes compatibility scores between user MBTI personality profiles
and venue characteristics. Integrates with the hybrid recommendation
engine to add personality-based features to XGBoost ranking.

Key insight: different MBTI dimensions correlate with distinct venue
preferences (e.g., Extraverts prefer social venues, Intuitives prefer
novel/unique experiences). This module quantifies those affinities.
"""
import logging
from typing import Optional

import numpy as np
from sklearn.preprocessing import normalize

from src.config.settings import get_config

logger = logging.getLogger(__name__)

# MBTI dimension indices in 4-dimensional binary label space
MBTI_DIMS = ["EI", "SN", "TF", "JP"]

# Full 16-type labels ordered by binary encoding (E=0/I=1, S=0/N=1, T=0/F=1, J=0/P=1)
MBTI_TYPES = [
    "ESTJ", "ESTP", "ESFJ", "ESFP",
    "ENTJ", "ENTP", "ENFJ", "ENFP",
    "ISTJ", "ISTP", "ISFJ", "ISFP",
    "INTJ", "INTP", "INFJ", "INFP",
]


def mbti_to_vector(mbti_type: str) -> np.ndarray:
    """
    Convert MBTI type string to a 4-dimensional binary vector.

    Convention: 0 = first letter (E, S, T, J), 1 = second letter (I, N, F, P)
    """
    if len(mbti_type) != 4:
        raise ValueError(f"Invalid MBTI type: {mbti_type}")

    vec = np.zeros(4, dtype=np.float32)
    vec[0] = 0.0 if mbti_type[0] == "E" else 1.0  # E/I
    vec[1] = 0.0 if mbti_type[1] == "S" else 1.0  # S/N
    vec[2] = 0.0 if mbti_type[2] == "T" else 1.0  # T/F
    vec[3] = 0.0 if mbti_type[3] == "J" else 1.0  # J/P
    return vec


class PersonalityScorer:
    """
    Computes personality-venue compatibility features for hybrid ranking.

    Features generated per user-venue pair:
      - MBTI dimension probabilities (4 values from classifier)
      - Personality-topic affinity (cosine similarity in embedding space)
      - MBTI type distribution entropy (confidence measure)
      - Cold-start indicator (binary: is user new?)
    """

    def __init__(
        self,
        user_mbti_probs: Optional[np.ndarray] = None,
        user_mbti_embeddings: Optional[np.ndarray] = None,
        venue_embeddings: Optional[np.ndarray] = None,
    ):
        """
        Initialize the personality scorer.

        Args:
            user_mbti_probs: Per-dimension MBTI probabilities [n_users, 4].
                Each value is P(second letter) - e.g., P(I), P(N), P(F), P(P).
            user_mbti_embeddings: BERT-MBTI CLS embeddings [n_users, 768].
            venue_embeddings: Venue embeddings [n_venues, dim].
        """
        self.user_mbti_probs = user_mbti_probs
        self.user_mbti_embeddings = user_mbti_embeddings
        self.venue_embeddings = venue_embeddings

        # Pre-compute normalized embeddings for cosine similarity.
        # Only enable if both embeddings exist AND share the same dim.
        self._user_emb_norm = None
        self._venue_emb_norm = None

        if user_mbti_embeddings is not None and venue_embeddings is not None:
            if user_mbti_embeddings.shape[1] == venue_embeddings.shape[1]:
                self._user_emb_norm = normalize(user_mbti_embeddings, axis=1)
                self._venue_emb_norm = normalize(venue_embeddings, axis=1)
            else:
                logger.warning(
                    f"PersonalityScorer: user dim={user_mbti_embeddings.shape[1]} "
                    f"!= venue dim={venue_embeddings.shape[1]}, "
                    f"skipping cosine similarity feature (will be zero)."
                )

    def compute_personality_features(
        self,
        user_indices: np.ndarray,
        venue_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute personality-aware features for user-venue pairs.

        Returns a feature matrix with columns:
          [0:4]  - MBTI dimension probabilities (P(I), P(N), P(F), P(P))
          [4]    - Personality-venue cosine similarity
          [5]    - MBTI confidence (1 - entropy of dim probs)
          [6]    - Cold-start indicator

        Args:
            user_indices: User index array [batch_size].
            venue_indices: Venue index array [batch_size].

        Returns:
            Feature matrix [batch_size, 7].
        """
        batch_size = len(user_indices)
        features = []

        # MBTI dimension probabilities
        if self.user_mbti_probs is not None:
            mbti_probs = self.user_mbti_probs[user_indices]
            features.append(mbti_probs)

            # MBTI confidence: 1 - normalized entropy
            # Higher confidence = probabilities far from 0.5
            entropy = -mbti_probs * np.log(mbti_probs + 1e-8) - (1 - mbti_probs) * np.log(1 - mbti_probs + 1e-8)
            avg_entropy = entropy.mean(axis=1, keepdims=True)
            max_entropy = np.log(2)
            confidence = 1.0 - (avg_entropy / max_entropy)
            features.append(confidence)
        else:
            features.append(np.full((batch_size, 4), 0.5, dtype=np.float32))
            features.append(np.zeros((batch_size, 1), dtype=np.float32))

        # Personality-venue cosine similarity
        if self._user_emb_norm is not None and self._venue_emb_norm is not None:
            user_emb = self._user_emb_norm[user_indices]
            venue_emb = self._venue_emb_norm[venue_indices]
            cos_sim = np.sum(user_emb * venue_emb, axis=1, keepdims=True)
            features.append(cos_sim)
        else:
            features.append(np.zeros((batch_size, 1), dtype=np.float32))

        # Cold-start indicator (user has no MBTI predictions)
        if self.user_mbti_probs is not None:
            cold_start = np.all(
                np.abs(self.user_mbti_probs[user_indices] - 0.5) < 0.01,
                axis=1, keepdims=True,
            ).astype(np.float32)
        else:
            cold_start = np.ones((batch_size, 1), dtype=np.float32)
        features.append(cold_start)

        return np.hstack(features)

    def get_feature_names(self) -> list[str]:
        """Get names for the personality features."""
        return [
            "mbti_prob_I", "mbti_prob_N", "mbti_prob_F", "mbti_prob_P",
            "mbti_confidence",
            "personality_venue_similarity",
            "cold_start_indicator",
        ]

    @staticmethod
    def compute_type_affinity_matrix(
        user_types: list[str],
        venue_topic_dists: np.ndarray,
    ) -> np.ndarray:
        """
        Compute affinity matrix between MBTI types and venue topics.

        Groups users by MBTI type, aggregates their preferred venue topics,
        and produces a [16, n_topics] affinity matrix.

        Args:
            user_types: List of MBTI type strings per user.
            venue_topic_dists: Topic distribution matrix [n_venues, n_topics].

        Returns:
            Affinity matrix [16, n_topics].
        """
        n_topics = venue_topic_dists.shape[1]
        affinity = np.zeros((16, n_topics), dtype=np.float32)
        counts = np.zeros(16, dtype=np.int32)

        type_to_idx = {t: i for i, t in enumerate(MBTI_TYPES)}

        for i, mbti_type in enumerate(user_types):
            mbti_type = mbti_type.upper()
            if mbti_type not in type_to_idx:
                continue
            idx = type_to_idx[mbti_type]
            if i < venue_topic_dists.shape[0]:
                affinity[idx] += venue_topic_dists[i]
                counts[idx] += 1

        # Normalize by count
        for i in range(16):
            if counts[i] > 0:
                affinity[i] /= counts[i]

        return affinity
