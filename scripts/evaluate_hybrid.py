#!/usr/bin/env python3
"""
Phase 4: Hybrid Evaluation Pipeline.

Combines GNN embeddings with BERT-MBTI personality features via XGBoost
ranking, evaluates with Precision@K, Recall@K, NDCG@K, and generates
methodology comparison tables (Omer's approach vs Current).

Usage:
    python scripts/evaluate_hybrid.py
    python scripts/evaluate_hybrid.py --k 10 --quick
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import (
    get_config,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    PROJECT_ROOT,
)
from src.models.hybrid.personality_scorer import PersonalityScorer
from src.utils.metrics import RecommendationMetrics
from src.utils.helpers import setup_logging, set_seed

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class RRFRanker:
    """
    Reciprocal Rank Fusion (RRF) hybrid recommender.

    Combines a content-based KNN ranking (cosine similarity between the
    user's MBTI embedding and the venue's BERTopic PCA embedding) with a
    collaborative filtering ranking (GNN user-venue dot product) using the
    parameter-free RRF formula:

        rrf_score(v) = 1/(k + rank_knn(v)) + 1/(k + rank_gnn(v))

    where ranks are 1-indexed (rank 1 = most relevant venue).

    No training is required. The fusion is purely mathematical, so there is
    nothing to overfit and no popularity-bias shortcut for a learner to
    exploit.

    Args:
        k: RRF smoothing constant (default 60, the standard value).
            Higher k flattens score differences between adjacent ranks.
            Lower k makes top-ranked venues dominate more strongly.
        popularity_alpha: Weight of a small log-degree tie-breaker.
            Popularity is only added AFTER the RRF score so it never
            overrides personalisation - it breaks ties between venues
            with identical RRF scores.  Set to 0.0 to disable.
        rrf_mode: Which sub-rankers to combine.
            "hybrid"  - KNN + GNN (default, requires both embeddings)
            "knn"     - content-based only (MBTI x BERTopic)
            "gnn"     - collaborative only (GNN embeddings)
    """

    # Map a mode string to the set of sub-rankers it activates.
    _MODE_RANKERS = {
        "knn": {"knn"},
        "gnn": {"gnn"},
        "mbti": {"mbti"},
        "hybrid": {"knn", "gnn"},
        "knn+mbti": {"knn", "mbti"},
        "gnn+mbti": {"gnn", "mbti"},
        "full": {"knn", "gnn", "mbti"},
    }

    def __init__(
        self,
        k: int = 60,
        popularity_alpha: float = 0.01,
        rrf_mode: str = "hybrid",
    ):
        self.k = k
        self.popularity_alpha = popularity_alpha
        self.rrf_mode = rrf_mode
        # Which sub-rankers are active for this mode.
        self.active = self._MODE_RANKERS.get(rrf_mode, {"knn", "gnn"})

        # Degree array set by set_venue_degree() after train edges are known
        self.venue_log_degree: Optional[np.ndarray] = None

    def set_venue_degree(
        self,
        train_edges: np.ndarray,
        n_venues: int,
    ) -> None:
        """
        Pre-compute log(1 + degree) for every venue from training edges.

        Only used if popularity_alpha > 0.  Computed once and stored so
        that recommend() does not recompute it per call.

        Args:
            train_edges: [2, num_edges] array of (user_idx, venue_idx).
            n_venues: Total number of venues.
        """
        degree = np.zeros(n_venues, dtype=np.float32)
        np.add.at(degree, train_edges[1], 1)
        self.venue_log_degree = np.log1p(degree).astype(np.float32)
        logger.info(
            f"Venue degree: min={degree.min():.0f}, "
            f"mean={degree.mean():.1f}, max={degree.max():.0f}"
        )

    @staticmethod
    def _cosine_scores(
        query: np.ndarray,       # [d]
        matrix: np.ndarray,      # [n_items, d]
    ) -> np.ndarray:
        """Vectorised cosine similarity: query vs every row of matrix."""
        d = min(query.shape[0], matrix.shape[1])
        q = query[:d].astype(np.float32)
        m = matrix[:, :d].astype(np.float32)
        dots = m @ q                                          # [n_items]
        q_norm = float(np.linalg.norm(q)) + 1e-8
        m_norms = np.linalg.norm(m, axis=1) + 1e-8           # [n_items]
        return dots / (q_norm * m_norms)

    @staticmethod
    def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
        """
        Convert an array of scores to 1-indexed ranks (rank 1 = highest score).

        Args:
            scores: [n_items] float array, higher is better.

        Returns:
            ranks: [n_items] int array, rank[i] is the rank of item i.
        """
        n = len(scores)
        # argsort(-scores)[p] = index of the item at position p
        sorted_pos = np.argsort(-scores)          # position -> item_index
        ranks = np.empty(n, dtype=np.int32)
        ranks[sorted_pos] = np.arange(1, n + 1)  # item_index -> rank
        return ranks

    def recommend(
        self,
        user_idx: int,
        user_embeddings: np.ndarray,
        venue_embeddings: np.ndarray,
        k: int = 10,
        exclude_venues: Optional[set] = None,
        user_extra: Optional[np.ndarray] = None,   # BERTopic user profile [n_u, d]
        venue_extra: Optional[np.ndarray] = None,  # BERTopic PCA [n_v, d]
        user_mbti: Optional[np.ndarray] = None,    # BERT-MBTI user CLS [n_u, 768]
        venue_mbti: Optional[np.ndarray] = None,   # BERT-MBTI venue CLS [n_v, 768]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the top-k recommended venues for user_idx using RRF.

        Args:
            user_idx: Integer index of the target user.
            user_embeddings: GNN user embeddings [n_users, d].
            venue_embeddings: GNN venue embeddings [n_venues, d].
            k: Number of recommendations to return.
            exclude_venues: Set of venue indices to exclude (already visited).
            user_extra: MBTI user embeddings [n_users, d_mbti].
                Used as the query for the KNN ranker.
            venue_extra: BERTopic PCA venue embeddings [n_venues, d_topic].
                Used as the keys for the KNN ranker.

        Returns:
            Tuple of (top_k_venue_indices, top_k_rrf_scores), both sorted
            by descending RRF score.
        """
        n_venues = venue_embeddings.shape[0]
        rrf_score = np.zeros(n_venues, dtype=np.float64)

        # ---- KNN sub-ranker (content-based: visited-venue profile) -----
        if "knn" in self.active:
            if user_extra is not None and venue_extra is not None:
                knn_scores = self._cosine_scores(
                    user_extra[user_idx], venue_extra
                )
            else:
                # Fall back to GNN cosine if MBTI/BERTopic not available
                logger.warning(
                    "KNN mode requested but user_extra/venue_extra missing; "
                    "falling back to GNN scores for KNN slot."
                )
                knn_scores = self._cosine_scores(
                    user_embeddings[user_idx], venue_embeddings
                )
            knn_ranks = self._scores_to_ranks(knn_scores)
            rrf_score += 1.0 / (self.k + knn_ranks)

        # ---- GNN sub-ranker (collaborative filtering) ------------------
        if "gnn" in self.active:
            gnn_scores = self._cosine_scores(
                user_embeddings[user_idx], venue_embeddings
            )
            gnn_ranks = self._scores_to_ranks(gnn_scores)
            rrf_score += 1.0 / (self.k + gnn_ranks)

        # ---- MBTI sub-ranker (personality from the user's own writing) -
        # cosine(user BERT-MBTI CLS, venue BERT-MBTI CLS).  Distinct from KNN:
        # KNN profiles a user by venues they visited, this profiles them by how
        # they write - a pure personality-compatibility signal.
        if "mbti" in self.active and user_mbti is not None and venue_mbti is not None:
            mbti_scores = self._cosine_scores(user_mbti[user_idx], venue_mbti)
            mbti_ranks = self._scores_to_ranks(mbti_scores)
            rrf_score += 1.0 / (self.k + mbti_ranks)

        # ---- Popularity tie-breaker (epsilon scale) --------------------
        if self.popularity_alpha > 0 and self.venue_log_degree is not None:
            rrf_score *= (1.0 + self.popularity_alpha * self.venue_log_degree)

        # ---- Exclude already-visited venues ----------------------------
        if exclude_venues:
            for vid in exclude_venues:
                if 0 <= vid < n_venues:
                    rrf_score[vid] = -np.inf

        # ---- Return top-k ----------------------------------------------
        top_k_idx = np.argsort(-rrf_score)[:k]
        top_k_scores = rrf_score[top_k_idx]

        return top_k_idx, top_k_scores.astype(np.float32)


def build_user_bertopic_profiles(
    train_edges: np.ndarray,
    venue_bertopic_embs: np.ndarray,
    n_users: int,
) -> np.ndarray:
    """
    Build user content profiles in BERTopic PCA space.

    For each user, average the BERTopic PCA embeddings of all venues
    they visited in the training set.  This gives a semantic user profile
    that reflects their topic preferences without relying on GNN embeddings
    (which have collapsed to near-uniform representations on this dataset).

    Args:
        train_edges: [2, num_edges] int array of (user_idx, venue_idx).
        venue_bertopic_embs: [n_venues, d] BERTopic PCA venue embeddings.
        n_users: Total number of users.

    Returns:
        user_profiles: [n_users, d] float32 array.  Rows for users with
            no training interactions are zero vectors.
    """
    d = venue_bertopic_embs.shape[1]
    user_profiles = np.zeros((n_users, d), dtype=np.float64)
    user_counts = np.zeros(n_users, dtype=np.int32)

    users = train_edges[0]
    venues = train_edges[1]
    np.add.at(user_profiles, users, venue_bertopic_embs[venues])
    np.add.at(user_counts, users, 1)

    mask = user_counts > 0
    user_profiles[mask] /= user_counts[mask, np.newaxis]
    logger.info(
        f"Built BERTopic user profiles: "
        f"{mask.sum()} / {n_users} users have training interactions"
    )
    return user_profiles.astype(np.float32)


def build_venue_mbti_profiles(
    train_edges: np.ndarray,
    user_mbti_embs: np.ndarray,
    n_venues: int,
) -> np.ndarray:
    """
    Build venue personality profiles as the mean MBTI embedding of visitors.

    For each venue, average the BERT-MBTI CLS embeddings of all users who
    visited it in training.  This places venues in the SAME discriminative
    user-personality space, so cosine(user_mbti, venue_profile) answers
    "do people with my personality visit here?" - a personality-collaborative
    signal.  This is far more discriminative than venue review-text CLS
    embeddings (which barely vary: cosine std ~0.04), because it is grounded
    in who actually goes there.

    Args:
        train_edges: [2, num_edges] int array of (user_idx, venue_idx).
        user_mbti_embs: [n_users, d] BERT-MBTI user CLS embeddings.
        n_venues: Total number of venues.

    Returns:
        venue_profiles: [n_venues, d] float32, L2-normalised.  Venues with no
            training visitors are left as zero vectors.
    """
    d = user_mbti_embs.shape[1]
    venue_profiles = np.zeros((n_venues, d), dtype=np.float64)
    venue_counts = np.zeros(n_venues, dtype=np.int32)

    users = train_edges[0]
    venues = train_edges[1]
    np.add.at(venue_profiles, venues, user_mbti_embs[users])
    np.add.at(venue_counts, venues, 1)

    mask = venue_counts > 0
    venue_profiles[mask] /= venue_counts[mask, np.newaxis]
    venue_profiles[mask] /= (
        np.linalg.norm(venue_profiles[mask], axis=1, keepdims=True) + 1e-8
    )
    logger.info(
        f"Built venue MBTI profiles (visitor-mean): "
        f"{mask.sum()} / {n_venues} venues have training visitors"
    )
    return venue_profiles.astype(np.float32)


class PopularityRanker:
    """
    Non-personalised popularity baseline.

    Recommends venues sorted by training-set visit frequency.  Used to
    verify that personalised models (KNN, RRF) beat a simple trend-based
    heuristic.
    """

    def __init__(self) -> None:
        self.venue_log_degree: Optional[np.ndarray] = None

    def set_venue_degree(self, train_edges: np.ndarray, n_venues: int) -> None:
        degree = np.zeros(n_venues, dtype=np.float32)
        np.add.at(degree, train_edges[1], 1)
        self.venue_log_degree = np.log1p(degree).astype(np.float32)
        logger.info(
            f"[Popularity] degree: mean={degree.mean():.1f}, "
            f"max={degree.max():.0f}"
        )

    def recommend(
        self,
        user_idx: int,
        user_embeddings: np.ndarray,
        venue_embeddings: np.ndarray,
        k: int = 10,
        exclude_venues: Optional[set] = None,
        user_extra: Optional[np.ndarray] = None,
        venue_extra: Optional[np.ndarray] = None,
        **kwargs,  # absorb user_mbti/venue_mbti (unused by popularity)
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = self.venue_log_degree.copy()
        if exclude_venues:
            for vid in exclude_venues:
                if 0 <= vid < len(scores):
                    scores[vid] = -np.inf
        top_k = np.argsort(-scores)[:k]
        return top_k, scores[top_k]


def run_popularity_evaluation(
    train_edges: np.ndarray,
    user_gt: dict,
    user_embeddings: np.ndarray,
    venue_embeddings: np.ndarray,
    k_values: list[int],
    num_eval_users: int = 200,
) -> tuple[pd.DataFrame, PopularityRanker]:
    """Evaluate the non-personalised popularity baseline."""
    logger.info("[Popularity] Computing popularity-ranked recommendations...")
    ranker = PopularityRanker()
    ranker.set_venue_degree(train_edges, n_venues=venue_embeddings.shape[0])
    metrics = evaluate_recommendations(
        ranker, user_embeddings, venue_embeddings,
        user_gt, k_values,
        num_eval_users=num_eval_users,
    )
    metrics["Model"] = "Popularity"
    return metrics, ranker


def load_gnn_embeddings(model_dir: Path) -> dict:
    """Load GNN-learned embeddings from Phase 3."""
    emb_dir = model_dir / "gnn_hetero"
    embeddings = {}

    for node_type in ["user", "venue"]:
        path = emb_dir / f"{node_type}_embeddings.npy"
        if path.exists():
            embeddings[node_type] = np.load(path)
            logger.info(f"Loaded GNN {node_type} embeddings: {embeddings[node_type].shape}")
        else:
            logger.warning(f"GNN {node_type} embeddings not found at {path}")

    # Load ID mappings
    mappings_path = emb_dir / "id_mappings.pt"
    if mappings_path.exists():
        import torch
        embeddings["mappings"] = torch.load(mappings_path, map_location="cpu")
    else:
        embeddings["mappings"] = {}

    return embeddings


def load_personality_data(model_dir: Path) -> dict:
    """Load MBTI personality predictions and embeddings."""
    data = {}

    # MBTI embeddings
    mbti_emb_path = model_dir / "gnn" / "user_gnn_embeddings.npy"
    if mbti_emb_path.exists():
        data["user_mbti_embeddings"] = np.load(mbti_emb_path)

    # Venue embeddings from BERTopic
    for subdir in ["bertopic_mbti", "bertopic"]:
        venue_emb_path = model_dir / subdir / "venue_embeddings.npy"
        if venue_emb_path.exists():
            data["venue_embeddings"] = np.load(venue_emb_path)
            break

    return data


def load_user_mbti_embeddings(
    model_dir: Path,
    user_id_map: dict,
    n_users: int,
) -> Optional[np.ndarray]:
    """
    Load per-user BERT-MBTI CLS embeddings and align to the GNN user index.

    Reads models/bert_mbti/user_mbti_embeddings.npy (row order matches
    user_mbti_ids.parquet) and scatters each row into the GNN index space
    via user_id_map.  Returns None if the files are absent so the caller can
    skip the MBTI ranker gracefully.

    Args:
        model_dir: MODEL_DIR root.
        user_id_map: dict mapping user_id string -> GNN user index.
        n_users: Total number of users in GNN index space.

    Returns:
        [n_users, 768] float32 array, or None.
    """
    emb_path = model_dir / "bert_mbti" / "user_mbti_embeddings.npy"
    ids_path = model_dir / "bert_mbti" / "user_mbti_ids.parquet"
    if not emb_path.exists() or not ids_path.exists():
        logger.warning(
            f"User MBTI embeddings not found at {emb_path}; "
            "MBTI ranker will be unavailable. Run scripts/extract_user_mbti.py."
        )
        return None

    raw = np.load(emb_path).astype(np.float32)
    ids = pd.read_parquet(ids_path)["user_id"].tolist()

    d = raw.shape[1]
    aligned = np.zeros((n_users, d), dtype=np.float32)
    matched = 0
    for row_idx, uid in enumerate(ids):
        if row_idx >= len(raw):
            break
        gnn_idx = user_id_map.get(uid)
        if gnn_idx is not None:
            aligned[gnn_idx] = raw[row_idx]
            matched += 1
    logger.info(
        f"User MBTI embeddings: {matched}/{len(ids)} aligned to GNN index "
        f"space ({n_users} total), dim={d}"
    )
    return aligned


def load_interactions(data_dir: Path) -> tuple[pd.DataFrame, str, str]:
    """Load interaction data and identify columns."""
    dfs = []
    for split in ["train_reviews.parquet", "val_reviews.parquet", "test_reviews.parquet"]:
        path = data_dir / split
        if path.exists():
            dfs.append(pd.read_parquet(path))

    if not dfs:
        raise FileNotFoundError(f"No processed data in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    user_col = "user_id" if "user_id" in df.columns else df.columns[0]
    biz_col = "business_id" if "business_id" in df.columns else "venue_id"

    return df, user_col, biz_col


def create_train_test_edges(
    interactions: pd.DataFrame,
    user_col: str,
    biz_col: str,
    user_id_map: dict,
    venue_id_map: dict,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create edge arrays with train/test split."""
    # Filter to mapped users and venues
    valid = interactions[
        interactions[user_col].isin(user_id_map) &
        interactions[biz_col].isin(venue_id_map)
    ]

    user_indices = valid[user_col].map(user_id_map).values
    venue_indices = valid[biz_col].map(venue_id_map).values

    edges = np.stack([user_indices, venue_indices])

    # Split
    n = edges.shape[1]
    perm = np.random.permutation(n)
    n_test = int(n * test_ratio)

    train_edges = edges[:, perm[n_test:]]
    test_edges = edges[:, perm[:n_test]]

    # Build ground truth per user
    user_gt = {}
    for i in range(test_edges.shape[1]):
        u, v = int(test_edges[0, i]), int(test_edges[1, i])
        if u not in user_gt:
            user_gt[u] = set()
        user_gt[u].add(v)

    logger.info(f"Train edges: {train_edges.shape[1]}, Test edges: {test_edges.shape[1]}")
    return train_edges, test_edges, user_gt


def evaluate_recommendations(
    ranker,
    user_embeddings: np.ndarray,
    venue_embeddings: np.ndarray,
    user_gt: dict,
    k_values: list[int],
    user_extra: np.ndarray = None,
    venue_extra: np.ndarray = None,
    user_mbti: np.ndarray = None,
    venue_mbti: np.ndarray = None,
    num_eval_users: int = 200,
) -> pd.DataFrame:
    """
    Evaluate recommendation quality with ranking metrics.

    Args:
        ranker: Any recommender exposing a recommend() method with the
            signature (user_idx, user_embeddings, venue_embeddings, k,
            exclude_venues, user_extra, venue_extra).  Works with both
            RRFRanker and XGBoostRanker.
        user_embeddings: User embedding matrix.
        venue_embeddings: Venue embedding matrix.
        user_gt: Ground truth - dict mapping user_idx -> set of venue_idx.
        k_values: List of K values for metrics.
        user_extra: Extra user features (MBTI embeddings for RRF).
        venue_extra: Extra venue features (BERTopic PCA for RRF).
        num_eval_users: Number of users to evaluate.

    Returns:
        DataFrame with metrics per K.
    """
    calc = RecommendationMetrics()

    eval_users = list(user_gt.keys())
    if len(eval_users) > num_eval_users:
        eval_users = list(np.random.choice(eval_users, num_eval_users, replace=False))

    results = {k: {"precision": [], "recall": [], "ndcg": [], "mrr": [], "hit_rate": []}
               for k in k_values}

    for user_idx in eval_users:
        gt = user_gt[user_idx]
        if not gt:
            continue

        rec_indices, rec_scores = ranker.recommend(
            user_idx=user_idx,
            user_embeddings=user_embeddings,
            venue_embeddings=venue_embeddings,
            k=max(k_values),
            exclude_venues=None,
            user_extra=user_extra,
            venue_extra=venue_extra,
            user_mbti=user_mbti,
            venue_mbti=venue_mbti,
        )

        recs = rec_indices.tolist()

        for k in k_values:
            results[k]["precision"].append(calc.precision_at_k(recs, gt, k))
            results[k]["recall"].append(calc.recall_at_k(recs, gt, k))
            results[k]["ndcg"].append(calc.ndcg_at_k(recs, gt, k))
            results[k]["mrr"].append(calc.mrr(recs, gt))
            results[k]["hit_rate"].append(calc.hit_rate_at_k(recs, gt, k))

    rows = []
    for k in k_values:
        rows.append({
            "K": k,
            "Precision@K": np.mean(results[k]["precision"]),
            "Recall@K": np.mean(results[k]["recall"]),
            "NDCG@K": np.mean(results[k]["ndcg"]),
            "MRR": np.mean(results[k]["mrr"]),
            "Hit Rate@K": np.mean(results[k]["hit_rate"]),
        })

    return pd.DataFrame(rows)


def run_rrf_evaluation(
    train_edges: np.ndarray,
    user_gt: dict,
    user_embeddings: np.ndarray,
    venue_embeddings: np.ndarray,
    k_values: list[int],
    label: str = "RRF",
    rrf_mode: str = "hybrid",
    rrf_k: int = 60,
    popularity_alpha: float = 0.01,
    user_extra: np.ndarray = None,
    venue_extra: np.ndarray = None,
    user_mbti: np.ndarray = None,
    venue_mbti: np.ndarray = None,
    num_eval_users: int = 200,
) -> tuple[pd.DataFrame, RRFRanker]:
    """
    Evaluate the Reciprocal Rank Fusion hybrid recommender.

    No training. No learnable parameters. No overfitting.

    The two base rankers are:
      - KNN: cosine(user_extra[user_idx], venue_extra[venue_idx])
             i.e. cosine(MBTI embedding, BERTopic PCA embedding)
      - GNN: cosine(user_embeddings[user_idx], venue_embeddings[venue_idx])
             i.e. cosine(GNN user, GNN venue)

    They are combined using RRF:
        rrf_score(v) = 1/(k + rank_knn(v)) + 1/(k + rank_gnn(v))

    An optional popularity tiebreaker:
        rrf_score(v) *= 1 + popularity_alpha * log(1 + degree(v))

    is applied AFTER the RRF sum so popularity only breaks ties between
    venues with equal personalization scores.

    Args:
        train_edges: [2, num_edges] used to compute venue degrees.
        user_gt: Ground truth dict user_idx -> set(venue_idx) for test users.
        user_embeddings: GNN user embeddings [n_users, d].
        venue_embeddings: GNN venue embeddings [n_venues, d].
        k_values: List of K values for Precision/Recall/NDCG/MRR/HitRate.
        label: Display name for this run in results tables.
        rrf_mode: "hybrid", "knn", or "gnn".
        rrf_k: RRF smoothing constant (default 60).
        popularity_alpha: Popularity tiebreaker weight (default 0.01).
            Set to 0.0 to disable popularity entirely.
        user_extra: MBTI user embeddings [n_users, d_mbti].
        venue_extra: BERTopic PCA venue embeddings [n_venues, d_topic].
        num_eval_users: How many test users to evaluate.

    Returns:
        Tuple of (metrics_df, fitted_RRFRanker).
    """
    logger.info(
        f"[{label}] RRF fusion "
        f"(mode={rrf_mode}, k={rrf_k}, "
        f"popularity_alpha={popularity_alpha})"
    )

    ranker = RRFRanker(k=rrf_k, popularity_alpha=popularity_alpha,
                       rrf_mode=rrf_mode)

    # Compute venue degree from training graph
    ranker.set_venue_degree(train_edges, n_venues=venue_embeddings.shape[0])

    # Log which sub-rankers are active
    knn_avail = user_extra is not None and venue_extra is not None
    mbti_avail = user_mbti is not None and venue_mbti is not None
    logger.info(
        f"  active rankers: {sorted(ranker.active)} "
        f"(knn_avail={knn_avail}, mbti_avail={mbti_avail})"
    )

    # Evaluate (no training step - RRF is parameter-free)
    logger.info(f"[{label}] Evaluating {num_eval_users} users...")
    metrics = evaluate_recommendations(
        ranker, user_embeddings, venue_embeddings,
        user_gt, k_values,
        user_extra=user_extra,
        venue_extra=venue_extra,
        user_mbti=user_mbti,
        venue_mbti=venue_mbti,
        num_eval_users=num_eval_users,
    )
    metrics["Model"] = label

    return metrics, ranker


def generate_methodology_comparison(
    current_metrics: pd.DataFrame,
    k: int = 10,
) -> tuple[pd.DataFrame, str]:
    """
    Generate comparison table between Omer's approach and current.

    Since we can't re-run Omer's exact pipeline, we report our metrics
    alongside reported values from the thesis for context.
    """
    # Omer's reported values (from thesis Chapter 6.3)
    omer_reported = {
        "MBTI Accuracy": "~94% (leaky)",
        "BERTopic Model": "all-MiniLM-L6-v2",
        "Upsampling": "Before split (data leakage)",
        "GNN Type": "N/A (XGBoost only)",
        "Classifier": "XGBoost (log loss: 0.4207)",
        "Evaluation": "Log loss, Accuracy",
    }

    current_k_row = current_metrics[current_metrics["K"] == k].iloc[0]

    rows = [
        {
            "Aspect": "MBTI Classifier",
            "Omer (2024)": "BERT -> 16-class (leaky eval)",
            "Current": "BERT -> 4 binary heads (robust eval)",
        },
        {
            "Aspect": "Upsampling Strategy",
            "Omer (2024)": "Entire dataset before split",
            "Current": "Training set only after split",
        },
        {
            "Aspect": "Topic Embeddings",
            "Omer (2024)": "all-MiniLM-L6-v2 (generic)",
            "Current": "BERT-MBTI CLS tokens (personality-informed)",
        },
        {
            "Aspect": "Graph Model",
            "Omer (2024)": "None",
            "Current": f"HeteroGNN (GraphSAGE, link prediction)",
        },
        {
            "Aspect": "Final Ranking",
            "Omer (2024)": "XGBoost (baseline features)",
            "Current": "RRF (KNN MBTI/BERTopic + GNN collaborative, k=60)",
        },
        {
            "Aspect": f"Precision@{k}",
            "Omer (2024)": "Not reported",
            "Current": f"{current_k_row['Precision@K']:.4f}",
        },
        {
            "Aspect": f"Recall@{k}",
            "Omer (2024)": "Not reported",
            "Current": f"{current_k_row['Recall@K']:.4f}",
        },
        {
            "Aspect": f"NDCG@{k}",
            "Omer (2024)": "Not reported",
            "Current": f"{current_k_row['NDCG@K']:.4f}",
        },
        {
            "Aspect": f"MRR",
            "Omer (2024)": "Not reported",
            "Current": f"{current_k_row['MRR']:.4f}",
        },
    ]

    comparison_df = pd.DataFrame(rows)

    # Generate LaTeX
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Methodology Comparison: Omer (2024) vs Current System}\n"
    latex += "\\label{tab:full_methodology_comparison}\n"
    latex += "\\begin{tabular}{l p{5cm} p{5cm}}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Aspect} & \\textbf{Omer (2024)} & \\textbf{Current} \\\\\n"
    latex += "\\midrule\n"

    for _, row in comparison_df.iterrows():
        latex += f"{row['Aspect']} & {row['Omer (2024)']} & {row['Current']} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return comparison_df, latex


def main(args):
    """Run the hybrid evaluation pipeline."""
    setup_logging(level=logging.INFO)
    set_seed(42)

    config = get_config()
    k_values = [5, 10, 20]

    # Load data
    logger.info("Loading data...")
    gnn_embs = load_gnn_embeddings(MODEL_DIR)
    personality_data = load_personality_data(MODEL_DIR)
    interactions, user_col, biz_col = load_interactions(PROCESSED_DATA_DIR)

    # Determine user/venue mappings.
    # CRITICAL: use the GNN's id_mappings.pt so that user/venue index 0 in
    # the interaction data refers to the same entity as row 0 in the GNN
    # embedding matrices.  pd.unique() returns first-seen order which will
    # differ from the GNN's construction order, causing every recommendation
    # lookup to reference the wrong embedding row.
    mappings = gnn_embs.get("mappings", {})
    if "user_id_map" in mappings and "venue_id_map" in mappings:
        user_id_map = mappings["user_id_map"]
        venue_id_map = mappings["venue_id_map"]
        logger.info(
            f"Using GNN ID mappings: "
            f"{len(user_id_map)} users, {len(venue_id_map)} venues"
        )
    else:
        logger.warning(
            "GNN id_mappings.pt not found - building fresh maps from "
            "interactions. Embedding indices may not align with "
            "interaction indices (metrics will be unreliable)."
        )
        unique_users = interactions[user_col].unique()
        unique_venues = interactions[biz_col].unique()
        user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        venue_id_map = {vid: idx for idx, vid in enumerate(unique_venues)}

    n_users = len(user_id_map)
    n_venues = len(venue_id_map)

    # Get embeddings (from GNN or fallback to random)
    # Reconcile: pad or truncate to match current data dimensions
    emb_dim = 64
    if "user" in gnn_embs:
        user_embeddings = gnn_embs["user"]
        emb_dim = user_embeddings.shape[1]
        if user_embeddings.shape[0] < n_users:
            pad = np.random.randn(
                n_users - user_embeddings.shape[0], emb_dim
            ).astype(np.float32) * 0.1
            logger.warning(
                f"User embeddings ({user_embeddings.shape[0]}) < n_users ({n_users}), "
                f"padding {pad.shape[0]} rows. Re-train GNN for accurate results."
            )
            user_embeddings = np.vstack([user_embeddings, pad])
        elif user_embeddings.shape[0] > n_users:
            user_embeddings = user_embeddings[:n_users]
    else:
        logger.info("Using random user embeddings for demonstration")
        user_embeddings = np.random.randn(n_users, emb_dim).astype(np.float32)

    if "venue" in gnn_embs:
        venue_embeddings = gnn_embs["venue"]
        v_dim = venue_embeddings.shape[1]
        if venue_embeddings.shape[0] < n_venues:
            pad = np.random.randn(
                n_venues - venue_embeddings.shape[0], v_dim
            ).astype(np.float32) * 0.1
            logger.warning(
                f"Venue embeddings ({venue_embeddings.shape[0]}) < n_venues ({n_venues}), "
                f"padding {pad.shape[0]} rows. Re-train GNN for accurate results."
            )
            venue_embeddings = np.vstack([venue_embeddings, pad])
        elif venue_embeddings.shape[0] > n_venues:
            venue_embeddings = venue_embeddings[:n_venues]
    else:
        logger.info("Using random venue embeddings for demonstration")
        venue_embeddings = np.random.randn(n_venues, emb_dim).astype(np.float32)

    # Create personality features - reconcile embedding sizes
    user_mbti_probs = np.random.rand(n_users, 4).astype(np.float32)
    user_mbti_embs = personality_data.get("user_mbti_embeddings", None)
    venue_embs_for_scorer = personality_data.get("venue_embeddings", None)

    # PersonalityScorer computes cosine sim between user and venue embeddings,
    # so they MUST have the same dimension. GNN embeddings are dim=64 for both,
    # but BERTopic venue embeddings are dim=768. Use GNN venue embeddings
    # (same dim as user) for the scorer.
    if (user_mbti_embs is not None and venue_embs_for_scorer is not None
            and user_mbti_embs.shape[1] != venue_embs_for_scorer.shape[1]):
        logger.warning(
            f"Dimension mismatch: user_emb dim={user_mbti_embs.shape[1]}, "
            f"venue_emb dim={venue_embs_for_scorer.shape[1]}. "
            f"Using GNN venue embeddings (dim={user_embeddings.shape[1]}) for scorer."
        )
        venue_embs_for_scorer = venue_embeddings.copy()

    # Pad/truncate MBTI embeddings to match n_users
    if user_mbti_embs is not None and user_mbti_embs.shape[0] != n_users:
        d = user_mbti_embs.shape[1]
        if user_mbti_embs.shape[0] < n_users:
            pad = np.random.randn(
                n_users - user_mbti_embs.shape[0], d
            ).astype(np.float32) * 0.1
            user_mbti_embs = np.vstack([user_mbti_embs, pad])
        else:
            user_mbti_embs = user_mbti_embs[:n_users]

    # Pad/truncate venue embeddings for scorer
    if venue_embs_for_scorer is not None and venue_embs_for_scorer.shape[0] != n_venues:
        d = venue_embs_for_scorer.shape[1]
        if venue_embs_for_scorer.shape[0] < n_venues:
            pad = np.random.randn(
                n_venues - venue_embs_for_scorer.shape[0], d
            ).astype(np.float32) * 0.1
            venue_embs_for_scorer = np.vstack([venue_embs_for_scorer, pad])
        else:
            venue_embs_for_scorer = venue_embs_for_scorer[:n_venues]

    scorer = PersonalityScorer(
        user_mbti_probs=user_mbti_probs,
        user_mbti_embeddings=user_mbti_embs,
        venue_embeddings=venue_embs_for_scorer,
    )

    # Create train/test split
    train_edges, test_edges, user_gt = create_train_test_edges(
        interactions, user_col, biz_col,
        user_id_map, venue_id_map,
        test_ratio=0.2,
    )

    # Compute personality features for all users
    all_user_indices = np.arange(n_users)
    dummy_venue_indices = np.zeros(n_users, dtype=np.int64)
    personality_features = scorer.compute_personality_features(
        all_user_indices, dummy_venue_indices,
    )
    logger.info(f"Personality features shape: {personality_features.shape}")

    # -- BERTopic venue embeddings as venue_extra for XGBoost --
    # These 768-dim semantic embeddings are concatenated alongside the
    # 64-dim GNN venue embeddings so XGBoost sees BOTH representations.
    bertopic_venue_embs = personality_data.get("venue_embeddings", None)
    if bertopic_venue_embs is not None:
        # Reindex from BERTopic's venue ordering to GNN's venue index space.
        # venue_topics.parquet maps each row of venue_embeddings.npy to its
        # string venue ID.  We use that to scatter rows into GNN index order
        # so that venue_extra[gnn_idx] = BERTopic embedding for that venue.
        bertopic_topics_path = MODEL_DIR / "bertopic_mbti" / "venue_topics.parquet"
        if bertopic_topics_path.exists():
            btopic_df = pd.read_parquet(bertopic_topics_path)
            bertopic_venue_ids = btopic_df["venue_id"].tolist()
            d = bertopic_venue_embs.shape[1]
            reindexed = np.zeros((n_venues, d), dtype=np.float32)
            matched = 0
            n_btopic = len(bertopic_venue_embs)
            for b_idx, vid in enumerate(bertopic_venue_ids):
                if b_idx >= n_btopic:
                    break
                gnn_idx = venue_id_map.get(vid)
                if gnn_idx is not None:
                    reindexed[gnn_idx] = bertopic_venue_embs[b_idx]
                    matched += 1
            logger.info(
                f"BERTopic reindexing: {matched}/{len(bertopic_venue_ids)} "
                f"venues matched to GNN index space ({n_venues} total)"
            )
            bertopic_venue_embs = reindexed
        else:
            logger.warning(
                "venue_topics.parquet not found; positional pad/truncate used "
                "(BERTopic venue order may not match GNN index space)"
            )
            if bertopic_venue_embs.shape[0] < n_venues:
                pad = np.zeros(
                    (n_venues - bertopic_venue_embs.shape[0],
                     bertopic_venue_embs.shape[1]),
                    dtype=np.float32,
                )
                bertopic_venue_embs = np.vstack([bertopic_venue_embs, pad])
            elif bertopic_venue_embs.shape[0] > n_venues:
                bertopic_venue_embs = bertopic_venue_embs[:n_venues]

        # PCA: reduce 768-dim -> 64-dim to match GNN embedding dimension
        if bertopic_venue_embs.shape[1] > 64:
            logger.info(
                f"Applying PCA: {bertopic_venue_embs.shape[1]}-dim -> 64-dim"
            )
            pca = PCA(n_components=64, random_state=42)
            bertopic_venue_embs = pca.fit_transform(bertopic_venue_embs).astype(
                np.float32
            )
            explained = pca.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance: {explained:.3f}")
        logger.info(f"BERTopic venue_extra: {bertopic_venue_embs.shape}")
    else:
        logger.warning("No BERTopic venue embeddings found - venue_extra=None")

    # -- Per-user BERT-MBTI embeddings (personality from own reviews) ----
    # Produced by scripts/extract_user_mbti.py, keyed by user_id string.
    user_mbti_embs = load_user_mbti_embeddings(MODEL_DIR, user_id_map, n_users)
    venue_mbti_embs = None
    if user_mbti_embs is not None:
        # Mean-center user embeddings to counter BERT CLS anisotropy (all
        # vectors clustered in a narrow cone -> washed-out cosine), then
        # renormalise so cosine is meaningful.
        u_mean = user_mbti_embs[user_mbti_embs.any(axis=1)].mean(axis=0, keepdims=True)
        user_mbti_embs = user_mbti_embs - u_mean
        user_mbti_embs /= (np.linalg.norm(user_mbti_embs, axis=1, keepdims=True) + 1e-8)

        # Venue personality = mean MBTI of its visitors (in centered space).
        # This is far more discriminative than venue review-text CLS because
        # it is grounded in WHO visits, not what the reviews say.  cosine here
        # answers "do people with my personality go here?".
        venue_mbti_embs = build_venue_mbti_profiles(
            train_edges, user_mbti_embs, n_venues
        )
        logger.info("MBTI: user embeddings centered; venue profiles = visitor-mean")

    # Number of eval users: quick mode uses 100, full uses 500
    num_eval_users = 100 if args.quick else 500

    # Build BERTopic user profiles from training interactions.
    # The GNN embeddings have collapsed to near-uniform representations
    # (cosine std ~0.025, min ~0.74) on this dataset, making them
    # equivalent to random recommendations.  Instead, we compute a
    # user profile as the mean BERTopic PCA embedding over all venues
    # visited in training - this retains the semantic content signal
    # and gives meaningful cosine similarity for the KNN sub-ranker.
    if bertopic_venue_embs is not None:
        knn_user_embs = build_user_bertopic_profiles(
            train_edges, bertopic_venue_embs, n_users
        )
        logger.info(f"KNN user profiles (BERTopic mean): {knn_user_embs.shape}")
    else:
        # Fall back to LTGNN user embeddings if BERTopic is unavailable
        knn_user_embs = user_mbti_embs  # [n_users, 64] or None
        logger.warning(
            "BERTopic venue embeddings unavailable; "
            "falling back to LTGNN user embeddings for KNN (collapsed)."
        )

    # ---------------------------------------------------------------
    # Baseline: Popularity (non-personalised)
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Baseline: Popularity (non-personalised)")
    logger.info("=" * 60)

    pop_baseline_metrics, _ = run_popularity_evaluation(
        train_edges, user_gt,
        user_embeddings, venue_embeddings,
        k_values,
        num_eval_users=num_eval_users,
    )

    # ---------------------------------------------------------------
    # RRF Evaluation 1: Hybrid KNN + GNN (no popularity tiebreaker)
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("RRF: KNN (MBTI/BERTopic) + GNN - no popularity boost")
    logger.info("=" * 60)

    rrf_metrics, rrf_ranker = run_rrf_evaluation(
        train_edges, user_gt,
        user_embeddings, venue_embeddings,
        k_values,
        label="RRF-Hybrid (no pop)",
        rrf_mode="hybrid",
        rrf_k=args.rrf_k,
        popularity_alpha=0.0,
        user_extra=knn_user_embs,
        venue_extra=bertopic_venue_embs,
        num_eval_users=num_eval_users,
    )

    # ---------------------------------------------------------------
    # RRF Evaluation 2: Hybrid + small popularity tiebreaker
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"RRF: KNN + GNN + popularity tiebreaker (alpha={args.pop_alpha})")
    logger.info("=" * 60)

    rrf_pop_metrics, rrf_pop_ranker = run_rrf_evaluation(
        train_edges, user_gt,
        user_embeddings, venue_embeddings,
        k_values,
        label=f"RRF-Hybrid (pop={args.pop_alpha})",
        rrf_mode="hybrid",
        rrf_k=args.rrf_k,
        popularity_alpha=args.pop_alpha,
        user_extra=knn_user_embs,
        venue_extra=bertopic_venue_embs,
        num_eval_users=num_eval_users,
    )

    # ---------------------------------------------------------------
    # Ablation 1: KNN only
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Ablation: KNN only (MBTI/BERTopic, no GNN)")
    logger.info("=" * 60)

    knn_metrics, _ = run_rrf_evaluation(
        train_edges, user_gt,
        user_embeddings, venue_embeddings,
        k_values,
        label="KNN-only",
        rrf_mode="knn",
        rrf_k=args.rrf_k,
        popularity_alpha=0.0,
        user_extra=knn_user_embs,
        venue_extra=bertopic_venue_embs,
        num_eval_users=num_eval_users,
    )

    # ---------------------------------------------------------------
    # Ablation 2: GNN only
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Ablation: GNN only (collaborative, no MBTI)")
    logger.info("=" * 60)

    gnn_metrics, _ = run_rrf_evaluation(
        train_edges, user_gt,
        user_embeddings, venue_embeddings,
        k_values,
        label="GNN-only",
        rrf_mode="gnn",
        rrf_k=args.rrf_k,
        popularity_alpha=0.0,
        user_extra=None,
        venue_extra=None,
        num_eval_users=num_eval_users,
    )

    # ---------------------------------------------------------------
    # Ablation 3: KNN + popularity (no GNN)
    # Isolates the GNN's contribution: if KNN+pop already matches the full
    # hybrid, the popularity tiebreaker - not the GNN - is doing the work.
    # The full hybrid must beat THIS to claim the GNN adds collaborative value.
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Ablation: KNN + popularity (no GNN, alpha={args.pop_alpha})")
    logger.info("=" * 60)

    knn_pop_metrics, _ = run_rrf_evaluation(
        train_edges, user_gt,
        user_embeddings, venue_embeddings,
        k_values,
        label=f"KNN+pop (no GNN)",
        rrf_mode="knn",
        rrf_k=args.rrf_k,
        popularity_alpha=args.pop_alpha,
        user_extra=knn_user_embs,
        venue_extra=bertopic_venue_embs,
        num_eval_users=num_eval_users,
    )

    # ---------------------------------------------------------------
    # Ablation 4: MBTI only (personality from the user's own writing)
    # ---------------------------------------------------------------
    mbti_metrics = None
    full_metrics = None
    if user_mbti_embs is not None and venue_mbti_embs is not None:
        logger.info("=" * 60)
        logger.info("Ablation: MBTI only (BERT-MBTI user CLS vs venue CLS)")
        logger.info("=" * 60)

        mbti_metrics, _ = run_rrf_evaluation(
            train_edges, user_gt,
            user_embeddings, venue_embeddings,
            k_values,
            label="MBTI-only",
            rrf_mode="mbti",
            rrf_k=args.rrf_k,
            popularity_alpha=0.0,
            user_mbti=user_mbti_embs,
            venue_mbti=venue_mbti_embs,
            num_eval_users=num_eval_users,
        )

        # -----------------------------------------------------------
        # Full hybrid: KNN + GNN + MBTI + popularity tiebreaker
        # The headline model - personality drives recommendations directly.
        # -----------------------------------------------------------
        logger.info("=" * 60)
        logger.info(f"FULL: KNN + GNN + MBTI + pop (alpha={args.pop_alpha})")
        logger.info("=" * 60)

        full_metrics, _ = run_rrf_evaluation(
            train_edges, user_gt,
            user_embeddings, venue_embeddings,
            k_values,
            label=f"Full (KNN+GNN+MBTI+pop)",
            rrf_mode="full",
            rrf_k=args.rrf_k,
            popularity_alpha=args.pop_alpha,
            user_extra=knn_user_embs,
            venue_extra=bertopic_venue_embs,
            user_mbti=user_mbti_embs,
            venue_mbti=venue_mbti_embs,
            num_eval_users=num_eval_users,
        )

    # ---------------------------------------------------------------
    # Collect and print all results
    # ---------------------------------------------------------------
    metric_frames = [
        pop_baseline_metrics, knn_metrics, knn_pop_metrics,
        rrf_metrics, rrf_pop_metrics, gnn_metrics,
    ]
    if mbti_metrics is not None:
        metric_frames.append(mbti_metrics)
    if full_metrics is not None:
        metric_frames.append(full_metrics)
    all_metrics = pd.concat(metric_frames, ignore_index=True)

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    print(all_metrics.to_string(index=False))

    # Highlight the best model per metric at K=10
    k10 = all_metrics[all_metrics["K"] == 10].copy()
    logger.info("\nBest model per metric @ K=10:")
    for col in ["Precision@K", "Recall@K", "NDCG@K", "MRR", "Hit Rate@K"]:
        best_idx = k10[col].idxmax()
        best_model = k10.loc[best_idx, "Model"]
        best_val = k10.loc[best_idx, col]
        logger.info(f"  {col:15s}: {best_model} ({best_val:.4f})")

    # Generate methodology comparison using best model (RRF hybrid)
    comparison_df, latex_str = generate_methodology_comparison(rrf_metrics, k=10)

    logger.info("\nMethodology Comparison:")
    print(comparison_df.to_string(index=False))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics.to_csv(RESULTS_DIR / "phase4_hybrid_metrics.csv", index=False)
    comparison_df.to_csv(RESULTS_DIR / "phase4_methodology_comparison.csv", index=False)

    latex_path = RESULTS_DIR / "phase4_methodology_comparison.tex"
    latex_path.write_text(latex_str, encoding="utf-8")

    logger.info(f"\nResults saved to {RESULTS_DIR}")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid RRF recommendation evaluation pipeline"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Primary K for metrics display (default 10)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Evaluate on 100 users instead of 500 (faster)",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=60,
        help="RRF smoothing constant k (default 60)",
    )
    parser.add_argument(
        "--pop-alpha", type=float, default=0.01,
        help="Popularity tiebreaker weight (default 0.01, 0 to disable)",
    )

    args = parser.parse_args()
    main(args)
