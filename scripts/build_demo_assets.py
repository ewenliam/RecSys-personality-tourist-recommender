#!/usr/bin/env python3
"""
Precompute compact artifacts for the Streamlit demo (demo/app.py).

Mirrors the exact preparation done by scripts/evaluate_hybrid.py (same ID
mappings, same BERTopic reindexing, same PCA seed, same MBTI centering, same
visitor-mean venue profiles) but does it ONCE and saves small aligned arrays
so the demo starts in seconds instead of minutes.

One deliberate deviation from the evaluation script: profiles, popularity and
visited-venue lists here use ALL interactions (train+val+test), not a train
split - the demo is a serving surface, not an evaluation, so it should use
every interaction we know about.

Outputs (models/demo/):
    venue_meta.parquet          venue_id, name, city, state, stars,
                                categories, n_visits  (row = venue index)
    user_index.parquet          user_id, n_visits, has_mbti (row = user index)
    venue_bertopic_pca64.npy    [n_v, 64]  content keys (KNN signal)
    knn_user_profiles.npy       [n_u, 64]  fp16, mean PCA of visited venues
    user_mbti_centered.npy      [n_u, 768] fp16, centered+renormalised CLS
    user_traits.npy             [n_u, 4]   fp16, P(I),P(N),P(F),P(P)
    venue_mbti_profiles.npy     [n_v, 768] fp16, visitor-mean personality
    mbti_center.npy             [768]      the mean used for centering
    venue_log_degree.npy        [n_v]      popularity tie-breaker
    visits_indptr.npy           [n_u+1]    CSR pointers into visits_venues
    visits_venues.npy           [nnz]      visited venue indices per user

Usage:
    python scripts/build_demo_assets.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import PROCESSED_DATA_DIR, MODEL_DIR
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)

OUT_DIR = MODEL_DIR / "demo"


def load_id_maps() -> tuple[dict, dict]:
    """Load the canonical GNN id mappings (index space of all embeddings)."""
    mappings = torch.load(
        MODEL_DIR / "gnn_hetero" / "id_mappings.pt",
        map_location="cpu", weights_only=False,
    )
    return mappings["user_id_map"], mappings["venue_id_map"]


def load_all_edges(user_id_map: dict, venue_id_map: dict) -> np.ndarray:
    """All (user_idx, venue_idx) interactions across train/val/test."""
    frames = []
    for split in ("train_reviews.parquet", "val_reviews.parquet",
                  "test_reviews.parquet"):
        path = PROCESSED_DATA_DIR / split
        if path.exists():
            frames.append(pd.read_parquet(
                path, columns=["user_id", "business_id"]))
    df = pd.concat(frames, ignore_index=True)
    valid = df[df["user_id"].isin(user_id_map)
               & df["business_id"].isin(venue_id_map)]
    edges = np.stack([
        valid["user_id"].map(user_id_map).values.astype(np.int32),
        valid["business_id"].map(venue_id_map).values.astype(np.int32),
    ])
    logger.info(f"Edges: {edges.shape[1]:,} interactions "
                f"({len(df) - edges.shape[1]:,} unmapped dropped)")
    return edges


def build_venue_pca64(venue_id_map: dict, n_venues: int) -> np.ndarray:
    """Reindex BERTopic MBTI venue embeddings to GNN space, PCA to 64."""
    raw = np.load(MODEL_DIR / "bertopic_mbti" / "venue_embeddings.npy")
    ids = pd.read_parquet(
        MODEL_DIR / "bertopic_mbti" / "venue_topics.parquet")["venue_id"]
    reindexed = np.zeros((n_venues, raw.shape[1]), dtype=np.float32)
    matched = 0
    for b_idx, vid in enumerate(ids):
        if b_idx >= len(raw):
            break
        g = venue_id_map.get(vid)
        if g is not None:
            reindexed[g] = raw[b_idx]
            matched += 1
    logger.info(f"BERTopic reindexing: {matched}/{len(ids)} venues matched")

    # Same reduction as evaluate_hybrid.py (PCA 768 -> 64, seed 42).
    pca = PCA(n_components=64, random_state=42)
    out = pca.fit_transform(reindexed).astype(np.float32)
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    return out


def build_user_profiles(edges: np.ndarray, venue_pca: np.ndarray,
                        n_users: int) -> np.ndarray:
    """User content profile = mean venue PCA embedding over visited venues."""
    d = venue_pca.shape[1]
    prof = np.zeros((n_users, d), dtype=np.float64)
    cnt = np.zeros(n_users, dtype=np.int32)
    np.add.at(prof, edges[0], venue_pca[edges[1]])
    np.add.at(cnt, edges[0], 1)
    mask = cnt > 0
    prof[mask] /= cnt[mask, None]
    logger.info(f"KNN user profiles: {mask.sum():,}/{n_users:,} users covered")
    return prof.astype(np.float32)


def build_mbti_arrays(user_id_map: dict, n_users: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align user MBTI CLS embeddings + traits, center + renormalise."""
    emb = np.load(MODEL_DIR / "bert_mbti" / "user_mbti_embeddings.npy"
                  ).astype(np.float32)
    traits = np.load(MODEL_DIR / "bert_mbti" / "user_mbti_traits.npy"
                     ).astype(np.float32)
    ids = pd.read_parquet(
        MODEL_DIR / "bert_mbti" / "user_mbti_ids.parquet")["user_id"]

    aligned = np.zeros((n_users, emb.shape[1]), dtype=np.float32)
    aligned_traits = np.full((n_users, 4), np.nan, dtype=np.float32)
    matched = 0
    for row, uid in enumerate(ids):
        if row >= len(emb):
            break
        g = user_id_map.get(uid)
        if g is not None:
            aligned[g] = emb[row]
            aligned_traits[g] = traits[row]
            matched += 1
    logger.info(f"User MBTI: {matched:,}/{len(ids):,} aligned")

    # Same anisotropy correction as evaluate_hybrid.py: subtract the mean of
    # non-zero rows, renormalise. The center is SAVED so that cold-start
    # users in the demo get the identical transform.
    center = aligned[aligned.any(axis=1)].mean(axis=0, keepdims=True)
    aligned -= center
    aligned /= np.linalg.norm(aligned, axis=1, keepdims=True) + 1e-8
    return aligned, aligned_traits, center[0]


def build_venue_mbti(edges: np.ndarray, user_mbti: np.ndarray,
                     n_venues: int) -> np.ndarray:
    """Venue personality = L2-normalised mean MBTI of its visitors."""
    d = user_mbti.shape[1]
    prof = np.zeros((n_venues, d), dtype=np.float64)
    cnt = np.zeros(n_venues, dtype=np.int32)
    np.add.at(prof, edges[1], user_mbti[edges[0]])
    np.add.at(cnt, edges[1], 1)
    mask = cnt > 0
    prof[mask] /= cnt[mask, None]
    prof[mask] /= np.linalg.norm(prof[mask], axis=1, keepdims=True) + 1e-8
    logger.info(f"Venue MBTI profiles: {mask.sum():,}/{n_venues:,} covered")
    return prof.astype(np.float32)


def build_venue_meta(venue_id_map: dict, n_venues: int,
                     n_visits: np.ndarray) -> pd.DataFrame:
    """Venue metadata table aligned to the GNN venue index."""
    biz = pd.read_parquet(PROCESSED_DATA_DIR / "businesses.parquet")
    cols = [c for c in ("business_id", "name", "city", "state", "stars",
                        "categories") if c in biz.columns]
    biz = biz[cols].set_index("business_id")

    rows = {c: [None] * n_venues for c in biz.columns}
    ids = [None] * n_venues
    for vid, idx in venue_id_map.items():
        ids[idx] = vid
    meta = pd.DataFrame({"venue_id": ids})
    joined = meta.join(biz, on="venue_id")
    joined["n_visits"] = n_visits.astype(np.int32)
    return joined


def main():
    setup_logging(level=logging.INFO)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    user_id_map, venue_id_map = load_id_maps()
    n_users, n_venues = len(user_id_map), len(venue_id_map)
    logger.info(f"Index space: {n_users:,} users x {n_venues:,} venues")

    edges = load_all_edges(user_id_map, venue_id_map)

    # --- Popularity + per-entity visit counts ---------------------------
    venue_deg = np.zeros(n_venues, dtype=np.float32)
    np.add.at(venue_deg, edges[1], 1)
    np.save(OUT_DIR / "venue_log_degree.npy", np.log1p(venue_deg))

    user_deg = np.zeros(n_users, dtype=np.int32)
    np.add.at(user_deg, edges[0], 1)

    # --- Content (KNN) signal -------------------------------------------
    venue_pca = build_venue_pca64(venue_id_map, n_venues)
    np.save(OUT_DIR / "venue_bertopic_pca64.npy", venue_pca)
    np.save(OUT_DIR / "knn_user_profiles.npy",
            build_user_profiles(edges, venue_pca, n_users).astype(np.float16))

    # --- Personality (MBTI) signal ---------------------------------------
    user_mbti, user_traits, center = build_mbti_arrays(user_id_map, n_users)
    np.save(OUT_DIR / "mbti_center.npy", center.astype(np.float32))
    np.save(OUT_DIR / "user_traits.npy", user_traits.astype(np.float16))
    np.save(OUT_DIR / "venue_mbti_profiles.npy",
            build_venue_mbti(edges, user_mbti, n_venues).astype(np.float16))
    np.save(OUT_DIR / "user_mbti_centered.npy", user_mbti.astype(np.float16))

    # --- Visited venues as CSR (exclusion + display) ---------------------
    order = np.argsort(edges[0], kind="stable")
    sorted_users = edges[0][order]
    visits_venues = edges[1][order]
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.add.at(indptr, sorted_users + 1, 1)
    indptr = np.cumsum(indptr)
    np.save(OUT_DIR / "visits_indptr.npy", indptr)
    np.save(OUT_DIR / "visits_venues.npy", visits_venues)

    # --- Metadata tables --------------------------------------------------
    build_venue_meta(venue_id_map, n_venues, venue_deg).to_parquet(
        OUT_DIR / "venue_meta.parquet", index=False)

    uids = [None] * n_users
    for uid, idx in user_id_map.items():
        uids[idx] = uid
    pd.DataFrame({
        "user_id": uids,
        "n_visits": user_deg,
        "has_mbti": ~np.isnan(user_traits[:, 0]),
    }).to_parquet(OUT_DIR / "user_index.parquet", index=False)

    total_mb = sum(f.stat().st_size for f in OUT_DIR.iterdir()) / 1e6
    logger.info(f"Demo assets written to {OUT_DIR} ({total_mb:.0f} MB total)")


if __name__ == "__main__":
    main()
