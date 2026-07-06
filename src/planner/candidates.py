"""
Candidate generation: the existing RRF recommender, restricted to a city.

The scoring mathematics mirrors RRFRanker in scripts/evaluate_hybrid.py
(cosine per signal -> 1-indexed ranks -> sum 1/(k+rank) -> popularity
multiplier), but ranks are computed WITHIN the persona's city pool, which is
the meaningful reference set for a one-day itinerary. Per-signal ranks are
kept on every Candidate - they are the explanation layer's evidence.
"""
from __future__ import annotations

import numpy as np

from src.planner.assets import PlannerAssets
from src.planner.interfaces import Candidate, Persona

RRF_K = 60
POP_ALPHA = 0.01


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    d = min(query.shape[0], matrix.shape[1])
    q = query[:d].astype(np.float32)
    m = matrix[:, :d]
    return (m @ q) / ((np.linalg.norm(q) + 1e-8)
                      * (np.linalg.norm(m, axis=1) + 1e-8))


def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return ranks


def persona_queries(persona: Persona, assets: PlannerAssets,
                    m_vec: np.ndarray) -> dict[str, np.ndarray | None]:
    """
    Signal query vectors for a persona.

    mbti: always available (from the persona's own text, centered the same
    way as the precomputed users). content / collaborative: only when the
    persona has seed venues - content profile is the mean venue PCA vector
    of the seeds; the collaborative query is the mean GNN venue embedding of
    the seeds (item-item similarity in GNN space; a synthetic persona has no
    trained user embedding, and this proxy must be described as such).
    """
    queries: dict[str, np.ndarray | None] = {
        "mbti": m_vec, "content": None, "gnn": None}
    if persona.seed_venues:
        seeds = np.asarray(persona.seed_venues)
        queries["content"] = assets.venue_pca[seeds].mean(axis=0)
        queries["gnn"] = assets.venue_gnn[seeds].mean(axis=0)
    return queries


def candidate_set(persona: Persona, assets: PlannerAssets,
                  m_vec: np.ndarray, top_n: int = 50) -> list[Candidate]:
    pool = assets.city_venues(persona.city, require_hours=True)
    pool = pool[~np.isin(pool, persona.seed_venues)]
    if len(pool) == 0:
        raise ValueError(f"No venues with hours in city '{persona.city}'")

    venue_mats = {"mbti": assets.venue_mbti, "content": assets.venue_pca,
                  "gnn": assets.venue_gnn}
    rrf = np.zeros(len(pool), dtype=np.float64)
    per_ranks: dict[str, np.ndarray] = {}
    for name, q in persona_queries(persona, assets, m_vec).items():
        if q is None:
            continue
        ranks = scores_to_ranks(cosine_scores(q, venue_mats[name][pool]))
        per_ranks[name] = ranks
        rrf += 1.0 / (RRF_K + ranks)

    rrf *= 1.0 + POP_ALPHA * assets.venue_logdeg[pool]
    per_ranks["popularity"] = scores_to_ranks(assets.venue_logdeg[pool])

    # Deduplicate by venue name (franchises appear as many venues) so an
    # itinerary never visits the same brand twice; keep the best-scoring one.
    names = assets.venue_meta["name"].to_numpy()
    seen: set[str] = set()
    out: list[Candidate] = []
    for i in np.argsort(-rrf):
        name = str(names[pool[i]]).strip().lower()
        if name in seen:
            continue
        seen.add(name)
        out.append(Candidate(
            venue_idx=int(pool[i]),
            rrf_score=float(rrf[i]),
            signal_ranks={k: int(v[i]) for k, v in per_ranks.items()},
            city_pool_size=len(pool),
        ))
        if len(out) >= top_n:
            break
    return out
