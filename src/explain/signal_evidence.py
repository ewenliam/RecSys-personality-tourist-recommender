"""
EXACT explanation segments: quantities read directly out of the ranking
computation and the training data. Nothing here is an approximation - every
number can be re-derived from the artifacts by hand.
"""
from __future__ import annotations

import numpy as np

from src.planner.assets import PlannerAssets
from src.planner.candidates import cosine_scores
from src.planner.interfaces import Candidate, Persona

TRAIT_NAMES = [("E", "I"), ("S", "N"), ("T", "F"), ("J", "P")]


def trait_letters(traits: np.ndarray) -> str:
    return "".join(b if p >= 0.5 else a
                   for (a, b), p in zip(TRAIT_NAMES, traits))


def personality_evidence(m_vec: np.ndarray, traits: np.ndarray,
                         venue_idx: int, city_pool: np.ndarray,
                         assets: PlannerAssets) -> dict:
    """
    Decompose the personality signal for one venue:
    - cosine(persona m, venue visitor-mean profile) and its percentile
      among the city pool (exact: this IS the mbti sub-ranker's input);
    - the venue's mean visitor trait probabilities vs the persona's,
      reporting the axes where both lean the same way (exact, from data).
    """
    cos_all = cosine_scores(m_vec, assets.venue_mbti[city_pool])
    pos = int(np.where(city_pool == venue_idx)[0][0])
    percentile = float((cos_all < cos_all[pos]).mean() * 100)

    vt = assets.venue_trait_means[venue_idx]
    matches = []
    if not np.isnan(vt[0]):
        for (a, b), pv, pu in zip(TRAIT_NAMES, vt, traits):
            if (pv >= 0.5) == (pu >= 0.5):
                letter = b if pu >= 0.5 else a
                strength = pv if pv >= 0.5 else 1 - pv
                matches.append((letter, float(strength)))
    matches.sort(key=lambda x: -x[1])
    return {"cosine": float(cos_all[pos]), "city_percentile": percentile,
            "visitor_traits": None if np.isnan(vt[0]) else
            [float(x) for x in vt],
            "matching_axes": matches}


def covisit_evidence(persona: Persona, venue_idx: int,
                     assets: PlannerAssets) -> dict | None:
    """Users who visited BOTH a seed venue and this venue (training data)."""
    if not persona.seed_venues:
        return None
    visitors = set(assets.visitors_of(venue_idx).tolist())
    overlap = {}
    for s in persona.seed_venues:
        n = len(visitors.intersection(assets.visitors_of(s).tolist()))
        if n:
            overlap[s] = n
    return {"n_visitors": len(visitors),
            "covisitors_per_seed": overlap,
            "total_covisitors": sum(overlap.values())}


def rank_breakdown(cand: Candidate) -> dict:
    """The RRF arithmetic, verbatim - the fusion explanation IS its input."""
    return {"signal_ranks": cand.signal_ranks,
            "city_pool_size": cand.city_pool_size,
            "rrf_score": cand.rrf_score}
