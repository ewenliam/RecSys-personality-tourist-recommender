#!/usr/bin/env python3
"""
Planning-quality evaluation: planner vs. no-planning baseline.

Baseline = the same top-N ranked candidates executed in rank order (the
"unordered list" a plain recommender gives you), simulated under the same
travel/dwell model but WITHOUT feasibility checks - each stop is visited in
sequence regardless of opening hours, stopping when the time budget runs
out. We then count how many of its visited stops violate an opening window.

Metrics per persona: feasibility rate (fraction of visited stops that are
actually open for the whole visit), number of stops, cumulative preference
(RRF) score of FEASIBLE stops.

Usage:
    python scripts/eval_planning.py [--device cuda]
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.planner.assets import PlannerAssets
from src.planner.candidates import candidate_set
from src.planner.orienteering import plan_itinerary, haversine_min
from src.explain.text_attrib import MBTIExplainer
from src.utils.helpers import setup_logging
from scripts.run_itineraries import PERSONAS, SEEDED_PERSONA_IDS, assign_seeds

logger = logging.getLogger(__name__)


def baseline_walk(persona, candidates, assets):
    """Visit candidates in rank order until the budget ends; no checks."""
    geo = assets.venue_geo
    here = assets.city_centroid(persona.city)
    t = persona.start_min
    visited, feasible_score, n_feasible = 0, 0.0, 0
    for c in candidates:
        lat = float(geo["lat"].iloc[c.venue_idx])
        lon = float(geo["lon"].iloc[c.venue_idx])
        arrive = t + haversine_min(here[0], here[1], lat, lon)
        depart = arrive + persona.dwell_min
        if depart > persona.end_min:
            break
        visited += 1
        open_m = int(geo[f"{persona.day}_open"].iloc[c.venue_idx])
        close_m = int(geo[f"{persona.day}_close"].iloc[c.venue_idx])
        if open_m >= 0 and arrive >= open_m and depart <= close_m:
            n_feasible += 1
            feasible_score += c.rrf_score
        t, here = depart, (lat, lon)
    return visited, n_feasible, feasible_score


def main(args):
    setup_logging(level=logging.INFO)
    assets = PlannerAssets()
    explainer = MBTIExplainer(device=args.device)

    rows = []
    for persona in PERSONAS:
        if persona.persona_id in SEEDED_PERSONA_IDS:
            assign_seeds(persona, assets)
        m_vec, _ = explainer.embed(persona.text, assets.mbti_center)
        cands = candidate_set(persona, assets, m_vec, top_n=50)

        it = plan_itinerary(persona, cands, assets)
        b_visited, b_feasible, b_score = baseline_walk(persona, cands, assets)

        rows.append({
            "persona": persona.persona_id,
            "planner_stops": len(it.stops),
            "planner_feasible_rate": 1.0,     # feasible by construction
            "planner_score": round(it.total_score, 4),
            "baseline_stops": b_visited,
            "baseline_feasible_rate": round(b_feasible / max(b_visited, 1), 3),
            "baseline_feasible_score": round(b_score, 4),
        })

    df = pd.DataFrame(rows)
    out = Path("results") / "planning_eval.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nmean feasible-stop score: planner "
          f"{df['planner_score'].mean():.4f} vs baseline "
          f"{df['baseline_feasible_score'].mean():.4f}; baseline mean "
          f"feasibility {df['baseline_feasible_rate'].mean():.1%}")
    logger.info(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
