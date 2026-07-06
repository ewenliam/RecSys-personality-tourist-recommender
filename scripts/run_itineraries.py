#!/usr/bin/env python3
"""
Generate explained single-day itineraries for synthetic personas.

For each persona (MBTI-flavoured text x city): infer personality, build the
within-city candidate set, solve the orienteering problem, and emit one
auditable explanation trace per stop (JSON) plus a rendered sentence.

Usage:
    python scripts/run_itineraries.py [--device cuda] [--no-attributions]
"""
import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.planner.assets import PlannerAssets
from src.planner.candidates import candidate_set
from src.planner.interfaces import Persona
from src.planner.orienteering import plan_itinerary
from src.explain.text_attrib import MBTIExplainer
from src.explain.trace import explain_itinerary
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)
OUT_DIR = Path(__file__).parent.parent / "results" / "itineraries"

PERSONAS = [
    Persona(
        persona_id="p1_introvert_philadelphia",
        city="Philadelphia",
        text=("I avoid crowded and loud places whenever I can. A quiet "
              "coffee shop with a bookshelf, a small gallery, or a calm "
              "park bench is my perfect afternoon. Big parties drain me "
              "and I prefer deep one-on-one conversations. I always plan "
              "my day carefully in advance and read reviews before "
              "going anywhere new.")),
    Persona(
        persona_id="p2_extravert_tampa",
        city="Tampa",
        text=("Honestly the louder the better! I love packed sports bars, "
              "live music, dancing with strangers and huge group dinners "
              "where everyone talks at once. I decide where to go on the "
              "spot, whatever feels fun in the moment, and I get bored "
              "sitting still for long. Bring the energy and the crowd!")),
    Persona(
        persona_id="p3_practical_tucson",
        city="Tucson",
        text=("I like things that work: a reliable diner with fair prices, "
              "a hardware store that has what I need, a barber who is on "
              "time. I do not care for fancy concepts or trends. I keep a "
              "routine, stick to a schedule, and value places with "
              "consistent, no-nonsense service and concrete details.")),
    Persona(
        persona_id="p4_seeded_enthusiast_indianapolis",
        city="Indianapolis",
        text=("There is nothing better than discovering something new - a "
              "quirky mural, a pop-up food stall, a stranger with a great "
              "story! I bounce between ideas and plans, follow my "
              "curiosity wherever it leads, and love places bursting with "
              "creativity, people and possibility."),
        seed_venues=[]),   # filled below from the city pool
    Persona(
        persona_id="p5_seeded_analyst_nashville",
        city="Nashville",
        text=("I enjoy quiet places where I can think - a bookshop, a "
              "coffee house with good espresso and no music, a museum on "
              "a weekday morning. I like understanding how things work "
              "more than small talk, and I would rather explore an idea "
              "in depth than rush between attractions."),
        seed_venues=[]),   # filled below
]


def assign_seeds(persona: Persona, assets: PlannerAssets, k: int = 5,
                 seed: int = 42) -> None:
    """Give a seeded persona k 'already visited' venues from its city
    (popularity-weighted sample: plausible visit history, reproducible)."""
    pool = assets.city_venues(persona.city, require_hours=True)
    weights = np.expm1(assets.venue_logdeg[pool])
    rng = np.random.default_rng(seed)
    persona.seed_venues = [int(v) for v in rng.choice(
        pool, size=k, replace=False, p=weights / weights.sum())]


def main(args):
    setup_logging(level=logging.INFO)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    assets = PlannerAssets()
    explainer = MBTIExplainer(device=args.device)

    for persona in PERSONAS:
        if persona.persona_id.startswith(("p4", "p5")):
            assign_seeds(persona, assets)
        logger.info(f"=== {persona.persona_id} ({persona.city}, "
                    f"{len(persona.seed_venues)} seeds) ===")

        m_vec, traits = explainer.embed(persona.text, assets.mbti_center)
        candidates = candidate_set(persona, assets, m_vec, top_n=args.top_n)
        itinerary = plan_itinerary(persona, candidates, assets)

        attributions = None
        if not args.no_attributions:
            attributions = explainer.occlusion(persona.text)

        traces = explain_itinerary(persona, itinerary, m_vec, traits,
                                   assets, attributions)

        record = {
            "persona": dataclasses.asdict(persona),
            "predicted_traits": {d: float(p) for d, p in
                                 zip(explainer.dims, traits)},
            "itinerary": {
                "total_score": itinerary.total_score,
                "total_travel_min": round(itinerary.total_travel_min, 1),
                "n_stops": len(itinerary.stops),
                "n_rejected": len(itinerary.rejected),
                "rejected": itinerary.rejected,
            },
            "stops": traces,
        }
        out = OUT_DIR / f"{persona.persona_id}.json"
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info(f"wrote {out.name}: {len(itinerary.stops)} stops, "
                    f"score={itinerary.total_score:.4f}, "
                    f"travel={itinerary.total_travel_min:.0f} min")
        for t in traces:
            print("  " + t["sentence"])

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--no-attributions", action="store_true")
    main(parser.parse_args())
