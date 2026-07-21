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
    # --- expansion wave: p6..p15, added to widen the evaluation sample ---
    Persona(
        persona_id="p6_warm_organiser_neworleans",
        city="New Orleans",
        text=("I am the one who books the table and makes sure nobody is "
              "left out. I remember what my friends like and I choose "
              "places where they will feel looked after. Warm staff matter "
              "more to me than a clever menu. I set the plan a week ahead, "
              "send everyone the times, and I feel genuinely happy when the "
              "whole group is together and comfortable."),
        day="fri", start_min=11 * 60, end_min=21 * 60),
    Persona(
        persona_id="p7_seeded_dutiful_edmonton",
        city="Edmonton",
        text=("I go to the same three places and I see no reason to change "
              "that. I check the opening hours before I leave, I arrive "
              "early, and I keep the receipt. What I want is clean tables, "
              "the order correct, and the price the same as last month. I "
              "do not need surprises, I need somewhere dependable that does "
              "the ordinary things properly."),
        seed_venues=[], day="tue", start_min=9 * 60, end_min=18 * 60),
    Persona(
        persona_id="p8_theorist_saintlouis",
        city="Saint Louis",
        text=("What interests me is the underlying system. I will happily "
              "spend an hour working out why one roaster tastes different "
              "from another, or how a building was put together. I would "
              "rather sit alone with a notebook than make conversation. My "
              "plans stay loose because a better question usually turns up "
              "halfway through the day."),
        day="wed", start_min=11 * 60, end_min=20 * 60),
    Persona(
        persona_id="p9_thrillseeker_reno",
        city="Reno",
        text=("I want action right now. I walk in, size the place up in ten "
              "seconds, and if it is dull I am already outside looking for "
              "the next one. Give me a busy floor, something physical to "
              "do, a bit of risk and a lot of noise. I never plan ahead, I "
              "just move and things happen."),
        day="sat", start_min=12 * 60, end_min=22 * 60),
    Persona(
        persona_id="p10_seeded_dreamer_santabarbara",
        city="Santa Barbara",
        text=("Some places just feel right to me and I cannot fully explain "
              "why. I like a small shop that somebody clearly built with "
              "love, a garden where I can sit and think about what matters "
              "to me. I care that a place has integrity. I drift rather "
              "than schedule, and the day usually turns into something I "
              "did not expect."),
        seed_venues=[], day="sun", start_min=10 * 60, end_min=19 * 60),
    Persona(
        persona_id="p11_strategist_boise",
        city="Boise",
        text=("I treat a day out like a project. I decide the objective, "
              "order the stops so nothing is wasted, and I expect the "
              "places I pick to deliver. Inefficiency irritates me: slow "
              "service, vague answers, a queue that nobody is managing. I "
              "make the call quickly, I hold people to it, and I move on to "
              "the next thing."),
        day="thu", start_min=9 * 60, end_min=20 * 60),
    Persona(
        persona_id="p12_spontaneous_clearwater",
        city="Clearwater",
        text=("Life is too short to sit indoors! I love being out with "
              "people, good food, music playing, sun on the water. I decide "
              "everything last minute and it always works out. I am not one "
              "for heavy conversations about the future, I just want today "
              "to feel good and for everyone around me to be enjoying it "
              "too."),
        day="sat", start_min=11 * 60, end_min=22 * 60),
    Persona(
        persona_id="p13_seeded_caretaker_metairie",
        city="Metairie",
        text=("I usually plan around my family. I know which places are "
              "quiet enough for my mother, which ones have staff who are "
              "patient, and I go back to those. I do not need anything new "
              "or impressive. I would rather be somewhere familiar where "
              "people are kind and I know exactly what to expect."),
        seed_venues=[], day="sun", start_min=10 * 60, end_min=18 * 60),
    Persona(
        persona_id="p14_planner_saintpetersburg",
        city="Saint Petersburg",
        text=("I research before I go anywhere. I read what the place is "
              "trying to do, decide whether it is actually good at it, and "
              "build the day around the two or three that pass. I have "
              "little patience for crowds or for chatting to strangers. A "
              "long visit to one serious collection beats six shallow "
              "stops."),
        day="mon", start_min=10 * 60, end_min=19 * 60),
    Persona(
        persona_id="p15_debater_wilmington",
        city="Wilmington",
        text=("I love arguing about ideas with people who push back. Take "
              "me somewhere with an odd concept and I will pull it apart "
              "over a drink and enjoy every minute. I start five plans and "
              "finish two, I get restless with routine, and the best days "
              "are the ones that go sideways into a conversation I did not "
              "see coming."),
        day="fri", start_min=12 * 60, end_min=22 * 60),
]

# Personas that receive a synthetic visit history. p4 and p5 are the
# original seeded pair (unchanged); p7, p10 and p13 join them in the
# expansion wave.
SEEDED_PERSONA_IDS = frozenset({
    "p4_seeded_enthusiast_indianapolis",
    "p5_seeded_analyst_nashville",
    "p7_seeded_dutiful_edmonton",
    "p10_seeded_dreamer_santabarbara",
    "p13_seeded_caretaker_metairie",
})


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
        if persona.persona_id in SEEDED_PERSONA_IDS:
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
