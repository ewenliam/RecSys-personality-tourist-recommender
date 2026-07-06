"""
Assemble the per-stop explanation trace and render it as one sentence.

The renderer is deliberately template-based: every clause is filled from a
computed quantity carried on the trace, so the sentence can be audited
field-by-field against the JSON. No language model is involved.
"""
from __future__ import annotations

import numpy as np

from src.explain.signal_evidence import (
    covisit_evidence, personality_evidence, rank_breakdown, trait_letters,
    TRAIT_NAMES,
)
from src.planner.assets import PlannerAssets
from src.planner.interfaces import Itinerary, Persona, Stop

AXIS_OF_LETTER = {"E": "EI", "I": "EI", "S": "SN", "N": "SN",
                  "T": "TF", "F": "TF", "J": "JP", "P": "JP"}
LETTER_WORD = {"E": "Extraversion", "I": "Introversion", "S": "Sensing",
               "N": "Intuition", "T": "Thinking", "F": "Feeling",
               "J": "Judging", "P": "Perceiving"}
BINDING_TEXT = {
    "opening_window": "its opening hours bound this slot",
    "time_budget": "it was the last stop the day's time budget allowed",
    "travel_distance": "this position minimised added travel",
    "preference": "it was the highest-preference feasible choice",
}


def build_trace(persona: Persona, stop: Stop, m_vec: np.ndarray,
                traits: np.ndarray, city_pool: np.ndarray,
                assets: PlannerAssets,
                attributions: list[dict] | None) -> dict:
    v = stop.venue_idx
    pe = personality_evidence(m_vec, traits, v, city_pool, assets)
    trace = {
        "venue_idx": v,
        "venue_name": str(assets.venue_meta["name"].iloc[v]),
        "position": stop.position,
        "arrival": f"{stop.arrival_min // 60:02d}:{stop.arrival_min % 60:02d}",
        "trait_vector": {f"P({b})": float(p)
                         for (_, b), p in zip(TRAIT_NAMES, traits)},
        "predicted_type": trait_letters(traits),
        "personality_evidence": pe,
        "collaborative_evidence": covisit_evidence(persona, v, assets),
        "rank_breakdown": rank_breakdown(stop.candidate),
        "scheduling": {"binding": stop.binding,
                       "waited_min": stop.waited_min,
                       "travel_min_from_prev":
                           round(stop.travel_min_from_prev, 1)},
    }
    if attributions is not None:
        # Keep, per axis the persona actually leans to, the top CONTENT
        # words (stopwords and punctuation carry no explanatory value even
        # when their occlusion delta is large).
        stop = {"i", "a", "an", "the", "and", "or", "of", "in", "on", "to",
                "is", "it", "my", "me", "for", "with", "that", "this",
                "at", "be", "am", "do", "not", "no", "than", "as", "by"}
        dominant = trait_letters(traits)
        top_words = {}
        for letter in dominant:
            axis = AXIS_OF_LETTER[letter]
            sign = 1.0 if letter in "INFP" else -1.0
            ranked = sorted(attributions,
                            key=lambda a: -sign * a["deltas"][axis])
            words = []
            for a in ranked:
                w = a["word"].strip(".,!?;:'\"()-").lower()
                if len(w) >= 3 and w not in stop:
                    words.append(w)
                if len(words) == 3:
                    break
            top_words[letter] = words
        trace["token_evidence"] = top_words
    return trace


def _ordinal(n: int) -> str:
    n = min(int(n), 99)
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(
        n % 10 if n % 100 not in (11, 12, 13) else 0, "th")
    return f"{n}{suffix}"


SIGNAL_PHRASE = {"mbti": "personality match",
                 "content": "similarity to your visited venues",
                 "gnn": "co-visitation patterns",
                 "popularity": "overall popularity"}


def render_sentence(trace: dict) -> str:
    rb = trace["rank_breakdown"]["signal_ranks"]
    pool = trace["rank_breakdown"]["city_pool_size"]
    pe = trace["personality_evidence"]
    axes = pe["matching_axes"]

    # Lead with the signal that actually won this venue its place: the one
    # with the best (lowest) within-city rank. Anything else misrepresents
    # the fusion arithmetic.
    dominant = min(rb, key=rb.get)
    lead = (f"it ranked {_ordinal(rb[dominant])} of {pool} venues on "
            f"{SIGNAL_PHRASE[dominant]}")

    person_clause = ""
    if pe["city_percentile"] >= 50 and axes:
        letter, strength = axes[0]
        person_clause = (
            f"; visitors here lean {LETTER_WORD[letter]} ({strength:.0%}), "
            f"matching your profile "
            f"({_ordinal(pe['city_percentile'])}-percentile personality fit)")
        if "token_evidence" in trace:
            words = trace["token_evidence"].get(letter)
            if words:
                person_clause += (" - your words "
                                  + ", ".join(f"'{w}'" for w in words)
                                  + " signalled that trait")

    cov = trace["collaborative_evidence"]
    cov_clause = ""
    if cov and cov["total_covisitors"]:
        cov_clause = (f"; {cov['total_covisitors']} visitors of your seed "
                      f"venues also went here")

    others = ", ".join(f"{k} {v}/{pool}" for k, v in rb.items()
                       if k != dominant)

    sched = trace["scheduling"]
    sched_clause = BINDING_TEXT[sched["binding"]]
    if sched["waited_min"] > 0:
        sched_clause += f" (waited {sched['waited_min']} min for opening)"

    return (f"Stop {trace['position'] + 1} - {trace['venue_name']} "
            f"(arrive {trace['arrival']}): chosen because {lead}"
            f"{person_clause}{cov_clause} (other signals: {others}); "
            f"scheduled here because {sched_clause}.")


def explain_itinerary(persona: Persona, itinerary: Itinerary,
                      m_vec: np.ndarray, traits: np.ndarray,
                      assets: PlannerAssets,
                      attributions: list[dict] | None) -> list[dict]:
    city_pool = assets.city_venues(persona.city, require_hours=True)
    traces = []
    for stop in itinerary.stops:
        t = build_trace(persona, stop, m_vec, traits, city_pool, assets,
                        attributions)
        t["sentence"] = render_sentence(t)
        traces.append(t)
    return traces
