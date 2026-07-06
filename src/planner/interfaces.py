"""
Data contracts between the existing pipeline, the planning layer, and the
explanation layer. Every quantity an explanation may cite is carried on
these objects from the moment it is computed - the explainer never
re-derives anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Persona:
    """A synthetic user: personality text + city + optional seed venues."""
    persona_id: str
    text: str                       # personality inferred from this
    city: str
    seed_venues: list[int] = field(default_factory=list)  # venue indices
    day: str = "sat"                # mon..sun
    start_min: int = 10 * 60        # day starts 10:00
    end_min: int = 21 * 60          # ends 21:00
    dwell_min: int = 90             # time spent per stop
    start_latlon: tuple[float, float] | None = None  # None -> city centroid


@dataclass
class Candidate:
    """One venue in a persona's ranked candidate set (within-city ranks)."""
    venue_idx: int
    rrf_score: float                # fused score incl. popularity multiplier
    signal_ranks: dict[str, int]    # e.g. {"mbti": 3, "content": 17, ...}
    city_pool_size: int             # venues ranked against


@dataclass
class Stop:
    """One scheduled stop, with solver-recorded scheduling evidence."""
    venue_idx: int
    position: int                   # 0-based order in the itinerary
    arrival_min: int
    departure_min: int
    travel_min_from_prev: float
    waited_min: int                 # arrival clamped to opening time
    binding: str                    # opening_window | time_budget |
                                    # travel_distance | preference
    candidate: Candidate | None = None


@dataclass
class Itinerary:
    persona_id: str
    day: str
    stops: list[Stop]
    total_score: float              # sum of rrf_score over stops
    total_travel_min: float
    rejected: list[dict]            # {venue_idx, reason} rejection log
