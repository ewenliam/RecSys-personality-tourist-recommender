"""
Orienteering-style itinerary construction: select and order a subset of the
candidate venues to maximise cumulative preference score subject to opening
hours, travel time, and the day's time budget.

Solver: greedy best-insertion by descending RRF score with feasibility
checks, followed by 2-opt order improvement. Deliberately transparent
rather than optimal: every rejection and every accepted stop's binding
constraint is recorded AT DECISION TIME, which is what makes the
scheduling part of the explanation trace exact rather than reconstructed.
"""
from __future__ import annotations

import math

import numpy as np

from src.planner.assets import PlannerAssets
from src.planner.interfaces import Candidate, Itinerary, Persona, Stop

SPEED_KMH = 25.0  # simple urban travel-speed assumption for the PoC


def haversine_min(lat1, lon1, lat2, lon2) -> float:
    """Travel time in minutes between two points at SPEED_KMH."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    km = 2 * r * math.asin(math.sqrt(a))
    return km / SPEED_KMH * 60.0


class _Schedule:
    """Simulates a visit order; reports feasibility and the failing reason."""

    def __init__(self, persona: Persona, assets: PlannerAssets,
                 start_latlon: tuple[float, float]):
        self.p = persona
        self.geo = assets.venue_geo
        self.day = persona.day
        self.start_latlon = start_latlon

    def window(self, v: int) -> tuple[int, int]:
        return (int(self.geo[f"{self.day}_open"].iloc[v]),
                int(self.geo[f"{self.day}_close"].iloc[v]))

    def latlon(self, v: int) -> tuple[float, float]:
        return (float(self.geo["lat"].iloc[v]),
                float(self.geo["lon"].iloc[v]))

    def simulate(self, order: list[int]):
        """
        Walk the order; return (stops, total_travel, None) if feasible or
        (None, 0, (venue_idx, reason)) at the first violation.
        """
        t = self.p.start_min
        here = self.start_latlon
        stops, total_travel = [], 0.0
        for pos, v in enumerate(order):
            open_m, close_m = self.window(v)
            if open_m < 0:
                return None, 0.0, (v, "opening_window")
            lat, lon = self.latlon(v)
            travel = haversine_min(here[0], here[1], lat, lon)
            arrive = t + travel
            waited = max(0.0, open_m - arrive)
            arrive = max(arrive, float(open_m))
            depart = arrive + self.p.dwell_min
            if depart > close_m:
                return None, 0.0, (v, "opening_window")
            if depart > self.p.end_min:
                return None, 0.0, (v, "time_budget")
            stops.append(Stop(
                venue_idx=v, position=pos, arrival_min=int(arrive),
                departure_min=int(depart), travel_min_from_prev=travel,
                waited_min=int(waited), binding="preference"))
            t, here, total_travel = depart, (lat, lon), total_travel + travel
        return stops, total_travel, None


def plan_itinerary(persona: Persona, candidates: list[Candidate],
                   assets: PlannerAssets) -> Itinerary:
    start = persona.start_latlon or assets.city_centroid(persona.city)
    sched = _Schedule(persona, assets, start)
    by_idx = {c.venue_idx: c for c in candidates}

    order: list[int] = []
    rejected: list[dict] = []
    insertion_evidence: dict[int, str] = {}

    # Greedy best-insertion in descending preference order.
    for cand in sorted(candidates, key=lambda c: -c.rrf_score):
        best = None            # (total_travel, position, stops)
        window_blocked = time_blocked = 0
        for pos in range(len(order) + 1):
            trial = order[:pos] + [cand.venue_idx] + order[pos:]
            stops, travel, fail = sched.simulate(trial)
            if stops is None:
                if fail[1] == "opening_window":
                    window_blocked += 1
                else:
                    time_blocked += 1
                continue
            if best is None or travel < best[0]:
                best = (travel, pos, stops)
        if best is None:
            reason = ("time_budget" if time_blocked >= window_blocked
                      else "opening_window")
            rejected.append({"venue_idx": cand.venue_idx, "reason": reason})
            continue
        order.insert(best[1], cand.venue_idx)
        # Evidence for WHY it sits where it sits: if other positions were
        # window-blocked the window bound the position; if only one position
        # was ever feasible under the budget, the budget bound it; otherwise
        # the min-travel criterion chose it.
        if window_blocked > 0:
            insertion_evidence[cand.venue_idx] = "opening_window"
        elif time_blocked > 0:
            insertion_evidence[cand.venue_idx] = "time_budget"
        elif len(order) > 1:
            insertion_evidence[cand.venue_idx] = "travel_distance"
        else:
            insertion_evidence[cand.venue_idx] = "preference"

    # 2-opt improvement on travel time, keeping feasibility.
    improved = True
    while improved and len(order) > 3:
        improved = False
        base_stops, base_travel, _ = sched.simulate(order)
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                trial = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                stops, travel, fail = sched.simulate(trial)
                if stops is not None and travel < base_travel - 1e-6:
                    order, base_travel, improved = trial, travel, True
        if improved:
            continue

    stops, total_travel, _ = sched.simulate(order)
    for s in stops:
        s.binding = insertion_evidence.get(s.venue_idx, "preference")
        if s.waited_min > 0:      # waiting for opening is definitive evidence
            s.binding = "opening_window"
        s.candidate = by_idx[s.venue_idx]

    return Itinerary(
        persona_id=persona.persona_id, day=persona.day, stops=stops,
        total_score=sum(s.candidate.rrf_score for s in stops),
        total_travel_min=total_travel, rejected=rejected)
