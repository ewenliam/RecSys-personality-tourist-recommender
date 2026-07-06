#!/usr/bin/env python3
"""
Precompute planner/explainer assets, aligned to the demo venue index space.

Requires models/demo/ (run scripts/build_demo_assets.py first). Adds:

    venue_geo.parquet        lat, lon, mon_open..sun_close (minutes since
                             midnight; -1 = closed/unknown), has_hours
    venue_trait_means.npy    [n_v, 4] mean visitor trait probs P(I),P(N),
                             P(F),P(P) (NaN where no visitor has traits)
    vusers_indptr.npy /      CSR venue -> visiting users (transpose of the
    vusers_users.npy         demo's user -> venues CSR), for co-visitation
                             evidence in explanations

Usage:
    python scripts/build_planner_assets.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import PROCESSED_DATA_DIR, MODEL_DIR
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)
DEMO_DIR = MODEL_DIR / "demo"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday"]
DAY_COLS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def parse_window(spec: str) -> tuple[int, int]:
    """'8:0-22:30' -> (480, 1350) minutes; overnight close clamps to 1439."""
    try:
        o, c = spec.split("-")
        oh, om = (int(x) for x in o.split(":"))
        ch, cm = (int(x) for x in c.split(":"))
        open_m, close_m = oh * 60 + om, ch * 60 + cm
        if close_m <= open_m:          # overnight or zero-length: clamp
            close_m = 24 * 60 - 1
        return open_m, close_m
    except (ValueError, AttributeError):
        return -1, -1


def build_geo(venue_meta: pd.DataFrame) -> pd.DataFrame:
    biz = pd.read_parquet(
        PROCESSED_DATA_DIR / "businesses.parquet",
        columns=["business_id", "latitude", "longitude", "hours"],
    ).set_index("business_id")
    joined = venue_meta[["venue_id"]].join(biz, on="venue_id")

    out = {"lat": joined["latitude"].to_numpy(np.float64),
           "lon": joined["longitude"].to_numpy(np.float64)}
    opens = {d: np.full(len(joined), -1, np.int32) for d in DAY_COLS}
    closes = {d: np.full(len(joined), -1, np.int32) for d in DAY_COLS}
    n_hours = 0
    for i, h in enumerate(joined["hours"].to_numpy()):
        if not isinstance(h, dict):
            continue
        n_hours += 1
        for day, col in zip(DAYS, DAY_COLS):
            if day in h and h[day]:
                opens[col][i], closes[col][i] = parse_window(h[day])
    for d in DAY_COLS:
        out[f"{d}_open"], out[f"{d}_close"] = opens[d], closes[d]
    out["has_hours"] = np.array(
        [isinstance(h, dict) for h in joined["hours"].to_numpy()])
    logger.info(f"Geo: {len(joined)} venues, {n_hours} with opening hours "
                f"({n_hours/len(joined):.0%})")
    return pd.DataFrame(out)


def build_trait_means(n_venues: int) -> np.ndarray:
    traits = np.load(DEMO_DIR / "user_traits.npy").astype(np.float32)
    indptr = np.load(DEMO_DIR / "visits_indptr.npy")
    venues = np.load(DEMO_DIR / "visits_venues.npy")
    users = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64),
                      np.diff(indptr))

    has = ~np.isnan(traits[:, 0])
    mask = has[users]
    u, v = users[mask], venues[mask]
    sums = np.zeros((n_venues, 4), np.float64)
    cnt = np.zeros(n_venues, np.int64)
    np.add.at(sums, v, traits[u])
    np.add.at(cnt, v, 1)
    means = np.full((n_venues, 4), np.nan, np.float32)
    nz = cnt > 0
    means[nz] = (sums[nz] / cnt[nz, None]).astype(np.float32)
    logger.info(f"Trait means: {nz.sum():,}/{n_venues:,} venues covered")
    return means, users, venues


def build_venue_users_csr(users: np.ndarray, venues: np.ndarray,
                          n_venues: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(venues, kind="stable")
    v_sorted, u_sorted = venues[order], users[order]
    indptr = np.zeros(n_venues + 1, np.int64)
    np.add.at(indptr, v_sorted + 1, 1)
    return np.cumsum(indptr), u_sorted.astype(np.int32)


def main():
    setup_logging(level=logging.INFO)
    venue_meta = pd.read_parquet(DEMO_DIR / "venue_meta.parquet")
    n_venues = len(venue_meta)

    build_geo(venue_meta).to_parquet(DEMO_DIR / "venue_geo.parquet",
                                     index=False)

    means, users, venues = build_trait_means(n_venues)
    np.save(DEMO_DIR / "venue_trait_means.npy", means)

    # Full (untyped) venue->users CSR for co-visitation evidence.
    all_users = np.repeat(
        np.arange(len(np.load(DEMO_DIR / "visits_indptr.npy")) - 1,
                  dtype=np.int64),
        np.diff(np.load(DEMO_DIR / "visits_indptr.npy")))
    all_venues = np.load(DEMO_DIR / "visits_venues.npy")
    indptr, vusers = build_venue_users_csr(all_users, all_venues, n_venues)
    np.save(DEMO_DIR / "vusers_indptr.npy", indptr)
    np.save(DEMO_DIR / "vusers_users.npy", vusers)
    logger.info("Planner assets written to models/demo/")


if __name__ == "__main__":
    main()
