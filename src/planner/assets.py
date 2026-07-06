"""
Plain-Python loader for the precomputed serving assets in models/demo/.

Same artifacts the Streamlit demo uses, without any UI dependency, plus the
planner extras (geo/hours, visitor trait means, venue->users CSR) written by
scripts/build_planner_assets.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import MODEL_DIR

DEMO_DIR = MODEL_DIR / "demo"
GNN_DIR = MODEL_DIR / "gnn_hetero"


class PlannerAssets:
    """Lazy container over the demo + planner artifacts."""

    def __init__(self, demo_dir: Path = DEMO_DIR, gnn_dir: Path = GNN_DIR):
        self.venue_meta = pd.read_parquet(demo_dir / "venue_meta.parquet")
        self.venue_geo = pd.read_parquet(demo_dir / "venue_geo.parquet")
        self.venue_pca = np.load(demo_dir / "venue_bertopic_pca64.npy")
        self.venue_mbti = np.load(
            demo_dir / "venue_mbti_profiles.npy").astype(np.float32)
        self.venue_gnn = np.load(
            gnn_dir / "venue_embeddings.npy").astype(np.float32)
        self.venue_logdeg = np.load(demo_dir / "venue_log_degree.npy")
        self.mbti_center = np.load(demo_dir / "mbti_center.npy")
        self.venue_trait_means = np.load(demo_dir / "venue_trait_means.npy")
        self.vusers_indptr = np.load(demo_dir / "vusers_indptr.npy")
        self.vusers_users = np.load(demo_dir / "vusers_users.npy")
        self.n_venues = len(self.venue_meta)

    def city_venues(self, city: str, require_hours: bool = True
                    ) -> np.ndarray:
        """Venue indices in a city; optionally only those with known hours."""
        mask = (self.venue_meta["city"].str.lower() == city.lower()).to_numpy()
        if require_hours:
            mask &= self.venue_geo["has_hours"].to_numpy()
        return np.where(mask)[0]

    def visitors_of(self, venue_idx: int) -> np.ndarray:
        lo, hi = self.vusers_indptr[venue_idx], self.vusers_indptr[venue_idx + 1]
        return self.vusers_users[lo:hi]

    def city_centroid(self, city: str) -> tuple[float, float]:
        idx = self.city_venues(city, require_hours=False)
        return (float(self.venue_geo["lat"].iloc[idx].median()),
                float(self.venue_geo["lon"].iloc[idx].median()))
