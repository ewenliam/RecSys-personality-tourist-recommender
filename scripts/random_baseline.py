#!/usr/bin/env python3
"""
Analytic random-recommender baseline for the hybrid evaluation setup.

Reproduces the 80/20 edge split of scripts/evaluate_hybrid.py (seed 42, same
GNN id mappings, same processed review files), reads the actual relevant-set
sizes |R_u|, and computes the expected metrics of a uniformly random ranker:

    E[HitRate@K] = 1 - C(N-|R|, K) / C(N, K)
    E[Precision@K] = |R| / N
    E[Recall@K]    = K / N

Output: results/random_baseline.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import PROCESSED_DATA_DIR, MODEL_DIR, PROJECT_ROOT as PR

K = 10
SEED = 42
NUM_EVAL_USERS = 500


def load_thesis_numbers() -> dict:
    """Read the canonical 5-seed full-hybrid K=10 row, no hardcoding."""
    agg = pd.read_csv(PROJECT_ROOT / "results" / "phase4_robustness_agg.csv")
    row = agg[(agg["Model"] == "Full (KNN+GNN+MBTI+pop)") & (agg["K"] == K)]
    assert len(row) == 1, f"expected 1 row, got {len(row)}"
    row = row.iloc[0]
    return {
        "hit_rate": float(row["Hit Rate@K_mean"]),
        "precision": float(row["Precision@K_mean"]),
        "recall": float(row["Recall@K_mean"]),
    }


def expected_hit_rate(r: int, n: int, k: int) -> float:
    """1 - C(n-r, k)/C(n, k), computed as a stable product."""
    if r <= 0:
        return 0.0
    miss = 1.0
    for i in range(k):
        num = n - r - i
        if num <= 0:
            return 1.0
        miss *= num / (n - i)
    return 1.0 - miss


def main() -> None:
    import torch

    THESIS = load_thesis_numbers()
    print(f"thesis full-hybrid K={K}: {THESIS}")

    mappings = torch.load(MODEL_DIR / "gnn_hetero" / "id_mappings.pt",
                          map_location="cpu")
    user_id_map = mappings["user_id_map"]
    venue_id_map = mappings["venue_id_map"]
    n_users, n_venues = len(user_id_map), len(venue_id_map)
    print(f"n_users={n_users}  n_venues={n_venues}")

    dfs = []
    for split in ["train_reviews.parquet", "val_reviews.parquet",
                  "test_reviews.parquet"]:
        p = PROCESSED_DATA_DIR / split
        if p.exists():
            dfs.append(pd.read_parquet(p, columns=["user_id", "business_id"]))
    interactions = pd.concat(dfs, ignore_index=True)
    print(f"interactions loaded: {len(interactions)}")

    valid = interactions[
        interactions["user_id"].isin(user_id_map)
        & interactions["business_id"].isin(venue_id_map)
    ]
    edges = np.stack([
        valid["user_id"].map(user_id_map).to_numpy(),
        valid["business_id"].map(venue_id_map).to_numpy(),
    ])
    print(f"valid mapped edges: {edges.shape[1]}")

    np.random.seed(SEED)
    n = edges.shape[1]
    perm = np.random.permutation(n)
    n_test = int(n * 0.2)
    test_edges = edges[:, perm[:n_test]]
    print(f"train edges: {n - n_test}  test edges: {n_test}")

    user_gt: dict[int, set] = {}
    for u, v in zip(test_edges[0], test_edges[1]):
        user_gt.setdefault(int(u), set()).add(int(v))
    print(f"users with >=1 test edge: {len(user_gt)}")

    eval_users = list(user_gt.keys())
    sampled = list(np.random.choice(eval_users, NUM_EVAL_USERS, replace=False))

    rows = []
    for label, users in [("all_test_users", eval_users),
                         (f"sampled_{NUM_EVAL_USERS}_users", sampled)]:
        sizes = np.array([len(user_gt[u]) for u in users], dtype=np.int64)
        hr = np.array([expected_hit_rate(int(r), n_venues, K) for r in sizes])
        prec = sizes / n_venues
        rec = np.full(len(sizes), K / n_venues, dtype=float)
        rows.append({
            "user_set": label,
            "n_users": len(users),
            "mean_relevant_set_size": float(sizes.mean()),
            "median_relevant_set_size": float(np.median(sizes)),
            "max_relevant_set_size": int(sizes.max()),
            "n_venues": n_venues,
            "K": K,
            "random_hit_rate_at_k": float(hr.mean()),
            "random_precision_at_k": float(prec.mean()),
            "random_recall_at_k": float(rec.mean()),
            "thesis_hit_rate_at_k": THESIS["hit_rate"],
            "thesis_precision_at_k": THESIS["precision"],
            "thesis_recall_at_k": THESIS["recall"],
            "hit_rate_multiplier": THESIS["hit_rate"] / float(hr.mean()),
            "precision_multiplier": THESIS["precision"] / float(prec.mean()),
            "recall_multiplier": THESIS["recall"] / float(rec.mean()),
        })

    df = pd.DataFrame(rows)
    out = PR / "results" / "random_baseline.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.T.to_string())


if __name__ == "__main__":
    main()
