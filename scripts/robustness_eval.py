#!/usr/bin/env python3
"""
Robustness check: run the hybrid evaluation across several random seeds and
report mean +/- std for every model and metric.

Under the default (--split gnn) the interaction split is fixed, loaded from
the graph encoder's own held-out edges, so the only thing a seed changes is
WHICH test users are sampled. The spread across seeds therefore isolates
sampling noise. Note this was NOT true of the older --split random protocol,
where set_seed() ran before the split was drawn and each seed silently
resampled the interaction split as well.

Usage:
    python scripts/robustness_eval.py
    python scripts/robustness_eval.py --seeds 42,123,7,2024,99
    python scripts/robustness_eval.py --split random   # legacy protocol
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SEED_DIR = RESULTS_DIR / "_robustness"

METRIC_COLS = ["Precision@K", "Recall@K", "NDCG@K", "MRR", "Hit Rate@K"]
MODEL_ORDER = [
    "Popularity", "GNN-only", "MBTI-only", "KNN-only", "KNN+pop (no GNN)",
    "RRF-Hybrid (no pop)", "Full minus GNN (KNN+MBTI+pop)",
    "RRF-Hybrid (pop=0.01)", "Full (KNN+GNN+MBTI+pop)",
]


def run_seed(seed: int, split: str = "gnn") -> pd.DataFrame:
    """Run evaluate_hybrid.py for one seed and load its metrics CSV."""
    out_path = SEED_DIR / f"seed_{seed}_{split}.csv"
    if out_path.exists():
        print(f"[seed {seed}] cached -> {out_path}")
        return pd.read_csv(out_path)

    print(f"[seed {seed}] running evaluate_hybrid.py ...")
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_hybrid.py"),
        "--seed", str(seed), "--out", str(out_path), "--split", split,
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"Seed {seed} run failed (exit {result.returncode})")
    return pd.read_csv(out_path)


def main(args):
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    SEED_DIR.mkdir(parents=True, exist_ok=True)

    frames = [run_seed(s, args.split) for s in seeds]
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(RESULTS_DIR / "phase4_robustness_per_seed.csv", index=False)

    # Aggregate mean / std across seeds for each (Model, K).
    agg = (
        raw.groupby(["Model", "K"])[METRIC_COLS]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["Model", "K"] + [
        f"{m}_{stat}" for m in METRIC_COLS for stat in ("mean", "std")
    ]
    # std is NaN when only one seed; report 0 spread in that case.
    for m in METRIC_COLS:
        agg[f"{m}_std"] = agg[f"{m}_std"].fillna(0.0)

    order = {name: i for i, name in enumerate(MODEL_ORDER)}
    agg["__o"] = agg["Model"].map(lambda m: order.get(m, 99))
    agg = agg.sort_values(["K", "__o"]).drop(columns="__o").reset_index(drop=True)
    agg.to_csv(RESULTS_DIR / "phase4_robustness_agg.csv", index=False)

    # Readable mean +/- std table at K=10.
    k = args.k
    sub = agg[agg["K"] == k]
    rows = []
    for _, r in sub.iterrows():
        row = {"Model": r["Model"]}
        for m in METRIC_COLS:
            row[m] = f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / f"phase4_robustness_k{k}.csv", index=False)

    print("\n" + "=" * 90)
    print(f"ROBUSTNESS: mean ± std over {len(seeds)} seeds {seeds} @ K={k}")
    print("=" * 90)
    print(table.to_string(index=False))

    # Best model per metric (by mean) + how often it wins across seeds.
    print(f"\nBest model per metric @ K={k} (by mean):")
    k_sub = agg[agg["K"] == k]
    raw_k = raw[raw["K"] == k]
    for m in METRIC_COLS:
        idx = k_sub[f"{m}_mean"].idxmax()
        best_model = k_sub.loc[idx, "Model"]
        mean_v = k_sub.loc[idx, f"{m}_mean"]
        std_v = k_sub.loc[idx, f"{m}_std"]
        # Win rate: in how many seeds is best_model the top scorer on m?
        wins = 0
        for s in seeds:
            seed_rows = raw_k[raw_k["seed"] == s]
            if not seed_rows.empty and seed_rows.loc[seed_rows[m].idxmax(), "Model"] == best_model:
                wins += 1
        print(f"  {m:13s}: {best_model:24s} {mean_v:.4f} ± {std_v:.4f} "
              f"(wins {wins}/{len(seeds)} seeds)")

    print(f"\nSaved: phase4_robustness_per_seed.csv, _agg.csv, _k{k}.csv")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed robustness check")
    parser.add_argument("--seeds", type=str, default="42,123,7,2024,99")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--split", type=str, default="gnn",
                        choices=["gnn", "random"],
                        help="Passed through to evaluate_hybrid.py. Cached "
                             "per-seed results are keyed by split, so the two "
                             "protocols never overwrite each other.")
    args = parser.parse_args()
    main(args)
