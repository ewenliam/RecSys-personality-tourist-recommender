#!/usr/bin/env python3
"""
Measure whether the four RRF ranking signals are actually complementary.

For a sample of eligible users, compute the four full ranking scores exactly
as demo/app.py does (cosine similarity per signal; popularity = log-degree,
user-independent), then compute pairwise Spearman rank correlation over a
fixed random subsample of venues.

Outputs:
    results/signal_correlation.csv
    docs/thesis/figures/fig_signal_correlation.pdf
    docs/thesis/wut/tex/img/fig_signal_correlation.pdf
"""
import shutil
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).parent.parent
DEMO_DIR = PROJECT_ROOT / "models" / "demo"
GNN_DIR = PROJECT_ROOT / "models" / "gnn_hetero"
RESULTS_DIR = PROJECT_ROOT / "results"

N_USERS = 200
MIN_VISITS = 10
N_VENUE_SUBSAMPLE = 5000
SEED = 42

SIGNALS = ["content", "gnn", "mbti", "popularity"]
LABELS = {"content": "Content", "gnn": "Collaborative",
          "mbti": "Personality", "popularity": "Popularity"}


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Identical to demo/app.py cosine_scores."""
    d = min(query.shape[0], matrix.shape[1])
    q = query[:d].astype(np.float32)
    m = matrix[:, :d]
    dots = m @ q
    return dots / ((np.linalg.norm(q) + 1e-8)
                   * (np.linalg.norm(m, axis=1) + 1e-8))


def main() -> None:
    rng = np.random.default_rng(SEED)

    user_index = pd.read_parquet(DEMO_DIR / "user_index.parquet")
    n_venues_total = np.load(DEMO_DIR / "venue_log_degree.npy").shape[0]

    eligible = user_index.index[
        (user_index["n_visits"] >= MIN_VISITS) & user_index["has_mbti"]
    ].to_numpy()
    print(f"eligible users (n_visits>={MIN_VISITS}, has_mbti): {len(eligible)}")

    users = rng.choice(eligible, size=min(N_USERS, len(eligible)),
                       replace=False)
    venues = np.sort(rng.choice(n_venues_total, size=N_VENUE_SUBSAMPLE,
                                replace=False))
    print(f"sampled users: {len(users)}, venue subsample: {len(venues)} "
          f"of {n_venues_total}")

    # Venue-side matrices, restricted to the venue subsample.
    v_content = np.load(DEMO_DIR / "venue_bertopic_pca64.npy",
                        mmap_mode="r")[venues].astype(np.float32)
    v_mbti = np.load(DEMO_DIR / "venue_mbti_profiles.npy",
                     mmap_mode="r")[venues].astype(np.float32)
    v_gnn = np.load(GNN_DIR / "venue_embeddings.npy",
                    mmap_mode="r")[venues].astype(np.float32)
    v_pop = np.load(DEMO_DIR / "venue_log_degree.npy")[venues].astype(np.float32)

    u_content = np.load(DEMO_DIR / "knn_user_profiles.npy", mmap_mode="r")
    u_mbti = np.load(DEMO_DIR / "user_mbti_centered.npy", mmap_mode="r")
    u_gnn = np.load(GNN_DIR / "user_embeddings.npy", mmap_mode="r")

    pairs = list(combinations(SIGNALS, 2))
    per_pair = {p: [] for p in pairs}

    for n, uid in enumerate(users):
        scores = {
            "content": cosine_scores(
                np.asarray(u_content[uid], np.float32), v_content),
            "gnn": cosine_scores(
                np.asarray(u_gnn[uid], np.float32), v_gnn),
            "mbti": cosine_scores(
                np.asarray(u_mbti[uid], np.float32), v_mbti),
            "popularity": v_pop,
        }
        for a, b in pairs:
            rho = spearmanr(scores[a], scores[b]).statistic
            per_pair[(a, b)].append(float(rho))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(users)} users done")

    method_note = (
        f"Spearman over a fixed random subsample of {N_VENUE_SUBSAMPLE} of "
        f"{n_venues_total} venues (seed {SEED}); {len(users)} users sampled "
        f"from those with n_visits>={MIN_VISITS} and has_mbti; scores computed "
        f"exactly as demo/app.py (cosine per signal, popularity = log-degree)"
    )

    rows = []
    for a, b in pairs:
        vals = np.asarray(per_pair[(a, b)])
        rows.append({
            "pair": f"{a}-{b}",
            "mean_spearman": float(vals.mean()),
            "std_spearman": float(vals.std(ddof=1)),
            "n_users": len(vals),
            "method_note": method_note,
        })
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / "signal_correlation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print(df[["pair", "mean_spearman", "std_spearman", "n_users"]]
          .to_string(index=False))

    # ---------------------------------------------------------------- figure
    mat = np.eye(len(SIGNALS))
    for a, b in pairs:
        i, j = SIGNALS.index(a), SIGNALS.index(b)
        m = float(np.mean(per_pair[(a, b)]))
        mat[i, j] = mat[j, i] = m

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    names = [LABELS[s] for s in SIGNALS]
    ax.set_xticks(range(len(SIGNALS)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(SIGNALS)), names)
    for i in range(len(SIGNALS)):
        for j in range(len(SIGNALS)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=13,
                    color="white" if abs(mat[i, j]) > 0.5 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean Spearman rank correlation")
    fig.tight_layout()

    fig_dir = PROJECT_ROOT / "docs" / "thesis" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf = fig_dir / "fig_signal_correlation.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_signal_correlation.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf}")

    img_dir = PROJECT_ROOT / "docs" / "thesis" / "wut" / "tex" / "img"
    if img_dir.exists():
        shutil.copyfile(pdf, img_dir / "fig_signal_correlation.pdf")
        print(f"wrote {img_dir / 'fig_signal_correlation.pdf'}")
    else:
        print(f"WARNING: {img_dir} does not exist, skipped copy")


if __name__ == "__main__":
    main()
