#!/usr/bin/env python3
"""
Generate additional thesis figures (dataset, classifier, robustness) from the
REAL data and result files on disk, plus print the dataset statistics used in
the "Dataset and Data Pre-processing" chapter.

Outputs land in docs/thesis/figures/ (pdf) and are copied by the caller into
docs/thesis/wut/tex/img/.

Usage:
    python scripts/generate_extra_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS = PROJECT_ROOT / "results"
FIGDIR = PROJECT_ROOT / "docs" / "thesis" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 14, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "lines.linewidth": 2.5, "lines.markersize": 8,
    "axes.linewidth": 1.2,
})
BLUE, ORANGE, GRAY = "#1565c0", "#e65100", "#9e9e9e"
GOOD, BAD = "#2e7d32", "#c62828"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


# ---------------------------------------------------------------- MBTI data --
def mbti_figures():
    df = pd.read_csv(DATA_DIR / "raw" / "mbti_1.csv")
    df["type"] = df["type"].str.upper()
    counts = df["type"].value_counts()

    # 16-type distribution
    fig, ax = plt.subplots(figsize=(7, 3))
    counts.plot.bar(ax=ax, color=BLUE)
    ax.set_ylabel("Users")
    ax.set_xlabel("MBTI type")
    save(fig, "fig_mbti_type_dist")

    # Per-axis balance
    axes_def = [("E", "I", 0), ("S", "N", 1), ("T", "F", 2), ("J", "P", 3)]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    labels, first_frac = [], []
    for a, b, pos in axes_def:
        frac_a = (df["type"].str[pos] == a).mean()
        labels.append(f"{a}/{b}")
        first_frac.append(frac_a)
    y = np.arange(len(labels))
    ax.barh(y, first_frac, color=BLUE, label="first letter")
    ax.barh(y, [1 - f for f in first_frac], left=first_frac, color=ORANGE,
            label="second letter")
    for i, f in enumerate(first_frac):
        ax.text(f / 2, i, f"{labels[i][0]} {f:.0%}", va="center",
                ha="center", color="white", fontsize=9)
        ax.text(f + (1 - f) / 2, i, f"{labels[i][2]} {1 - f:.0%}", va="center",
                ha="center", color="white", fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of users")
    ax.axvline(0.5, color=GRAY, ls="--", lw=1)
    save(fig, "fig_mbti_axis_balance")

    # Posts per user and words per post
    n_posts = df["posts"].str.split(r"\|\|\|").str.len()
    words = df["posts"].str.split(r"\|\|\|").explode().str.split().str.len()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.8))
    ax1.hist(n_posts, bins=30, color=BLUE)
    ax1.set_xlabel("Posts per user")
    ax1.set_ylabel("Users")
    ax2.hist(words.clip(upper=80), bins=40, color=BLUE)
    ax2.set_xlabel("Words per post (clipped at 80)")
    ax2.set_ylabel("Posts")
    save(fig, "fig_mbti_post_stats")

    print("MBTI stats:")
    print(f"  users={len(df)}, posts={int(n_posts.sum()):,}, "
          f"median posts/user={n_posts.median():.0f}")
    print(f"  words/post: median={words.median():.0f}, mean={words.mean():.1f}")
    for a, b, pos in axes_def:
        print(f"  {a}/{b}: {a}={100*(df['type'].str[pos]==a).mean():.1f}%")
    print(f"  most/least common type: {counts.index[0]} ({counts.iloc[0]}), "
          f"{counts.index[-1]} ({counts.iloc[-1]})")


# ---------------------------------------------------------------- Yelp data --
def yelp_figures():
    # Interactions per user (all splits, user_id column only).
    parts = []
    for split in ("train_reviews.parquet", "val_reviews.parquet",
                  "test_reviews.parquet"):
        p = DATA_DIR / "processed" / split
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["user_id"]))
    per_user = pd.concat(parts, ignore_index=True)["user_id"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(per_user.clip(upper=100), bins=60, color=BLUE, log=True)
    ax.set_xlabel("Reviews per user (clipped at 100)")
    ax.set_ylabel("Users (log scale)")
    save(fig, "fig_user_activity")
    print("Yelp interaction stats:")
    print(f"  users={len(per_user):,}, interactions={int(per_user.sum()):,}")
    print(f"  reviews/user: median={per_user.median():.0f}, "
          f"mean={per_user.mean():.1f}, max={per_user.max()}")
    print(f"  users with <=5 reviews: {(per_user<=5).mean():.1%}")

    # Venue stars + top categories.
    biz = pd.read_parquet(DATA_DIR / "processed" / "businesses.parquet")
    fig, ax = plt.subplots(figsize=(5, 2.8))
    biz["stars"].value_counts().sort_index().plot.bar(ax=ax, color=BLUE)
    ax.set_xlabel("Average star rating")
    ax.set_ylabel("Venues")
    save(fig, "fig_venue_stars")

    cats = (biz["categories"].dropna().str.split(", ").explode()
            .value_counts().head(20))
    fig, ax = plt.subplots(figsize=(6, 4))
    cats.iloc[::-1].plot.barh(ax=ax, color=BLUE)
    ax.set_xlabel("Venues")
    save(fig, "fig_top_categories")
    print(f"  venues={len(biz):,}; top categories: "
          f"{', '.join(cats.index[:5])}")
    if "city" in biz.columns:
        print(f"  cities={biz['city'].nunique():,}; top: "
              f"{', '.join(biz['city'].value_counts().index[:5])}")


# ------------------------------------------------------------- result plots --
def result_figures():
    # Canonical laptop training curve (per-post val accuracy per epoch, from
    # the recorded canonical run).
    val_acc = [0.6554, 0.6803, 0.6788, 0.6926]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.plot(range(1, 5), val_acc, "o-", color=BLUE)
    ax.set_xticks(range(1, 5))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (per post)")
    save(fig, "fig_training_curve")

    # Backbone ablation with bert seed spread. The canonical configuration
    # (bert-base/128) is green; the tested alternatives are neutral grey.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = ["bert-base\n128", "roberta-base\n256", "deberta-v3\n128",
             "deberta-v3\n256"]
    vals = [0.752, 0.740, 0.743, 0.761]
    ax.axhspan(0.742, 0.772, color=GRAY, alpha=0.25,
               label="bert-base seed spread (3 seeds)")
    ax.bar(names, vals, color=[GOOD, GRAY, GRAY, GRAY], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=12)
    ax.set_ylim(0.70, 0.79)
    ax.set_xlabel("Backbone / context length (tokens)")
    ax.set_ylabel("Balanced accuracy (per user)")
    ax.legend(loc="upper left")
    save(fig, "fig_backbone_ablation")

    # K sensitivity + per-seed variance from the robustness CSVs.
    agg_path = RESULTS / "phase4_robustness_agg.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        models = {"Popularity": BAD, "KNN-only": BLUE,
                  "Full (KNN+GNN+MBTI+pop)": GOOD}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for m, c in models.items():
            sub = agg[agg["Model"] == m].sort_values("K")
            label = "Full hybrid" if m.startswith("Full") else (
                "Content only" if m == "KNN-only" else m)
            ax1.errorbar(sub["K"], sub["Hit Rate@K_mean"],
                         yerr=sub["Hit Rate@K_std"], fmt="o-", color=c,
                         label=label, capsize=3)
            ax2.errorbar(sub["K"], sub["NDCG@K_mean"],
                         yerr=sub["NDCG@K_std"], fmt="o-", color=c,
                         label=label, capsize=3)
        for ax, name in ((ax1, "Hit Rate@K"), (ax2, "NDCG@K")):
            ax.set_xticks([5, 10, 20])
            ax.set_xlabel("K")
            ax.set_ylabel(name)
        ax1.legend(fontsize=8)
        save(fig, "fig_k_sensitivity")

    per_seed_path = RESULTS / "phase4_robustness_per_seed.csv"
    if per_seed_path.exists():
        raw = pd.read_csv(per_seed_path)
        k10 = raw[raw["K"] == 10]
        order = ["Popularity", "GNN-only", "MBTI-only", "KNN-only",
                 "RRF-Hybrid (pop=0.01)", "Full (KNN+GNN+MBTI+pop)"]
        pretty = ["Popularity", "GNN only", "MBTI only", "Content only",
                  "Hybrid (no MBTI)", "Full hybrid"]
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        for i, m in enumerate(order):
            vals = k10[k10["Model"] == m]["Hit Rate@K"]
            ax.scatter([i] * len(vals), vals, color="#455a64", alpha=0.7,
                       s=55, label="single seed" if i == 0 else None)
            ax.scatter([i], [vals.mean()], color=GOOD, marker="_",
                       s=700, lw=3.5, label="mean of 5 seeds" if i == 0 else None)
        ax.set_xticks(range(len(order)), pretty, rotation=20, ha="right")
        ax.set_ylabel("Hit Rate@10")
        ax.legend(loc="upper left")
        save(fig, "fig_seed_variance")


if __name__ == "__main__":
    print("== MBTI dataset ==")
    mbti_figures()
    print("== Yelp dataset ==")
    yelp_figures()
    print("== Result figures ==")
    result_figures()
    print("Done.")
