#!/usr/bin/env python3
"""
Regenerate the deletion-test faithfulness figure from the real per-persona
results in results/faithfulness_eval.csv (15 personas). Blue bars are the mean
axis-probability drop under attribution-ordered deletion, grey bars under random
deletion. Matches the thesis figure style used in generate_extra_figures.py.

Outputs docs/thesis/figures/fig_faithfulness.pdf and copies it into
docs/thesis/wut/tex/img/.

Usage:
    python scripts/plot_faithfulness.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
CSV = ROOT / "results" / "faithfulness_eval.csv"
FIGDIR = ROOT / "docs" / "thesis" / "figures"
IMGDIR = ROOT / "docs" / "thesis" / "wut" / "tex" / "img"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 14, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 13,
    "legend.fontsize": 12, "lines.linewidth": 2.5, "axes.linewidth": 1.2,
})
BLUE, GRAY = "#1565c0", "#9e9e9e"

SHORT = {
    "introvert": "Introvert", "extravert": "Extravert", "practical": "Pragmatist",
    "enthusiast": "Enthusiast", "analyst": "Analyst", "organiser": "Warm org.",
    "dutiful": "Dutiful", "theorist": "Theorist", "thrillseeker": "Thrill-sk.",
    "dreamer": "Dreamer", "strategist": "Strategist", "spontaneous": "Spontaneous",
    "caretaker": "Caretaker", "planner": "Planner", "debater": "Debater",
}


def short_label(pid):
    parts = [p for p in pid.split("_") if p != "seeded"]
    for tok in parts[1:-1]:
        if tok in SHORT:
            return SHORT[tok]
    return parts[1].capitalize()


def main():
    df = pd.read_csv(CSV)
    labels = [short_label(p) for p in df["persona"]]
    x = np.arange(len(df))
    w = 0.4

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.bar(x - w / 2, df["aoc_attribution"], w, color=BLUE, label="Attributed words")
    ax.bar(x + w / 2, df["aoc_random"], w, color=GRAY, label="Random words")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean axis-probability drop")
    ax.legend(frameon=False)

    fig.tight_layout()
    out = FIGDIR / "fig_faithfulness.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    if IMGDIR.exists():
        shutil.copy(out, IMGDIR / "fig_faithfulness.pdf")
        print(f"copied to {IMGDIR / 'fig_faithfulness.pdf'}")
    print(f"personas: {len(df)} | mean gap: {df['gap'].mean():.4f}")


if __name__ == "__main__":
    main()
