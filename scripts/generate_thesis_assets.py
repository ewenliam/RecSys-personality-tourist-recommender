#!/usr/bin/env python3
"""
Generate thesis-ready LaTeX tables and figures from the final results.

Reads results/phase4_robustness_agg.csv (the 5-seed aggregated recommender
metrics) and combines it with the final MBTI-classifier numbers and dataset
statistics (constants below, from the reproduced runs).

Outputs:
    docs/thesis/tables.tex          - booktabs LaTeX tables
    docs/thesis/figures/*.pdf/.png  - vector + raster figures

Usage:
    python scripts/generate_thesis_assets.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results"
OUT = PROJECT_ROOT / "docs" / "thesis"
FIGDIR = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- Final constants from the reproduced runs --------------------------
DATASET = {
    "Yelp users": "235,643", "Yelp venues": "85,857",
    "Train interactions": "2,831,185", "Test interactions": "707,796",
    "MBTI users (Kaggle)": "8,675", "MBTI posts": "411,294",
    "MBTI train/val/test users": "6,940 / 867 / 868",
}
# Per-dimension balanced accuracy: before (leaky, raw) vs after (user-disjoint).
CLF_DIMS = ["E/I", "S/N", "T/F", "J/P"]
CLF_BEFORE_BAL = [0.579, 0.585, 0.629, 0.565]      # old, leaky split
CLF_AFTER_BAL = [0.696, 0.782, 0.836, 0.693]       # new, user-disjoint
CLF_AFTER_ACC = [0.826, 0.813, 0.834, 0.730]       # per-user accuracy
CLF_SUMMARY = {"mean_acc": 0.801, "mean_bal": 0.752, "exact16": 0.445}

PRETTY = {
    "Popularity": "Popularity",
    "GNN-only": "GNN only",
    "MBTI-only": "MBTI only",
    "KNN-only": "Content (KNN) only",
    "KNN+pop (no GNN)": "Content + popularity",
    "RRF-Hybrid (no pop)": "Hybrid KNN+GNN (no pop.)",
    "RRF-Hybrid (pop=0.01)": "Hybrid KNN+GNN (+pop.)",
    "Full (KNN+GNN+MBTI+pop)": "Full (all four signals)",
}
ORDER = list(PRETTY.keys())
METRICS = ["Precision@K", "Recall@K", "NDCG@K", "MRR", "Hit Rate@K"]


def load_agg():
    df = pd.read_csv(RESULTS / "phase4_robustness_agg.csv")
    df["__o"] = df["Model"].map(lambda m: ORDER.index(m) if m in ORDER else 99)
    return df.sort_values(["K", "__o"]).drop(columns="__o")


def fmt(m, s):
    return f"{m:.4f} $\\pm$ {s:.4f}"


def latex_main_table(agg, k=10):
    sub = agg[agg["K"] == k]
    lines = [
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{Recommendation performance at $K={k}$ "
        "(mean $\\pm$ std over 5 seeds). Best per column in \\textbf{bold}.}",
        f"\\label{{tab:main-results-k{k}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}", "\\toprule",
        "Model & Precision@" + str(k) + " & Recall@" + str(k)
        + " & NDCG@" + str(k) + " & MRR & Hit Rate@" + str(k) + " \\\\",
        "\\midrule",
    ]
    best = {m: sub[f"{m}_mean"].max() for m in METRICS}
    for _, r in sub.iterrows():
        cells = []
        for m in METRICS:
            txt = fmt(r[f"{m}_mean"], r[f"{m}_std"])
            if abs(r[f"{m}_mean"] - best[m]) < 1e-9:
                txt = "\\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(PRETTY[r["Model"]] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_classifier_table():
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{MBTI classifier per-dimension balanced accuracy before "
        "(post-level split, leaky) and after (user-disjoint split, "
        "class-weighted, per-user). Only T/F was genuinely learned before; "
        "all four axes are learned after.}",
        "\\label{tab:classifier}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "Dimension & Bal. acc.\\ (before) & Bal. acc.\\ (after) "
        "& Per-user acc.\\ (after) \\\\", "\\midrule",
    ]
    for d, b, a, acc in zip(CLF_DIMS, CLF_BEFORE_BAL, CLF_AFTER_BAL, CLF_AFTER_ACC):
        lines.append(f"{d} & {b:.3f} & {a:.3f} & {acc:.3f} \\\\")
    lines += [
        "\\midrule",
        f"\\textbf{{Mean}} & {np.mean(CLF_BEFORE_BAL):.3f} & "
        f"{CLF_SUMMARY['mean_bal']:.3f} & {CLF_SUMMARY['mean_acc']:.3f} \\\\",
        "\\bottomrule", "\\end{tabular}",
        "\\\\[2pt]\\footnotesize Exact 16-type per-user accuracy (after): "
        f"{CLF_SUMMARY['exact16']:.3f} (random $=0.0625$).",
        "\\end{table}", "",
    ]
    return "\n".join(lines)


def latex_dataset_table():
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Dataset statistics.}", "\\label{tab:dataset}",
        "\\begin{tabular}{lr}", "\\toprule", "Quantity & Value \\\\", "\\midrule",
    ]
    for k, v in DATASET.items():
        lines.append(f"{k} & {v} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def latex_methodology_table():
    rows = [
        ("MBTI classifier", "BERT, 16-class", "BERT, 4 binary heads"),
        ("MBTI evaluation", "Upsample before split (\\textasciitilde94\\%, leaked)",
         "User-disjoint + class weights (80.1\\%/75.2\\%, honest)"),
        ("Topic embeddings", "Generic sentence-transformer",
         "BERT-MBTI CLS (personality-informed)"),
        ("Graph model", "None", "Heterogeneous GNN (GraphSAGE + BPR)"),
        ("Personality in ranking", "Feature in XGBoost",
         "Direct ranker (visitor-mean profiles)"),
        ("Final ranker", "XGBoost", "Parameter-free RRF (4 signals)"),
        ("Ranking metrics", "Not reported", "Full ablation, 5 seeds"),
    ]
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Methodology comparison with the prior approach "
        "(Omer, 2024).}", "\\label{tab:methodology}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lll}", "\\toprule",
        "Aspect & Omer (2024) & This thesis \\\\", "\\midrule",
    ]
    for a, o, c in rows:
        lines.append(f"{a} & {o} & {c} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


# ---- Figures -----------------------------------------------------------
def fig_classifier(agg=None):
    x = np.arange(len(CLF_DIMS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(x - w/2, CLF_BEFORE_BAL, w, label="Before (leaky split)", color="#b4b2a9")
    ax.bar(x + w/2, CLF_AFTER_BAL, w, label="After (honest, fixed)", color="#534AB7")
    ax.axhline(0.5, ls="--", lw=0.8, color="#888780", label="Random (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(CLF_DIMS)
    ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title("MBTI per-dimension balanced accuracy: before vs after")
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_classifier_before_after.{ext}", dpi=200)
    plt.close(fig)


def fig_recommender(agg, k=10):
    sub = agg[agg["K"] == k].set_index("Model").reindex(ORDER)
    labels = [PRETTY[m] for m in ORDER]
    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(8, 4))
    hr = sub["Hit Rate@K_mean"].values
    hr_e = sub["Hit Rate@K_std"].values
    colors = ["#b4b2a9"] * (len(ORDER) - 1) + ["#534AB7"]
    ax.bar(x, hr, yerr=hr_e, capsize=3, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"Hit Rate@{k}")
    ax.set_title(f"Hit Rate@{k} by model (mean $\\pm$ std, 5 seeds)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_recommender_hitrate.{ext}", dpi=200)
    plt.close(fig)


def fig_ablation(agg, k=10):
    """Contribution of each signal: build up from popularity to full."""
    chain = ["Popularity", "KNN-only", "RRF-Hybrid (no pop)",
             "RRF-Hybrid (pop=0.01)", "Full (KNN+GNN+MBTI+pop)"]
    chain_lbl = ["Popularity", "+Content", "+GNN", "+Popularity", "+MBTI (Full)"]
    sub = agg[agg["K"] == k].set_index("Model")
    ndcg = [sub.loc[m, "NDCG@K_mean"] for m in chain]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(range(len(chain)), ndcg, "-o", color="#1D9E75", lw=2)
    ax.set_xticks(range(len(chain))); ax.set_xticklabels(chain_lbl, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"NDCG@{k}")
    ax.set_title("Incremental contribution of each signal")
    for i, v in enumerate(ndcg):
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_ablation_ndcg.{ext}", dpi=200)
    plt.close(fig)


def main():
    agg = load_agg()

    tex = "\n".join([
        "% Auto-generated by scripts/generate_thesis_assets.py - do not edit by hand.",
        "% Requires \\usepackage{booktabs,graphicx} in the preamble.", "",
        latex_dataset_table(),
        latex_classifier_table(),
        latex_methodology_table(),
        latex_main_table(agg, k=10),
        latex_main_table(agg, k=5),
        latex_main_table(agg, k=20),
    ])
    (OUT / "tables.tex").write_text(tex, encoding="utf-8")
    print(f"Wrote {OUT/'tables.tex'}")

    fig_classifier()
    fig_recommender(agg, k=10)
    fig_ablation(agg, k=10)
    print(f"Wrote figures to {FIGDIR}/ (pdf + png):")
    for p in sorted(FIGDIR.glob("*.pdf")):
        print("  ", p.name)
    print("Done.")


if __name__ == "__main__":
    main()
