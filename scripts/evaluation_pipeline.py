#!/usr/bin/env python3
"""
Grand Evaluation Pipeline: Unified Comparison of All Models.

Loads results from all model pipelines and produces a single
comparative table with HR@10, NDCG@10, MRR, and ECE across:

  - Traditional baselines (LogReg, DT, RF, SVM, KNN)
  - XGBoost V1 (temperature calibration)
  - XGBoost V2 (Platt/Isotonic calibration)
  - GNN-only (LTGNN link prediction)

Also generates LaTeX-formatted tables for direct thesis inclusion.

Usage:
    python scripts/evaluation_pipeline.py
    python scripts/evaluation_pipeline.py --run-all
    python scripts/evaluation_pipeline.py --latex
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import PROJECT_ROOT, MODEL_DIR, PROCESSED_DATA_DIR
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def load_traditional_baselines() -> pd.DataFrame:
    """Load traditional baseline results."""
    path = RESULTS_DIR / "traditional_baselines.csv"
    if not path.exists():
        logger.warning(f"Traditional baselines not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded traditional baselines: {len(df)} rows")
    return df


def load_xgboost_v2_results() -> pd.DataFrame:
    """Load XGBoost V2 comparison results."""
    path = RESULTS_DIR / "xgboost_v2_comparison.csv"
    if not path.exists():
        logger.warning(f"XGBoost V2 results not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded XGBoost V2 results: {len(df)} rows")
    return df


def load_gnn_results() -> pd.DataFrame:
    """Load GNN training results if available."""
    # Check for GNN evaluation metrics saved during training
    for subdir in ["gnn_hetero", "gnn"]:
        path = MODEL_DIR / subdir / "eval_metrics.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["Model"] = f"LTGNN ({subdir})"
            logger.info(f"Loaded GNN results from {subdir}")
            return df

    # Try to reconstruct from saved embeddings
    logger.warning("No GNN eval metrics found. Will skip GNN row.")
    return pd.DataFrame()


def load_bertopic_v2_summary() -> pd.DataFrame:
    """Load BERTopic V2 summary for reporting."""
    path = RESULTS_DIR / "bertopic_v2_summary.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded BERTopic V2 summary")
    return df


def build_grand_table(k: int = 10) -> pd.DataFrame:
    """
    Build the grand comparison table at a specific K.

    Columns: Model | HR@K | NDCG@K | MRR | ECE
    """
    rows = []

    # Traditional baselines
    trad = load_traditional_baselines()
    if not trad.empty:
        trad_k = trad[trad["K"] == k]
        for _, row in trad_k.iterrows():
            rows.append({
                "Model": row["Model"],
                f"HR@{k}": row.get("Hit Rate@K", 0),
                f"NDCG@{k}": row.get("NDCG@K", 0),
                "MRR": row.get("MRR", 0),
                "ECE": row.get("ECE", 0),
                "Category": "Traditional",
            })

    # XGBoost V1/V2
    xgb = load_xgboost_v2_results()
    if not xgb.empty:
        xgb_k = xgb[xgb["K"] == k]
        for _, row in xgb_k.iterrows():
            rows.append({
                "Model": f"XGBoost {row['Model']}",
                f"HR@{k}": row.get("Hit Rate@K", 0),
                f"NDCG@{k}": row.get("NDCG@K", 0),
                "MRR": row.get("MRR", 0),
                "ECE": row.get("ECE", 0),
                "Category": "Hybrid",
            })

    # GNN
    gnn = load_gnn_results()
    if not gnn.empty and "K" in gnn.columns:
        gnn_k = gnn[gnn["K"] == k]
        for _, row in gnn_k.iterrows():
            rows.append({
                "Model": row.get("Model", "LTGNN"),
                f"HR@{k}": row.get("Hit Rate@K", row.get("hit_rate", 0)),
                f"NDCG@{k}": row.get("NDCG@K", row.get("ndcg", 0)),
                "MRR": row.get("MRR", row.get("mrr", 0)),
                "ECE": row.get("ECE", 0),
                "Category": "GNN",
            })

    if not rows:
        logger.error("No results found. Run individual pipelines first.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort by NDCG descending (best model first)
    ndcg_col = f"NDCG@{k}"
    df = df.sort_values(ndcg_col, ascending=False).reset_index(drop=True)

    return df


def generate_latex_table(df: pd.DataFrame, k: int = 10, caption: str = None) -> str:
    """
    Generate a LaTeX table from the grand comparison DataFrame.

    Bolds the best value in each metric column.
    """
    if df.empty:
        return "% No data available"

    metric_cols = [c for c in df.columns if c not in ("Model", "Category")]

    # Find best values for bolding
    best = {}
    for col in metric_cols:
        if col == "ECE":
            best[col] = df[col].min()  # Lower is better
        else:
            best[col] = df[col].max()  # Higher is better

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")

    if caption is None:
        caption = f"Grand Comparison of Recommendation Models (K={k})"
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{tab:grand_comparison}")

    # Column format
    col_fmt = "l" + "r" * len(metric_cols)
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    # Header
    header = "Model"
    for col in metric_cols:
        header += f" & {col}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    prev_category = None
    for _, row in df.iterrows():
        cat = row.get("Category", "")
        if cat != prev_category and prev_category is not None:
            lines.append(r"\midrule")
        prev_category = cat

        line = row["Model"].replace("_", r"\_")
        for col in metric_cols:
            val = row[col]
            formatted = f"{val:.4f}"
            if abs(val - best[col]) < 1e-6:
                formatted = r"\textbf{" + formatted + "}"
            line += f" & {formatted}"
        line += r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_ablation_table(k: int = 10) -> pd.DataFrame:
    """
    Generate ablation study table showing component contributions.

    Compares:
      - MBTI Only (LogReg baseline)
      - BERTopic Only (no personality)
      - MBTI + BERTopic (traditional baseline)
      - MBTI + BERTopic + GNN (hybrid V1)
      - MBTI + BERTopic + GNN + Calibration (hybrid V2)
    """
    trad = load_traditional_baselines()
    xgb = load_xgboost_v2_results()

    rows = []

    # Get LogReg as "MBTI + BERTopic" baseline
    if not trad.empty:
        lr = trad[(trad["Model"] == "Logistic Regression") & (trad["K"] == k)]
        if not lr.empty:
            r = lr.iloc[0]
            rows.append({
                "Components": "MBTI + BERTopic",
                f"HR@{k}": r.get("Hit Rate@K", 0),
                f"NDCG@{k}": r.get("NDCG@K", 0),
                "MRR": r.get("MRR", 0),
                "ECE": r.get("ECE", 0),
            })

    # XGBoost with GNN
    if not xgb.empty:
        for _, r in xgb[xgb["K"] == k].iterrows():
            model_name = r["Model"]
            if "V1" in model_name:
                components = "+ GNN (temperature)"
            else:
                components = "+ GNN + Calibration"
            rows.append({
                "Components": components,
                f"HR@{k}": r.get("Hit Rate@K", 0),
                f"NDCG@{k}": r.get("NDCG@K", 0),
                "MRR": r.get("MRR", 0),
                "ECE": r.get("ECE", 0),
            })

    return pd.DataFrame(rows)


def run_pipeline(script_name: str) -> bool:
    """Run a pipeline script and return success status."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    logger.info(f"Running {script_name}...")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        logger.error(f"{script_name} failed:\n{result.stderr[-500:]}")
        return False

    logger.info(f"{script_name} completed successfully")
    return True


def main(args):
    """Run the grand evaluation pipeline."""
    setup_logging(level=logging.INFO)
    k = args.k

    # Optionally run all sub-pipelines
    if args.run_all:
        logger.info("=" * 60)
        logger.info("Running all sub-pipelines")
        logger.info("=" * 60)

        pipelines = [
            "traditional_baselines.py",
            "xgboost_ranker_v2.py",
        ]
        for script in pipelines:
            if not run_pipeline(script):
                logger.warning(f"Skipping failed pipeline: {script}")

    # Build grand comparison
    logger.info("=" * 60)
    logger.info(f"Building Grand Comparison Table (K={k})")
    logger.info("=" * 60)

    grand = build_grand_table(k=k)

    if grand.empty:
        logger.error(
            "No results available. Run individual pipelines first:\n"
            "  python scripts/traditional_baselines.py\n"
            "  python scripts/xgboost_ranker_v2.py\n"
        )
        return

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grand.to_csv(RESULTS_DIR / "grand_comparison.csv", index=False)

    # Print
    print("\n" + "=" * 80)
    print(f"GRAND COMPARISON (K={k})")
    print("=" * 80)
    display_cols = [c for c in grand.columns if c != "Category"]
    print(grand[display_cols].to_string(index=False))

    # Ablation study
    ablation = generate_ablation_table(k=k)
    if not ablation.empty:
        ablation.to_csv(RESULTS_DIR / "ablation_study.csv", index=False)
        print("\n" + "=" * 80)
        print("ABLATION STUDY")
        print("=" * 80)
        print(ablation.to_string(index=False))

    # BERTopic V2 summary
    bt_summary = load_bertopic_v2_summary()
    if not bt_summary.empty:
        print("\n" + "=" * 80)
        print("BERTOPIC V2 SUMMARY")
        print("=" * 80)
        print(bt_summary.to_string(index=False))

    # LaTeX output
    if args.latex:
        latex = generate_latex_table(grand, k=k)
        latex_path = RESULTS_DIR / "grand_comparison.tex"
        latex_path.write_text(latex, encoding="utf-8")
        print(f"\nLaTeX table saved to {latex_path}")

        if not ablation.empty:
            ablation_latex = generate_latex_table(
                ablation, k=k,
                caption=f"Ablation Study: Component Contributions (K={k})",
            )
            ablation_tex_path = RESULTS_DIR / "ablation_study.tex"
            ablation_tex_path.write_text(ablation_latex, encoding="utf-8")
            print(f"Ablation LaTeX saved to {ablation_tex_path}")

    # Feature importance from XGBoost V2
    fi_path = RESULTS_DIR / "xgboost_v2_feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        print("\n" + "=" * 80)
        print("XGBOOST V2 FEATURE IMPORTANCE (Top 15)")
        print("=" * 80)
        print(fi.head(15).to_string(index=False))

    logger.info(f"\nAll results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grand Evaluation Pipeline")
    parser.add_argument("--k", type=int, default=10,
                        help="K value for metrics (default: 10)")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all sub-pipelines before comparison")
    parser.add_argument("--latex", action="store_true", default=True,
                        help="Generate LaTeX tables")
    parser.add_argument("--no-latex", dest="latex", action="store_false")

    args = parser.parse_args()
    main(args)
