#!/usr/bin/env python3
"""
Phase 1: Methodology Correction - Leaky vs Robust MBTI Training.

Demonstrates the data leakage problem in Omer Amac's thesis:
  - "Leaky":  Upsample entire dataset THEN split -> inflated ~94% accuracy
  - "Robust": Split first THEN upsample training set only -> honest baseline

Usage:
    python scripts/compare_methodology.py
    python scripts/compare_methodology.py --epochs 5 --quick
"""
import argparse
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import (
    BERTConfig,
    get_config,
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
    PROJECT_ROOT,
)
from src.data.preprocessor import ReviewDataset, DataCollator
from src.data.sampler import DataSampler
from src.models.bert_mbti.model import MBTIMultiLabelClassifier
from src.models.bert_mbti.trainer import MBTITrainer, create_data_loaders, TrainingMetrics
from src.utils.helpers import setup_logging, set_seed, get_device

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"


def load_mbti_data(data_dir: Path) -> pd.DataFrame:
    """
    Load MBTI-labeled data for comparison.

    Returns a single DataFrame with 'clean_text' and 'mbti' columns.
    """
    train_path = data_dir / "train_reviews.parquet"
    if train_path.exists():
        df = pd.read_parquet(train_path)
        # Combine all splits to simulate the original full dataset
        for split in ["val_reviews.parquet", "test_reviews.parquet"]:
            split_path = data_dir / split
            if split_path.exists():
                df = pd.concat([df, pd.read_parquet(split_path)], ignore_index=True)
    else:
        raise FileNotFoundError(
            f"Processed data not found: {train_path}\n"
            "Run 'python scripts/prepare_data.py' first."
        )

    # Ensure MBTI labels exist
    if "mbti" not in df.columns:
        logger.warning(
            "MBTI labels not found. Generating synthetic labels for demonstration."
        )
        np.random.seed(42)
        mbti_types = [
            "INTJ", "INTP", "ENTJ", "ENTP",
            "INFJ", "INFP", "ENFJ", "ENFP",
            "ISTJ", "ISFJ", "ESTJ", "ESFJ",
            "ISTP", "ISFP", "ESTP", "ESFP",
        ]
        # Create intentionally imbalanced distribution (mimicking Kaggle MBTI dataset)
        weights = [0.11, 0.16, 0.03, 0.08, 0.15, 0.21, 0.02, 0.07,
                   0.02, 0.02, 0.01, 0.01, 0.03, 0.02, 0.02, 0.04]
        df["mbti"] = np.random.choice(mbti_types, size=len(df), p=weights)

    text_col = "clean_text" if "clean_text" in df.columns else "text"
    return df[[text_col, "mbti"]].rename(columns={text_col: "clean_text"})


def run_leaky_pipeline(
    full_df: pd.DataFrame,
    config: BERTConfig,
    device: torch.device,
    sampler: DataSampler,
) -> dict:
    """
    Replicated Leaky Methodology (Omer):
    Upsample the full dataset FIRST, then split into train/val/test.

    This causes data leakage: upsampled copies of minority-class samples
    appear in BOTH training and test sets, inflating accuracy.
    """
    logger.info("=" * 60)
    logger.info("LEAKY PIPELINE: Upsample -> Split (Omer's approach)")
    logger.info("=" * 60)

    # Step 1: Upsample the ENTIRE dataset (the flaw)
    upsampled_df = sampler.upsample_minority_classes(full_df, label_column="mbti")
    logger.info(f"After upsampling full dataset: {len(upsampled_df)} samples")

    # Step 2: Split the upsampled data
    train_df, val_df, test_df = sampler.train_val_test_split(
        upsampled_df, stratify_column="mbti"
    )

    return _train_and_evaluate(
        train_df, val_df, test_df, config, device,
        checkpoint_dir=CHECKPOINT_DIR / "bert_mbti_leaky",
        label="Leaky",
    )


def run_robust_pipeline(
    full_df: pd.DataFrame,
    config: BERTConfig,
    device: torch.device,
    sampler: DataSampler,
) -> dict:
    """
    Improved Robust Methodology (Current):
    Split first into train/val/test, THEN upsample only the training set.

    No data leakage: test and validation sets remain untouched and unbiased.
    """
    logger.info("=" * 60)
    logger.info("ROBUST PIPELINE: Split -> Upsample Train Only (Improved)")
    logger.info("=" * 60)

    # Step 1: Split the original data
    train_df, val_df, test_df = sampler.train_val_test_split(
        full_df, stratify_column="mbti"
    )

    # Step 2: Upsample ONLY the training set
    train_df = sampler.upsample_minority_classes(train_df, label_column="mbti")
    logger.info(f"After upsampling train only: {len(train_df)} train samples")

    return _train_and_evaluate(
        train_df, val_df, test_df, config, device,
        checkpoint_dir=CHECKPOINT_DIR / "bert_mbti_robust",
        label="Robust",
    )


def _train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: BERTConfig,
    device: torch.device,
    checkpoint_dir: Path,
    label: str,
) -> dict:
    """Shared training and evaluation logic for both pipelines."""
    logger.info(f"[{label}] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Create datasets
    train_dataset = ReviewDataset(
        df=train_df,
        text_column="clean_text",
        label_column="mbti",
        config=config,
    )
    val_dataset = ReviewDataset(
        df=val_df,
        text_column="clean_text",
        label_column="mbti",
        config=config,
    )
    test_dataset = ReviewDataset(
        df=test_df,
        text_column="clean_text",
        label_column="mbti",
        config=config,
    )

    # Create data loaders (num_workers=0 for Windows)
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset,
        batch_size=config.batch_size,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=DataCollator(),
        pin_memory=True,
    )

    # Create fresh model
    model = MBTIMultiLabelClassifier(config=config)

    # Create trainer
    trainer = MBTITrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Train
    start_time = time.time()
    history = trainer.train()
    train_time = time.time() - start_time

    # Evaluate on test set
    test_metrics = _evaluate_on_test(model, test_loader, device)

    # Get final validation metrics
    val_metrics = trainer.validate()

    result = {
        "label": label,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_time_seconds": train_time,
        "val_accuracy": val_metrics.accuracy,
        "val_precision": val_metrics.precision,
        "val_recall": val_metrics.recall,
        "val_f1": val_metrics.f1_score,
        "val_ei_acc": val_metrics.ei_acc,
        "val_ns_acc": val_metrics.ns_acc,
        "val_ft_acc": val_metrics.ft_acc,
        "val_jp_acc": val_metrics.jp_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_ei_acc": test_metrics["ei_acc"],
        "test_ns_acc": test_metrics["ns_acc"],
        "test_ft_acc": test_metrics["ft_acc"],
        "test_jp_acc": test_metrics["jp_acc"],
        "history": history,
    }

    logger.info(f"[{label}] Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"[{label}] Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"[{label}] Dimension Acc - EI: {test_metrics['ei_acc']:.4f}, "
                f"NS: {test_metrics['ns_acc']:.4f}, "
                f"FT: {test_metrics['ft_acc']:.4f}, "
                f"JP: {test_metrics['jp_acc']:.4f}")

    return result


@torch.no_grad()
def _evaluate_on_test(
    model: MBTIMultiLabelClassifier,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate the trained model on the test set."""
    model.eval()
    dims = ["EI", "SN", "TF", "JP"]
    all_preds = {d: [] for d in dims}
    all_labels = {d: [] for d in dims}

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = {d: batch[d].to(device) for d in dims}

        outputs = model(input_ids, attention_mask)

        for d in dims:
            preds = torch.argmax(outputs["logits"][d], dim=-1)
            all_preds[d].extend(preds.cpu().numpy())
            all_labels[d].extend(labels[d].cpu().numpy())

    # Per-dimension accuracy
    dim_accs = {d: accuracy_score(all_labels[d], all_preds[d]) for d in dims}

    # Global metrics (flatten all dimensions)
    flat_labels = np.concatenate([all_labels[d] for d in dims])
    flat_preds = np.concatenate([all_preds[d] for d in dims])

    return {
        "accuracy": accuracy_score(flat_labels, flat_preds),
        "precision": precision_score(flat_labels, flat_preds, average="macro"),
        "recall": recall_score(flat_labels, flat_preds, average="macro"),
        "f1": f1_score(flat_labels, flat_preds, average="macro"),
        "ei_acc": dim_accs["EI"],
        "ns_acc": dim_accs["SN"],
        "ft_acc": dim_accs["FT"],
        "jp_acc": dim_accs["JP"],
    }


def generate_comparison_table(leaky: dict, robust: dict) -> pd.DataFrame:
    """Generate side-by-side comparison table for the thesis."""
    metrics = [
        ("Overall Accuracy", "test_accuracy"),
        ("Precision (macro)", "test_precision"),
        ("Recall (macro)", "test_recall"),
        ("F1 Score (macro)", "test_f1"),
        ("E/I Dimension Acc", "test_ei_acc"),
        ("S/N Dimension Acc", "test_ns_acc"),
        ("T/F Dimension Acc", "test_ft_acc"),
        ("J/P Dimension Acc", "test_jp_acc"),
        ("Training Samples", "train_samples"),
        ("Training Time (s)", "train_time_seconds"),
    ]

    rows = []
    for display_name, key in metrics:
        leaky_val = leaky[key]
        robust_val = robust[key]
        if isinstance(leaky_val, float) and key != "train_time_seconds":
            rows.append({
                "Metric": display_name,
                "Replicated Leaky (Omer)": f"{leaky_val:.4f}",
                "Improved Robust (Current)": f"{robust_val:.4f}",
                "Delta": f"{robust_val - leaky_val:+.4f}",
            })
        else:
            rows.append({
                "Metric": display_name,
                "Replicated Leaky (Omer)": f"{leaky_val}",
                "Improved Robust (Current)": f"{robust_val}",
                "Delta": "",
            })

    return pd.DataFrame(rows)


def generate_latex_table(df: pd.DataFrame) -> str:
    """Convert comparison DataFrame to LaTeX table for thesis inclusion."""
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{MBTI Classification: Leaky vs Robust Methodology Comparison}\n"
    latex += "\\label{tab:methodology_comparison}\n"
    latex += "\\begin{tabular}{l c c c}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Metric} & \\textbf{Replicated Leaky (Omer)} & "
    latex += "\\textbf{Improved Robust (Current)} & \\textbf{Delta} \\\\\n"
    latex += "\\midrule\n"

    for _, row in df.iterrows():
        latex += f"{row['Metric']} & {row['Replicated Leaky (Omer)']} & "
        latex += f"{row['Improved Robust (Current)']} & {row['Delta']} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def main(args):
    """Run the methodology comparison."""
    setup_logging(level=logging.INFO)
    set_seed(42)

    config = get_config()
    if args.epochs:
        config.bert.num_epochs = args.epochs
    if args.quick:
        config.bert.num_epochs = min(config.bert.num_epochs, 3)
        config.bert.max_length = 64

    device = get_device(args.device)

    # Load data
    logger.info("Loading MBTI data...")
    full_df = load_mbti_data(PROCESSED_DATA_DIR)
    logger.info(f"Total samples: {len(full_df)}")
    logger.info(f"MBTI distribution:\n{full_df['mbti'].value_counts()}")

    sampler = DataSampler()

    # Run both pipelines
    leaky_results = run_leaky_pipeline(full_df, config.bert, device, sampler)
    robust_results = run_robust_pipeline(full_df, config.bert, device, sampler)

    # Generate comparison table
    comparison_df = generate_comparison_table(leaky_results, robust_results)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("METHODOLOGY COMPARISON RESULTS")
    logger.info("=" * 80)
    print(comparison_df.to_string(index=False))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "phase1_methodology_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved comparison table to: {csv_path}")

    latex_str = generate_latex_table(comparison_df)
    latex_path = RESULTS_DIR / "phase1_methodology_comparison.tex"
    latex_path.write_text(latex_str, encoding="utf-8")
    logger.info(f"Saved LaTeX table to: {latex_path}")

    logger.info("\nConclusion: The leaky methodology inflates metrics due to data")
    logger.info("leakage from upsampled duplicates appearing in the test set.")
    logger.info("The robust methodology provides an honest performance baseline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Leaky vs Robust MBTI training methodology"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced epochs")

    args = parser.parse_args()
    main(args)
