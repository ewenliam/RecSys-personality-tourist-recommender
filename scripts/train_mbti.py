#!/usr/bin/env python3
"""
Train BERT MBTI Classifier.

Loads the Kaggle MBTI dataset (mbti_1.csv), explodes posts, cleans text,
splits into train/val/test, upsamples the training set only (robust
methodology), and trains a MBTIMultiLabelClassifier (4 binary heads).

Usage:
    python scripts/train_mbti.py --epochs 3 --batch-size 32
    python scripts/train_mbti.py --resume checkpoint_epoch_5.pt
"""
import argparse
import gc
import logging
import os
import sys
from pathlib import Path

# Prevent CUDA memory fragmentation on Windows (must be set before torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.config.settings import PROCESSED_DATA_DIR, CHECKPOINT_DIR, DATA_DIR
from src.data.preprocessor import ReviewDataset, TextPreprocessor
from src.models.bert_mbti import (
    MBTIMultiLabelClassifier,
    MBTITrainer,
    create_data_loaders,
)
from src.utils.helpers import setup_logging, set_seed, get_device

logger = logging.getLogger(__name__)


def load_mbti_data() -> pd.DataFrame:
    """
    Load the Kaggle MBTI dataset (mbti_1.csv).

    The CSV has columns: 'type' (MBTI type) and 'posts' (multiple posts
    separated by '|||'). We explode posts into individual rows and clean
    the text.

    Returns:
        DataFrame with 'mbti' and 'clean_text' columns.
    """
    mbti_csv_path = DATA_DIR / "raw" / "mbti_1.csv"

    if not mbti_csv_path.exists():
        raise FileNotFoundError(
            f"MBTI dataset not found at {mbti_csv_path}\n"
            "Download it from Kaggle: https://www.kaggle.com/datasnaek/mbti-type\n"
            "Place mbti_1.csv in data/raw/"
        )

    # dtype_backend="numpy_nullable" avoids pyarrow strings which
    # cause a 6GB realloc during explode on the massive posts column.
    mbti_df = pd.read_csv(
        mbti_csv_path,
        dtype_backend="numpy_nullable",
    )
    logger.info(f"Loaded {len(mbti_df)} MBTI-labeled users")

    # Explode posts: each user has multiple posts separated by '|||'
    # Build rows in plain Python to avoid pyarrow memory explosion
    rows = []
    for _, row in mbti_df.iterrows():
        mbti_type = str(row["type"])
        posts = str(row["posts"]).split("|||")
        for post in posts:
            post = post.strip()
            if post:
                rows.append({"mbti": mbti_type, "text": post})

    mbti_exploded = pd.DataFrame(rows)

    # Clean text
    preprocessor = TextPreprocessor()
    mbti_exploded = preprocessor.process_dataframe(mbti_exploded, text_column="text")

    logger.info(f"After exploding posts: {len(mbti_exploded)} samples")
    logger.info(f"MBTI distribution:\n{mbti_exploded['mbti'].value_counts()}")

    return mbti_exploded


def split_and_upsample(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split first, then upsample the training set only (robust methodology).

    This prevents data leakage - upsampled copies never appear in val/test.

    Args:
        df: Full dataset with 'mbti' column.
        test_size: Fraction for temp (val+test) split.
        val_ratio: Fraction of temp to use as validation.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # Step 1: Split FIRST
    train_df, temp_df = train_test_split(
        df, test_size=test_size, stratify=df["mbti"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=val_ratio, stratify=temp_df["mbti"], random_state=42
    )

    logger.info(f"Before upsampling - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Step 2: Upsample ONLY the training set
    majority_class_size = train_df["mbti"].value_counts().max()
    upsampled_classes = []

    for mbti_type in train_df["mbti"].unique():
        class_df = train_df[train_df["mbti"] == mbti_type]
        if len(class_df) < majority_class_size:
            class_upsampled = resample(
                class_df,
                replace=True,
                n_samples=majority_class_size,
                random_state=42,
            )
            upsampled_classes.append(class_upsampled)
        else:
            upsampled_classes.append(class_df)

    train_df = pd.concat(upsampled_classes).sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"After upsampling - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def main(args):
    """Main training function."""
    setup_logging(level=logging.INFO)
    config = get_config()

    # Override config with command line args
    if args.epochs:
        config.bert.num_epochs = args.epochs
    if args.batch_size:
        config.bert.batch_size = args.batch_size
    if args.learning_rate:
        config.bert.learning_rate = args.learning_rate

    set_seed(config.data.random_seed)
    device = get_device(args.device)

    # Load Kaggle MBTI data
    logger.info("Loading MBTI data from mbti_1.csv...")
    mbti_df = load_mbti_data()

    # Split first, upsample train only
    train_df, val_df, test_df = split_and_upsample(mbti_df)

    # Determine text column
    text_col = "clean_text" if "clean_text" in train_df.columns else "text"

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ReviewDataset.from_dataframe(
        train_df,
        text_column=text_col,
        label_column="mbti",
        config=config.bert,
    )
    val_dataset = ReviewDataset.from_dataframe(
        val_df,
        text_column=text_col,
        label_column="mbti",
        config=config.bert,
    )

    # Create data loaders (num_workers=0 for Windows compatibility)
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=config.bert.batch_size,
        num_workers=0,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create model
    logger.info("Creating model...")
    model = MBTIMultiLabelClassifier(config=config.bert)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = MBTITrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.bert,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = CHECKPOINT_DIR / "bert_mbti" / args.resume
        trainer.load_checkpoint(checkpoint_path)

    # Free any stale allocations before training
    gc.collect()
    torch.cuda.empty_cache()

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    # Print results
    logger.info(f"\nBest validation loss: {history['best_val_loss']:.4f}")
    logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT MBTI classifier")

    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to train on",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint filename to resume from",
    )

    args = parser.parse_args()
    main(args)
