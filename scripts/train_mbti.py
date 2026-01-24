#!/usr/bin/env python3
"""
Train BERT MBTI Classifier.

Usage:
    python scripts/train_mbti.py --epochs 10 --batch-size 16
    python scripts/train_mbti.py --resume checkpoint_epoch_5.pt
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from src.config import get_config, PROCESSED_DATA_DIR, CHECKPOINT_DIR
from src.data.preprocessor import ReviewDataset
from src.models.bert_mbti import (
    MBTIClassifier,
    MBTITrainer,
    create_data_loaders,
)
from src.utils.helpers import setup_logging, set_seed, get_device

logger = logging.getLogger(__name__)


def load_mbti_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation data with MBTI labels.

    Note: This assumes you have MBTI-labeled data. If not, you'll need
    to either:
    1. Use an external MBTI dataset (e.g., Kaggle MBTI dataset)
    2. Generate pseudo-labels using a pre-trained MBTI classifier
    3. Use unsupervised clustering to create personality groups

    Args:
        data_dir: Directory with processed data.

    Returns:
        Tuple of (train_df, val_df).
    """
    train_path = data_dir / "train_reviews.parquet"
    val_path = data_dir / "val_reviews.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            "Run 'python scripts/prepare_data.py' first."
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Check for MBTI labels
    if "mbti" not in train_df.columns:
        logger.warning(
            "MBTI labels not found in data. Using synthetic labels for demo. "
            "For real training, provide MBTI-labeled data."
        )
        # Create synthetic labels for demonstration
        import numpy as np
        np.random.seed(42)
        mbti_types = [
            "INTJ", "INTP", "ENTJ", "ENTP",
            "INFJ", "INFP", "ENFJ", "ENFP",
            "ISTJ", "ISFJ", "ESTJ", "ESFJ",
            "ISTP", "ISFP", "ESTP", "ESFP",
        ]
        train_df["mbti"] = np.random.choice(mbti_types, size=len(train_df))
        val_df["mbti"] = np.random.choice(mbti_types, size=len(val_df))

    return train_df, val_df


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

    # Set seed for reproducibility
    set_seed(config.data.random_seed)

    # Get device
    device = get_device(args.device)

    # Load data
    logger.info("Loading data...")
    train_df, val_df = load_mbti_data(PROCESSED_DATA_DIR)

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")

    # Check text column
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

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=config.bert.batch_size,
        num_workers=args.num_workers,
    )

    # Create model
    logger.info("Creating model...")
    model = MBTIClassifier(config=config.bert)

    # Log model info
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

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    # Print classification report
    logger.info("\nClassification Report:")
    print(trainer.get_classification_report())

    # Save final metrics
    logger.info(f"\nBest validation loss: {history['best_val_loss']:.4f}")
    logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT MBTI classifier")

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to train on",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint filename to resume from",
    )

    args = parser.parse_args()
    main(args)
