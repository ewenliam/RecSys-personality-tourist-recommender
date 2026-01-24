"""
Training Pipeline for BERT MBTI Classifier.

Includes training loop, validation, and checkpoint management.
"""
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.config.settings import BERTConfig, get_config, CHECKPOINT_DIR
from src.utils.helpers import save_checkpoint, EarlyStopping, format_metrics
from .model import MBTIClassifier

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float
    accuracy: float
    f1_macro: float
    f1_weighted: float


class MBTITrainer:
    """Trainer for the MBTI classifier."""

    def __init__(
        self,
        model: MBTIClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[BERTConfig] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: MBTI classifier model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Training configuration.
            device: Device to train on.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or get_config().bert
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR / "bert_mbti"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy="linear",
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            mode="min",
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_history = []
        self.val_history = []

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with weight decay."""
        # Don't apply weight decay to bias and LayerNorm
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

    def train_epoch(self) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
        )

        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs["loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )

            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")

        return TrainingMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
        )

    @torch.no_grad()
    def validate(self) -> TrainingMetrics:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask, labels)
            total_loss += outputs["loss"].item()

            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")

        return TrainingMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
        )

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Dictionary with training history.
        """
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_metrics.loss:.4f}, Acc: {train_metrics.accuracy:.4f} | "
                f"Val Loss: {val_metrics.loss:.4f}, Acc: {val_metrics.accuracy:.4f}"
            )

            # Save checkpoint
            is_best = val_metrics.loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics.loss

            self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.early_stopping(val_metrics.loss):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return {
            "train_history": [vars(m) for m in self.train_history],
            "val_history": [vars(m) for m in self.val_history],
            "best_val_loss": self.best_val_loss,
        }

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save a training checkpoint."""
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        filepath = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt"
        save_checkpoint(state, filepath, is_best=is_best)

    def load_checkpoint(self, filepath: Path) -> None:
        """Load a training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch + 1}")

    @torch.no_grad()
    def get_classification_report(self) -> str:
        """Generate a detailed classification report."""
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            preds, _ = self.model.predict(input_ids, attention_mask)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

        return classification_report(
            all_labels,
            all_preds,
            target_names=MBTIClassifier.MBTI_TYPES,
            digits=4,
        )


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size: int = 16,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        batch_size: Batch size.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    from src.data.preprocessor import DataCollator

    collator = DataCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    return train_loader, val_loader
