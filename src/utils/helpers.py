"""
Helper utilities for the recommendation system.
"""
import os
import random
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


def get_device(preferred: str = "cuda") -> torch.device:
    """
    Get the best available device.

    Args:
        preferred: Preferred device ("cuda", "mps", or "cpu").

    Returns:
        torch.device for computation.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif preferred == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def save_checkpoint(
    state: dict,
    filepath: Path,
    is_best: bool = False,
) -> None:
    """
    Save model checkpoint.

    Args:
        state: Dictionary containing model state.
        filepath: Path to save checkpoint.
        is_best: If True, also save as best model.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

    if is_best:
        best_path = filepath.parent / "best_model.pt"
        torch.save(state, best_path)
        logger.info(f"Best model saved to {best_path}")


def load_checkpoint(
    filepath: Path,
    map_location: Optional[str] = None,
) -> dict:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file.
        map_location: Device to load tensors to.

    Returns:
        Checkpoint dictionary.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=map_location)
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format metrics dictionary for logging."""
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return " | ".join(lines)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: "min" for loss, "max" for metrics like accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")

        return self.should_stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level.
        log_file: Optional file to write logs to.
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
