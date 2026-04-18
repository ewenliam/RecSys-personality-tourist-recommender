"""
Training Pipeline for BERT MBTI Classifier.
Optimized for RTX 40-series GPUs with Mixed Precision.
"""
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np  # Add this line to fix the NameError
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
# Import autocast and GradScaler for Mixed Precision
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config.settings import BERTConfig, get_config, CHECKPOINT_DIR
from src.utils.helpers import save_checkpoint, EarlyStopping
from .model import MBTIClassifier

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Container for multi-label training metrics aligned with Table 3a2cb9."""
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    ei_acc: float
    ns_acc: float
    ft_acc: float
    jp_acc: float

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
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or get_config().bert
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR / "bert_mbti"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model.to(self.device)

        # Enable gradient checkpointing to cut activation memory ~50%
        # Essential for 8GB consumer GPUs (RTX 4060 etc.)
        if hasattr(self.model, "bert"):
            self.model.bert.gradient_checkpointing_enable()

        # Mixed Precision Setup: Initialize the GradScaler
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

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

        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            mode="min",
        )

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_history = []
        self.val_history = []

    def _create_optimizer(self) -> AdamW:
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

    def train_epoch(self) -> TrainingMetrics:
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
        )

        # Inside MBTITrainer.train_epoch
        # Inside trainer.py -> train_epoch
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
    
            # NEW: Extract the 4 separate dimension labels
            labels = {
            dim: batch[dim].to(self.device) 
            for dim in ["EI", "SN", "TF", "JP"]
            }

            self.optimizer.zero_grad()
            with autocast(device_type="cuda"):
                # Pass the dictionary of labels to the model
                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs["loss"]
    
            self.scaler.scale(loss).backward()
            
            # Unscale before clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

            if progress_bar.n % 10 == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Inside MBTITrainer.train_epoch
        avg_loss = total_loss / len(self.train_loader)
    
        return TrainingMetrics(
        loss=avg_loss,
        accuracy=0.0, 
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        ei_acc=0.0, 
        ns_acc=0.0, 
        ft_acc=0.0, 
        jp_acc=0.0 
    )

    @torch.no_grad()
    def validate(self) -> TrainingMetrics:
        self.model.eval()
        dims = ["EI", "SN", "TF", "JP"]
        all_preds = {d: [] for d in dims}
        all_labels = {d: [] for d in dims}
        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = {d: batch[d].to(self.device) for d in dims}

            with autocast(device_type="cuda"):
                outputs = self.model(input_ids, attention_mask, labels=labels)
                total_loss += outputs["loss"].item()

            for d in dims:
                preds = torch.argmax(outputs["logits"][d], dim=-1)
                all_preds[d].extend(preds.cpu().numpy())
                all_labels[d].extend(labels[d].cpu().numpy())

        # Calculate individual dimension accuracies
        dim_accs = {d: accuracy_score(all_labels[d], all_preds[d]) for d in dims}
        
        # Calculate global metrics for the main columns of your table
        # We flatten the results to treat all 4 trait-decisions equally
        # Inside MBTITrainer.validate
        flat_labels = np.concatenate([all_labels[d] for d in dims])
        flat_preds = np.concatenate([all_preds[d] for d in dims])

        return TrainingMetrics(
            loss=total_loss / len(self.val_loader),
            accuracy=accuracy_score(flat_labels, flat_preds),
            precision=precision_score(flat_labels, flat_preds, average='macro'),
            recall=recall_score(flat_labels, flat_preds, average='macro'),
            f1_score=f1_score(flat_labels, flat_preds, average='macro'),
            ei_acc=dim_accs["EI"],
            ns_acc=dim_accs["SN"],
            ft_acc=dim_accs["TF"],
            jp_acc=dim_accs["JP"]
        )
    
    def train(self) -> dict:
        logger.info(f"Starting training on {self.device}")
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            val_metrics = self.validate()
            self.val_history.append(val_metrics)

            logger.info(f"Epoch {epoch+1}: Train Loss {train_metrics.loss:.4f}, Val Acc {val_metrics.accuracy:.4f}")

            is_best = val_metrics.loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics.loss
            self.save_checkpoint(is_best=is_best)

            if self.early_stopping(val_metrics.loss):
                break

        return {
            "train_history": [vars(m) for m in self.train_history],
            "val_history": [vars(m) for m in self.val_history],
            "best_val_loss": self.best_val_loss,
        }

    def save_checkpoint(self, is_best: bool = False) -> None:
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

def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size: int = 16,
    num_workers: int = 0, # CHANGED: 0 is essential for Windows to prevent CPU bottlenecks
)    -> tuple[DataLoader, DataLoader]:
    from src.data.preprocessor import DataCollator
    collator = DataCollator()

    # pin_memory=True speeds up the transfer from CPU to GPU
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