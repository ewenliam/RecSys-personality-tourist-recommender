"""
Inference Pipeline for BERT MBTI Classifier.

Handles batch inference and embedding extraction for all users.
"""
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from src.config.settings import BERTConfig, get_config, CHECKPOINT_DIR
from src.data.preprocessor import ReviewDataset, DataCollator
from .model import MBTIClassifier

logger = logging.getLogger(__name__)


class MBTIInference:
    """Inference pipeline for MBTI prediction."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[BERTConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the inference pipeline.

        Args:
            model_path: Path to model checkpoint.
            config: BERT configuration.
            device: Device for inference.
        """
        self.config = config or get_config().bert
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Load model
        self.model = MBTIClassifier(config=self.config)
        if model_path:
            self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path: Path) -> None:
        """Load model from checkpoint."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        logger.info(f"Loaded model from {model_path}")

    @torch.no_grad()
    def predict_single(self, text: str) -> dict:
        """
        Predict MBTI type for a single text.

        Args:
            text: Review text.

        Returns:
            Dictionary with prediction, probabilities, and embedding.
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get prediction
        predictions, probs = self.model.predict(input_ids, attention_mask)
        mbti_type = MBTIClassifier.MBTI_TYPES[predictions[0].item()]

        # Get embedding
        embedding = self.model.get_embeddings(input_ids, attention_mask)

        return {
            "mbti_type": mbti_type,
            "probabilities": {
                mbti: probs[0, i].item()
                for i, mbti in enumerate(MBTIClassifier.MBTI_TYPES)
            },
            "embedding": embedding[0].cpu().numpy(),
        }

    @torch.no_grad()
    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        return_embeddings: bool = False,
    ) -> dict:
        """
        Predict MBTI types for multiple texts.

        Args:
            texts: List of review texts.
            batch_size: Batch size for inference.
            return_embeddings: Whether to return embeddings.

        Returns:
            Dictionary with predictions and optional embeddings.
        """
        # Create dataset
        dataset = ReviewDataset(
            texts=texts,
            labels=None,
            tokenizer=self.tokenizer,
            config=self.config,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollator(),
        )

        all_predictions = []
        all_probs = []
        all_embeddings = []

        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Get predictions
            predictions, probs = self.model.predict(input_ids, attention_mask)
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            # Get embeddings if requested
            if return_embeddings:
                embeddings = self.model.get_embeddings(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())

        # Convert to MBTI types
        mbti_types = [
            MBTIClassifier.MBTI_TYPES[pred]
            for pred in all_predictions
        ]

        result = {
            "mbti_types": mbti_types,
            "predictions": all_predictions,
            "probabilities": np.vstack(all_probs),
        }

        if return_embeddings:
            result["embeddings"] = np.vstack(all_embeddings)

        return result

    def predict_users(
        self,
        reviews_df: pd.DataFrame,
        user_id_col: str = "user_id",
        text_col: str = "clean_text",
        aggregation: str = "majority",
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Predict MBTI types for users by aggregating their reviews.

        Args:
            reviews_df: DataFrame with user reviews.
            user_id_col: Column name for user ID.
            text_col: Column name for review text.
            aggregation: How to aggregate predictions ("majority", "mean_probs").
            batch_size: Batch size for inference.

        Returns:
            DataFrame with user_id and predicted MBTI.
        """
        logger.info(f"Predicting MBTI for {reviews_df[user_id_col].nunique()} users")

        # Get predictions for all reviews
        texts = reviews_df[text_col].tolist()
        results = self.predict_batch(
            texts,
            batch_size=batch_size,
            return_embeddings=True,
        )

        # Add predictions to dataframe
        reviews_df = reviews_df.copy()
        reviews_df["mbti_pred"] = results["mbti_types"]
        reviews_df["mbti_pred_idx"] = results["predictions"]

        # Store probabilities
        for i, mbti in enumerate(MBTIClassifier.MBTI_TYPES):
            reviews_df[f"prob_{mbti}"] = results["probabilities"][:, i]

        # Aggregate by user
        if aggregation == "majority":
            # Most common prediction
            user_mbti = reviews_df.groupby(user_id_col)["mbti_pred"].agg(
                lambda x: x.value_counts().index[0]
            ).reset_index()
            user_mbti.columns = [user_id_col, "mbti_type"]

        elif aggregation == "mean_probs":
            # Average probabilities then argmax
            prob_cols = [f"prob_{mbti}" for mbti in MBTIClassifier.MBTI_TYPES]
            user_probs = reviews_df.groupby(user_id_col)[prob_cols].mean()
            user_mbti_idx = user_probs.values.argmax(axis=1)
            user_mbti = pd.DataFrame({
                user_id_col: user_probs.index,
                "mbti_type": [MBTIClassifier.MBTI_TYPES[i] for i in user_mbti_idx],
            })

            # Add confidence scores
            for i, mbti in enumerate(MBTIClassifier.MBTI_TYPES):
                user_mbti[f"prob_{mbti}"] = user_probs[f"prob_{mbti}"].values

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Aggregate embeddings (mean pooling)
        embedding_dim = results["embeddings"].shape[1]
        reviews_df["embedding"] = list(results["embeddings"])

        user_embeddings = reviews_df.groupby(user_id_col)["embedding"].apply(
            lambda x: np.mean(np.stack(x.values), axis=0)
        )

        user_mbti["embedding"] = user_mbti[user_id_col].map(user_embeddings)

        logger.info(f"Generated MBTI predictions for {len(user_mbti)} users")
        return user_mbti

    def get_user_embeddings(
        self,
        reviews_df: pd.DataFrame,
        user_id_col: str = "user_id",
        text_col: str = "clean_text",
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        """
        Get aggregated BERT embeddings for each user.

        Args:
            reviews_df: DataFrame with user reviews.
            user_id_col: Column name for user ID.
            text_col: Column name for review text.
            batch_size: Batch size for inference.

        Returns:
            Dictionary mapping user_id to embedding vector.
        """
        # Get embeddings for all reviews
        texts = reviews_df[text_col].tolist()
        results = self.predict_batch(
            texts,
            batch_size=batch_size,
            return_embeddings=True,
        )

        # Create temporary dataframe for aggregation
        temp_df = reviews_df[[user_id_col]].copy()
        temp_df["embedding"] = list(results["embeddings"])

        # Aggregate by user (mean pooling)
        user_embeddings = temp_df.groupby(user_id_col)["embedding"].apply(
            lambda x: np.mean(np.stack(x.values), axis=0)
        ).to_dict()

        return user_embeddings


def load_inference_pipeline(
    checkpoint_name: str = "best_model.pt",
    checkpoint_dir: Optional[Path] = None,
) -> MBTIInference:
    """
    Load inference pipeline with the best model.

    Args:
        checkpoint_name: Name of the checkpoint file.
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        MBTIInference instance.
    """
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR / "bert_mbti"
    model_path = checkpoint_dir / checkpoint_name

    return MBTIInference(model_path=model_path)
