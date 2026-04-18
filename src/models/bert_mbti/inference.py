"""
Inference Pipeline for BERT Multi-Label MBTI Classifier.
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
from .model import MBTIMultiLabelClassifier, MBTIClassifier

logger = logging.getLogger(__name__)


class MBTIInference:
    """Inference pipeline for MBTI Multi-Label prediction."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[BERTConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or get_config().bert
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # IMPORTANT: Initialize the Multi-Label Model
        self.model = MBTIMultiLabelClassifier(config=self.config)
        if model_path:
            self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path: Path) -> None:
        """Load model from checkpoint."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        logger.info(f"Loaded model from {model_path}")

    @torch.no_grad()
    def predict_single(self, text: str) -> dict:
        """Predict MBTI type and calculate joint probabilities for all 16 types."""
        import torch.nn.functional as F

        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 1. Get logits from the 4 multi-label heads
        outputs = self.model(input_ids, attention_mask)
        logits = outputs["logits"] # Dictionary: {'EI': tensor, 'SN': ..., 'TF': ..., 'JP': ...}

        # 2. Calculate probabilities for each dimension
        # Mapping: E=0/I=1, S=0/N=1, T=0/F=1, J=0/P=1
        probs = {
            dim: F.softmax(logits[dim], dim=-1)[0].cpu().numpy() 
            for dim in ["EI", "SN", "TF", "JP"]
        }

        # 3. Calculate joint probabilities for all 16 types
        type_probs = {}
        for mbti in MBTIClassifier.MBTI_TYPES:
            # Multiply the probability of each letter in the string
            p = 1.0
            p *= probs["EI"][1] if mbti[0] == "I" else probs["EI"][0]
            p *= probs["SN"][1] if mbti[1] == "N" else probs["SN"][0]
            p *= probs["TF"][1] if mbti[2] == "F" else probs["TF"][0]
            p *= probs["JP"][1] if mbti[3] == "P" else probs["JP"][0]
            type_probs[mbti] = float(p)

        # 4. Get the final predicted type and embedding
        mbti_type = self.model.predict_mbti_type(input_ids, attention_mask)[0]
        embedding = outputs["hidden_states"][-1][:, 0, :].cpu().numpy() # CLS token

        return {
            "mbti_type": mbti_type,
            "probabilities": type_probs, # This restores the missing key
            "embedding": embedding[0],
        }

    @torch.no_grad()
    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        return_embeddings: bool = False,
    ) -> dict:
        """Predict MBTI types for multiple texts."""
        dataset = ReviewDataset(
            df=pd.DataFrame({'clean_text': texts, 'mbti': ['INTJ']*len(texts)}), # Dummy labels to satisfy dataset
            text_column='clean_text',
            label_column='mbti',
            config=self.config,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollator(),
        )

        all_predictions = []
        all_embeddings = []

        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Get string predictions
            mbti_types = self.model.predict_mbti_type(input_ids, attention_mask)
            all_predictions.extend(mbti_types)

            if return_embeddings:
                outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu().numpy())

        result = {
            "mbti_types": all_predictions,
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
        """Aggregate multiple reviews for a user to find their final MBTI type."""
        logger.info(f"Predicting MBTI for {reviews_df[user_id_col].nunique()} users")

        texts = reviews_df[text_col].tolist()
        results = self.predict_batch(
            texts,
            batch_size=batch_size,
            return_embeddings=True,
        )

        reviews_df = reviews_df.copy()
        reviews_df["mbti_pred"] = results["mbti_types"]

        # Aggregate by finding the most common prediction ("majority" vote)
        user_mbti = reviews_df.groupby(user_id_col)["mbti_pred"].agg(
            lambda x: x.value_counts().index[0]
        ).reset_index()
        user_mbti.columns = [user_id_col, "mbti_type"]

        # Aggregate embeddings (mean pooling)
        reviews_df["embedding"] = list(results["embeddings"])
        user_embeddings = reviews_df.groupby(user_id_col)["embedding"].apply(
            lambda x: np.mean(np.stack(x.values), axis=0)
        )
        user_mbti["embedding"] = user_mbti[user_id_col].map(user_embeddings)

        logger.info(f"Generated MBTI predictions for {len(user_mbti)} users")
        return user_mbti

def load_inference_pipeline(
    checkpoint_name: str = "best_model.pt",
    checkpoint_dir: Optional[Path] = None,
) -> MBTIInference:
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR / "bert_mbti_balanced"
    model_path = checkpoint_dir / checkpoint_name
    return MBTIInference(model_path=model_path)