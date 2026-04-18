"""
BERT-MBTI Embedding Adapter for BERTopic.

Wraps the fine-tuned BERT-MBTI model to produce CLS token embeddings
compatible with BERTopic's embedding_model interface. This replaces
generic sentence-transformers with personality-informed representations.

The key insight: reviews embedded through a model fine-tuned on MBTI
classification carry implicit personality-discriminative features,
producing topic clusters that are semantically richer than generic
sentence embeddings.
"""
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from src.config.settings import BERTConfig, get_config, CHECKPOINT_DIR

logger = logging.getLogger(__name__)


class _TextDataset(Dataset):
    """Lightweight dataset for tokenizing text for BERT."""

    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


class MBTIEmbedder:
    """
    Wraps a fine-tuned BERT-MBTI model to extract CLS embeddings
    compatible with BERTopic's embedding_model protocol.

    BERTopic requires an object with:
        .encode(documents: List[str]) -> np.ndarray
    Optionally:
        .get_sentence_embedding_dimension() -> int

    This class provides both, using the BERT backbone from the
    trained MBTIMultiLabelClassifier.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[BERTConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the MBTI embedder.

        Args:
            model_path: Path to trained MBTI model checkpoint.
                Defaults to CHECKPOINT_DIR / "bert_mbti_robust" / "best_model.pt"
            config: BERT configuration.
            device: Compute device ("cuda", "cpu", "mps").
        """
        from src.models.bert_mbti.model import MBTIMultiLabelClassifier

        self.config = config or get_config().bert
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Load fine-tuned model
        self.model = MBTIMultiLabelClassifier(config=self.config)
        model_path = model_path or (CHECKPOINT_DIR / "bert_mbti_robust" / "best_model.pt")

        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded MBTI model from {model_path}")
        else:
            logger.warning(
                f"No checkpoint found at {model_path}. "
                "Using untrained BERT backbone for embeddings."
            )

        self.model.to(self.device)
        self.model.eval()

        # Cache the embedding dimension from BERT config
        self._embedding_dim = self.model.bert.config.hidden_size

    def get_sentence_embedding_dimension(self) -> int:
        """Return the embedding dimension (768 for bert-base-uncased)."""
        return self._embedding_dim

    @torch.no_grad()
    def encode(
        self,
        documents: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode documents to CLS token embeddings from the BERT-MBTI backbone.

        This is the method BERTopic calls. The signature matches
        sentence-transformers' SentenceTransformer.encode().

        Args:
            documents: List of text strings to encode.
            batch_size: Batch size for inference.
            show_progress_bar: Whether to show tqdm progress.

        Returns:
            Embeddings array of shape [len(documents), embedding_dim].
        """
        dataset = _TextDataset(
            texts=documents,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device == "cuda"),
        )

        all_embeddings = []

        iterator = tqdm(loader, desc="MBTI embedding") if show_progress_bar else loader

        for batch in iterator:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Extract CLS token from BERT backbone (not the classifiers)
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embeddings.cpu().numpy())

        embeddings = np.vstack(all_embeddings)

        # L2 normalize for cosine similarity compatibility
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        logger.info(
            f"Generated MBTI embeddings: {embeddings.shape} "
            f"(dim={self._embedding_dim})"
        )

        return embeddings

    def encode_venues(
        self,
        venue_reviews: dict[str, list[str]],
        max_reviews_per_venue: int = 5,
        max_chars_per_review: int = 500,
        batch_size: int = 32,
    ) -> tuple[list[str], np.ndarray]:
        """
        Encode venue reviews into aggregated venue embeddings.

        Concatenates up to max_reviews_per_venue reviews per venue,
        encodes them, then mean-pools to produce one embedding per venue.

        Args:
            venue_reviews: Dict mapping venue_id -> list of review texts.
            max_reviews_per_venue: Max reviews to use per venue.
            max_chars_per_review: Max characters per review (truncation).
            batch_size: Batch size for BERT inference.

        Returns:
            Tuple of (venue_ids, venue_embeddings [n_venues, dim]).
        """
        venue_ids = list(venue_reviews.keys())

        # Create concatenated documents per venue
        documents = []
        for vid in venue_ids:
            reviews = venue_reviews[vid][:max_reviews_per_venue]
            doc = " ".join(r[:max_chars_per_review] for r in reviews)
            documents.append(doc)

        # Encode all venue documents
        embeddings = self.encode(documents, batch_size=batch_size)

        logger.info(f"Encoded {len(venue_ids)} venues -> {embeddings.shape}")
        return venue_ids, embeddings
