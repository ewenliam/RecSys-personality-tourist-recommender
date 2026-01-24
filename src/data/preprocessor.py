"""
Text Preprocessing Pipeline.

Handles cleaning and tokenization of review text for BERT and topic modeling.
"""
import re
import html
import logging
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config.settings import BERTConfig, get_config

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Clean and normalize text data."""

    # Regex patterns for cleaning
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    URL_PATTERN = re.compile(r"http\S+|www\.\S+")
    EMAIL_PATTERN = re.compile(r"\S+@\S+")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    SPECIAL_CHAR_PATTERN = re.compile(r"[^\w\s.,!?;:'\"-]")

    def __init__(self, lowercase: bool = True, remove_urls: bool = True):
        """
        Initialize the preprocessor.

        Args:
            lowercase: Convert text to lowercase.
            remove_urls: Remove URLs from text.
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize a single text string.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text.
        """
        if not text or not isinstance(text, str):
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Remove HTML tags
        text = self.HTML_TAG_PATTERN.sub(" ", text)

        # Remove URLs
        if self.remove_urls:
            text = self.URL_PATTERN.sub(" ", text)

        # Remove emails
        text = self.EMAIL_PATTERN.sub(" ", text)

        # Remove special characters (keep basic punctuation)
        text = self.SPECIAL_CHAR_PATTERN.sub(" ", text)

        # Normalize whitespace
        text = self.WHITESPACE_PATTERN.sub(" ", text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        return text.strip()

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        output_column: str = "clean_text",
    ) -> pd.DataFrame:
        """
        Process all text in a DataFrame.

        Args:
            df: DataFrame with text data.
            text_column: Name of the column containing text.
            output_column: Name for the cleaned text column.

        Returns:
            DataFrame with cleaned text added.
        """
        logger.info(f"Cleaning {len(df)} texts")
        df = df.copy()
        df[output_column] = df[text_column].apply(self.clean_text)

        # Filter out empty texts
        original_count = len(df)
        df = df[df[output_column].str.len() > 0]
        removed = original_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} empty texts after cleaning")

        return df


class ReviewDataset(Dataset):
    """PyTorch Dataset for review data."""

    # MBTI type mapping
    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP",
    ]
    MBTI_TO_IDX = {mbti: idx for idx, mbti in enumerate(MBTI_TYPES)}
    IDX_TO_MBTI = {idx: mbti for idx, mbti in enumerate(MBTI_TYPES)}

    def __init__(
        self,
        texts: list[str],
        labels: Optional[list[str]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        config: Optional[BERTConfig] = None,
    ):
        """
        Initialize the dataset.

        Args:
            texts: List of review texts.
            labels: List of MBTI labels (optional for inference).
            tokenizer: HuggingFace tokenizer.
            config: BERT configuration.
        """
        self.texts = texts
        self.labels = labels
        self.config = config or get_config().bert

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        else:
            self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """Get a single item."""
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        # Add label if available
        if self.labels is not None:
            label = self.labels[idx]
            if isinstance(label, str):
                label_idx = self.MBTI_TO_IDX.get(label.upper(), 0)
            else:
                label_idx = int(label)
            item["labels"] = torch.tensor(label_idx, dtype=torch.long)

        return item

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        text_column: str = "clean_text",
        label_column: Optional[str] = "mbti",
        tokenizer: Optional[AutoTokenizer] = None,
        config: Optional[BERTConfig] = None,
    ) -> "ReviewDataset":
        """
        Create dataset from DataFrame.

        Args:
            df: DataFrame with text and optional labels.
            text_column: Column name for text.
            label_column: Column name for labels.
            tokenizer: HuggingFace tokenizer.
            config: BERT configuration.

        Returns:
            ReviewDataset instance.
        """
        texts = df[text_column].tolist()
        labels = df[label_column].tolist() if label_column and label_column in df.columns else None
        return cls(texts=texts, labels=labels, tokenizer=tokenizer, config=config)


class DataCollator:
    """Custom data collator for batching."""

    def __call__(self, features: list[dict]) -> dict:
        """Collate features into a batch."""
        batch = {}

        # Stack tensors
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])

        # Add labels if present
        if "labels" in features[0]:
            batch["labels"] = torch.stack([f["labels"] for f in features])

        return batch
