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

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ReviewDataset(Dataset):
    def __init__(self, df, text_column, label_column, config):
        # FIX: Store the dataframe as a list of dictionaries so 'self.data' exists
        self.data = df.reset_index(drop=True).to_dict('records')
        self.text_column = text_column
        self.label_column = label_column
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    @classmethod
    def from_dataframe(cls, df, text_column, label_column, config):
        """Standard method to initialize the dataset."""
        return cls(df, text_column, label_column, config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx] # This line was failing before
        text = str(item[self.text_column])
        mbti_type = str(item[self.label_column]).upper()

        # Map 'INTJ' -> 4 separate binary labels
        labels = {
            "EI": 1 if mbti_type[0] == "I" else 0,
            "SN": 1 if mbti_type[1] == "N" else 0,
            "TF": 1 if mbti_type[2] == "F" else 0,
            "JP": 1 if mbti_type[3] == "P" else 0
        }

        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            # Return the 4 individual labels as tensors
            **{k: torch.tensor(v, dtype=torch.long) for k, v in labels.items()}
        }

class DataCollator:
    def __call__(self, batch):
        """
        Dynamically stacks all keys present in the batch samples.
        This handles input_ids, attention_mask, and the 4 MBTI labels.
        """
        # Get all keys present in the first item of the batch
        keys = batch[0].keys()
        
        return {
            key: torch.stack([item[key] for item in batch])
            for key in keys
        }
