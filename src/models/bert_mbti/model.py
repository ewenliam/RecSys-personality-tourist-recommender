"""
BERT-based MBTI Personality Classifier.

Predicts one of 16 MBTI personality types from user review text.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional

from src.config.settings import BERTConfig, get_config


class MBTIClassifier(nn.Module):
    """BERT-based classifier for MBTI personality prediction."""

    # MBTI type labels
    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP",
    ]

    def __init__(
        self,
        config: Optional[BERTConfig] = None,
        pretrained: bool = True,
    ):
        """
        Initialize the MBTI classifier.

        Args:
            config: BERT configuration.
            pretrained: Whether to load pretrained weights.
        """
        super().__init__()
        self.config = config or get_config().bert

        # Load BERT model
        if pretrained:
            self.bert = AutoModel.from_pretrained(self.config.model_name)
        else:
            bert_config = AutoConfig.from_pretrained(self.config.model_name)
            self.bert = AutoModel.from_config(bert_config)

        # Get hidden size from BERT config
        self.hidden_size = self.bert.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.hidden_size, self.config.num_labels)

        # Initialize classifier weights
        self._init_weights(self.classifier)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for linear layers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            labels: Optional labels for training [batch_size].

        Returns:
            Dictionary with logits and optional loss.
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        result = {"logits": logits}

        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get BERT embeddings without classification.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].

        Returns:
            Embeddings [batch_size, hidden_size].
        """
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.last_hidden_state[:, 0, :]

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict MBTI types.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].

        Returns:
            Tuple of (predicted_labels, probabilities).
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        return predictions, probs

    def predict_mbti_type(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[str]:
        """
        Predict MBTI type as string labels.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].

        Returns:
            List of MBTI type strings.
        """
        predictions, _ = self.predict(input_ids, attention_mask)
        return [self.MBTI_TYPES[idx] for idx in predictions.cpu().tolist()]


class MBTIMultiLabelClassifier(nn.Module):
    """
    Multi-label MBTI classifier.

    Predicts each MBTI dimension separately:
    - E/I (Extraversion/Introversion)
    - S/N (Sensing/Intuition)
    - T/F (Thinking/Feeling)
    - J/P (Judging/Perceiving)
    """

    DIMENSIONS = ["EI", "SN", "TF", "JP"]

    def __init__(
        self,
        config: Optional[BERTConfig] = None,
        pretrained: bool = True,
    ):
        """
        Initialize the multi-label classifier.

        Args:
            config: BERT configuration.
            pretrained: Whether to load pretrained weights.
        """
        super().__init__()
        self.config = config or get_config().bert

        # Load BERT model
        if pretrained:
            self.bert = AutoModel.from_pretrained(self.config.model_name)
        else:
            bert_config = AutoConfig.from_pretrained(self.config.model_name)
            self.bert = AutoModel.from_config(bert_config)

        self.hidden_size = self.bert.config.hidden_size

        # Separate classifier for each dimension
        self.dropout = nn.Dropout(p=0.1)
        self.classifiers = nn.ModuleDict({
            dim: nn.Linear(self.hidden_size, 2)
            for dim in self.DIMENSIONS
        })

        # Optional per-dimension class weights for the loss.  Set via
        # set_class_weights() from the training-set label frequencies so the
        # minority class (e.g. "E", "S") is not ignored.  None -> unweighted.
        self.class_weights: Optional[dict[str, torch.Tensor]] = None

    def set_class_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Provide per-dimension class weights, e.g. {'EI': tensor([w_E, w_I])}."""
        self.class_weights = weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            labels: Optional dict of dimension -> binary labels.

        Returns:
            Dictionary with logits per dimension and optional loss.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Get logits for each dimension
        logits = {
            dim: classifier(pooled_output)
            for dim, classifier in self.classifiers.items()
        }

        result = {"logits": logits}

        # Calculate loss if labels provided.  Each dimension uses its own
        # class weights (if set) so imbalanced heads (E/I, S/N) are trained to
        # predict the minority class rather than collapsing to the prior.
        if labels is not None:
            total_loss = 0.0
            for dim in self.DIMENSIONS:
                if dim in labels:
                    weight = None
                    if self.class_weights is not None and dim in self.class_weights:
                        weight = self.class_weights[dim].to(logits[dim].device)
                    loss_fn = nn.CrossEntropyLoss(weight=weight)
                    total_loss += loss_fn(logits[dim], labels[dim])
            result["loss"] = total_loss / len(self.DIMENSIONS)

        return result

    def predict_mbti_type(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[str]:
        """
        Predict full MBTI type from dimension predictions.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].

        Returns:
            List of MBTI type strings.
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        batch_size = input_ids.size(0)
        mbti_types = []

        # Map dimension predictions to letters
        dim_to_letters = {
            "EI": ("E", "I"),
            "SN": ("S", "N"),
            "TF": ("T", "F"),
            "JP": ("J", "P"),
        }

        for i in range(batch_size):
            mbti_str = ""
            for dim in self.DIMENSIONS:
                logits = outputs["logits"][dim][i]
                pred = torch.argmax(logits).item()
                letters = dim_to_letters[dim]
                mbti_str += letters[pred]
            mbti_types.append(mbti_str)

        return mbti_types
