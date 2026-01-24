"""
Multimodal Embedding Pipeline.

Combines text embeddings (Sentence-BERT) with image embeddings (CLIP)
for venue representation learning.
"""
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.config.settings import BERTopicConfig, get_config

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Generate text embeddings using Sentence-BERT."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the text embedder.

        Args:
            model_name: Sentence-BERT model name.
            device: Device for inference.
        """
        from sentence_transformers import SentenceTransformer

        config = get_config().bertopic
        self.model_name = model_name or config.embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading Sentence-BERT model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.

        Returns:
            Embeddings array [num_texts, embedding_dim].
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings

    def encode_aggregated(
        self,
        text_groups: list[list[str]],
        aggregation: str = "mean",
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode groups of texts and aggregate.

        Args:
            text_groups: List of text groups (e.g., reviews per venue).
            aggregation: Aggregation method ("mean", "max", "first").
            batch_size: Batch size for encoding.

        Returns:
            Aggregated embeddings [num_groups, embedding_dim].
        """
        all_embeddings = []

        for texts in tqdm(text_groups, desc="Encoding text groups"):
            if not texts:
                # Empty group - use zero vector
                all_embeddings.append(np.zeros(self.embedding_dim))
                continue

            # Encode all texts in the group
            group_embeddings = self.encode(texts, batch_size=batch_size, show_progress=False)

            # Aggregate
            if aggregation == "mean":
                agg_embedding = np.mean(group_embeddings, axis=0)
            elif aggregation == "max":
                agg_embedding = np.max(group_embeddings, axis=0)
            elif aggregation == "first":
                agg_embedding = group_embeddings[0]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            all_embeddings.append(agg_embedding)

        return np.stack(all_embeddings)


class ImageEmbedder:
    """Generate image embeddings using CLIP."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the image embedder.

        Args:
            model_name: CLIP model name.
            device: Device for inference.
        """
        import open_clip

        config = get_config().bertopic
        self.model_name = model_name or config.clip_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading CLIP model: {self.model_name}")

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained="openai",
            device=self.device,
        )
        self.model.eval()

        # Get embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            self.embedding_dim = self.model.encode_image(dummy).shape[-1]

    def load_image(self, image_path: Union[str, Path]) -> Optional[torch.Tensor]:
        """Load and preprocess an image."""
        try:
            image = Image.open(image_path).convert("RGB")
            return self.preprocess(image).unsqueeze(0)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

    @torch.no_grad()
    def encode(
        self,
        image_paths: list[Union[str, Path]],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode images to embeddings.

        Args:
            image_paths: List of image file paths.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.

        Returns:
            Embeddings array [num_images, embedding_dim].
        """
        embeddings = []
        failed_indices = []

        iterator = tqdm(range(0, len(image_paths), batch_size), desc="Encoding images") if show_progress else range(0, len(image_paths), batch_size)

        for i in iterator:
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_valid_indices = []

            for j, path in enumerate(batch_paths):
                img_tensor = self.load_image(path)
                if img_tensor is not None:
                    batch_images.append(img_tensor)
                    batch_valid_indices.append(i + j)
                else:
                    failed_indices.append(i + j)

            if batch_images:
                batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
                batch_embeddings = self.model.encode_image(batch_tensor)
                batch_embeddings = batch_embeddings.cpu().numpy()

                # Normalize embeddings
                batch_embeddings = batch_embeddings / np.linalg.norm(
                    batch_embeddings, axis=1, keepdims=True
                )

                for idx, emb in zip(batch_valid_indices, batch_embeddings):
                    embeddings.append((idx, emb))

        # Sort by original index and fill in failed ones with zeros
        embeddings.sort(key=lambda x: x[0])
        result = np.zeros((len(image_paths), self.embedding_dim))

        for idx, emb in embeddings:
            result[idx] = emb

        if failed_indices:
            logger.warning(f"Failed to load {len(failed_indices)} images")

        return result

    def encode_aggregated(
        self,
        image_path_groups: list[list[Union[str, Path]]],
        aggregation: str = "mean",
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode groups of images and aggregate.

        Args:
            image_path_groups: List of image path groups (e.g., photos per venue).
            aggregation: Aggregation method ("mean", "max", "first").
            batch_size: Batch size for encoding.

        Returns:
            Aggregated embeddings [num_groups, embedding_dim].
        """
        all_embeddings = []

        for paths in tqdm(image_path_groups, desc="Encoding image groups"):
            if not paths:
                all_embeddings.append(np.zeros(self.embedding_dim))
                continue

            # Encode all images in the group
            group_embeddings = self.encode(paths, batch_size=batch_size, show_progress=False)

            # Filter out zero vectors (failed loads)
            valid_mask = np.any(group_embeddings != 0, axis=1)
            valid_embeddings = group_embeddings[valid_mask]

            if len(valid_embeddings) == 0:
                all_embeddings.append(np.zeros(self.embedding_dim))
                continue

            # Aggregate
            if aggregation == "mean":
                agg_embedding = np.mean(valid_embeddings, axis=0)
            elif aggregation == "max":
                agg_embedding = np.max(valid_embeddings, axis=0)
            elif aggregation == "first":
                agg_embedding = valid_embeddings[0]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            all_embeddings.append(agg_embedding)

        return np.stack(all_embeddings)


class MultimodalEmbedder:
    """Combine text and image embeddings for multimodal representation."""

    def __init__(
        self,
        text_model: Optional[str] = None,
        image_model: Optional[str] = None,
        fusion_method: str = "concat",
        text_weight: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize the multimodal embedder.

        Args:
            text_model: Sentence-BERT model name.
            image_model: CLIP model name.
            fusion_method: How to combine embeddings ("concat", "weighted_sum", "attention").
            text_weight: Weight for text embeddings in weighted_sum.
            device: Device for inference.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fusion_method = fusion_method
        self.text_weight = text_weight

        # Initialize embedders
        self.text_embedder = TextEmbedder(model_name=text_model, device=self.device)
        self.image_embedder = ImageEmbedder(model_name=image_model, device=self.device)

        # Calculate output dimension
        if fusion_method == "concat":
            self.embedding_dim = self.text_embedder.embedding_dim + self.image_embedder.embedding_dim
        else:
            # For weighted_sum, dimensions must match or we project
            self.embedding_dim = max(
                self.text_embedder.embedding_dim,
                self.image_embedder.embedding_dim,
            )

    def encode(
        self,
        texts: list[str],
        image_paths: Optional[list[Union[str, Path]]] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode text and optional images to multimodal embeddings.

        Args:
            texts: List of text strings.
            image_paths: Optional list of image paths (same length as texts).
            batch_size: Batch size for encoding.

        Returns:
            Multimodal embeddings array.
        """
        # Encode text
        text_embeddings = self.text_embedder.encode(texts, batch_size=batch_size)

        if image_paths is None:
            # Text only - pad with zeros for image part if concat
            if self.fusion_method == "concat":
                image_padding = np.zeros((len(texts), self.image_embedder.embedding_dim))
                return np.concatenate([text_embeddings, image_padding], axis=1)
            return text_embeddings

        # Encode images
        image_embeddings = self.image_embedder.encode(image_paths, batch_size=batch_size)

        # Fuse embeddings
        return self._fuse(text_embeddings, image_embeddings)

    def _fuse(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Fuse text and image embeddings."""
        if self.fusion_method == "concat":
            return np.concatenate([text_embeddings, image_embeddings], axis=1)

        elif self.fusion_method == "weighted_sum":
            # Project to same dimension if needed
            if text_embeddings.shape[1] != image_embeddings.shape[1]:
                # Simple projection via padding or truncation
                target_dim = self.embedding_dim
                text_embeddings = self._project(text_embeddings, target_dim)
                image_embeddings = self._project(image_embeddings, target_dim)

            return self.text_weight * text_embeddings + (1 - self.text_weight) * image_embeddings

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _project(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Project embeddings to target dimension."""
        current_dim = embeddings.shape[1]
        if current_dim == target_dim:
            return embeddings
        elif current_dim < target_dim:
            # Pad with zeros
            padding = np.zeros((embeddings.shape[0], target_dim - current_dim))
            return np.concatenate([embeddings, padding], axis=1)
        else:
            # Truncate
            return embeddings[:, :target_dim]

    def encode_venues(
        self,
        venue_texts: list[list[str]],
        venue_images: Optional[list[list[Union[str, Path]]]] = None,
        text_aggregation: str = "mean",
        image_aggregation: str = "mean",
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode venues with aggregated reviews and images.

        Args:
            venue_texts: List of review lists per venue.
            venue_images: Optional list of image path lists per venue.
            text_aggregation: How to aggregate text embeddings.
            image_aggregation: How to aggregate image embeddings.
            batch_size: Batch size for encoding.

        Returns:
            Venue embeddings array [num_venues, embedding_dim].
        """
        # Aggregate text embeddings
        text_embeddings = self.text_embedder.encode_aggregated(
            venue_texts,
            aggregation=text_aggregation,
            batch_size=batch_size,
        )

        if venue_images is None:
            if self.fusion_method == "concat":
                image_padding = np.zeros((len(venue_texts), self.image_embedder.embedding_dim))
                return np.concatenate([text_embeddings, image_padding], axis=1)
            return text_embeddings

        # Aggregate image embeddings
        image_embeddings = self.image_embedder.encode_aggregated(
            venue_images,
            aggregation=image_aggregation,
            batch_size=batch_size,
        )

        # Fuse
        return self._fuse(text_embeddings, image_embeddings)
