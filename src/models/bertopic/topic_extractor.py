"""
BERTopic-based Topic Extraction.

Extracts semantic topics from venue reviews using BERTopic with
custom embedding models and dimensionality reduction.
"""
import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.settings import BERTopicConfig, get_config, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class TopicInfo:
    """Information about an extracted topic."""
    topic_id: int
    name: str
    count: int
    representation: list[str]
    representative_docs: list[str]


class VenueTopicExtractor:
    """Extract topics from venue reviews using BERTopic."""

    def __init__(
        self,
        config: Optional[BERTopicConfig] = None,
        use_multimodal: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize the topic extractor.

        Args:
            config: BERTopic configuration.
            use_multimodal: Whether to use multimodal embeddings.
            device: Device for inference.
        """
        import torch

        self.config = config or get_config().bertopic
        self.use_multimodal = use_multimodal
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.topic_model = None
        self.embeddings = None
        self.topics = None
        self.probs = None

    def _create_embedding_model(self):
        """Create the embedding model."""
        if self.use_multimodal:
            from .multimodal import MultimodalEmbedder
            return MultimodalEmbedder(
                text_model=self.config.embedding_model,
                image_model=self.config.clip_model,
                device=self.device,
            )
        else:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(
                self.config.embedding_model,
                device=self.device,
            )

    def _create_umap_model(self):
        """Create UMAP dimensionality reduction model."""
        from umap import UMAP

        return UMAP(
            n_neighbors=self.config.umap_n_neighbors,
            n_components=self.config.umap_n_components,
            min_dist=self.config.umap_min_dist,
            metric=self.config.umap_metric,
            random_state=42,
        )

    def _create_hdbscan_model(self):
        """Create HDBSCAN clustering model."""
        from hdbscan import HDBSCAN

        return HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            metric="euclidean",
            prediction_data=True,
        )

    def _create_kmeans_model(self):
        """Create K-Means clustering model as alternative."""
        from sklearn.cluster import KMeans

        return KMeans(
            n_clusters=self.config.kmeans_n_clusters,
            random_state=42,
            n_init=10,
        )

    def _create_vectorizer_model(self):
        """Create vectorizer for c-TF-IDF."""
        from sklearn.feature_extraction.text import CountVectorizer

        return CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
        )

    def _create_ctfidf_model(self):
        """Create c-TF-IDF model."""
        from bertopic.vectorizers import ClassTfidfTransformer

        return ClassTfidfTransformer(reduce_frequent_words=True)

    def _create_representation_model(self):
        """Create representation model for topic naming."""
        from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

        # Use KeyBERT-inspired representation with MMR for diversity
        return [
            KeyBERTInspired(),
            MaximalMarginalRelevance(diversity=0.3),
        ]

    def build_model(self, use_kmeans: bool = False):
        """
        Build the BERTopic model with all components.

        Args:
            use_kmeans: Use K-Means instead of HDBSCAN for clustering.
        """
        from bertopic import BERTopic

        # Create components
        umap_model = self._create_umap_model()
        cluster_model = self._create_kmeans_model() if use_kmeans else self._create_hdbscan_model()
        vectorizer_model = self._create_vectorizer_model()
        ctfidf_model = self._create_ctfidf_model()
        representation_model = self._create_representation_model()

        # Build BERTopic
        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            nr_topics=self.config.nr_topics,
            verbose=True,
        )

        logger.info("BERTopic model built successfully")

    def fit(
        self,
        documents: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> tuple[list[int], np.ndarray]:
        """
        Fit the topic model on documents.

        Args:
            documents: List of document texts.
            embeddings: Optional pre-computed embeddings.

        Returns:
            Tuple of (topics, probabilities).
        """
        if self.topic_model is None:
            self.build_model()

        logger.info(f"Fitting BERTopic on {len(documents)} documents")

        # Fit the model
        self.topics, self.probs = self.topic_model.fit_transform(
            documents,
            embeddings=embeddings,
        )

        # Store embeddings if computed
        if embeddings is None and hasattr(self.topic_model, "embedding_model"):
            self.embeddings = self.topic_model._extract_embeddings(
                documents,
                method="document",
            )
        else:
            self.embeddings = embeddings

        logger.info(f"Found {len(set(self.topics)) - 1} topics (excluding outliers)")
        return self.topics, self.probs

    def fit_transform_venues(
        self,
        venue_reviews: dict[str, list[str]],
        embeddings: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Fit topics on venue reviews and return venue-topic mapping.

        Args:
            venue_reviews: Dict mapping venue_id to list of reviews.
            embeddings: Optional pre-computed venue embeddings.

        Returns:
            DataFrame with venue_id, topic, and topic probability.
        """
        # Aggregate reviews per venue
        venue_ids = list(venue_reviews.keys())
        documents = [" ".join(reviews[:10]) for reviews in venue_reviews.values()]  # Limit reviews

        # Fit
        topics, probs = self.fit(documents, embeddings)

        # Create result DataFrame
        result = pd.DataFrame({
            "venue_id": venue_ids,
            "topic": topics,
            "topic_prob": [p.max() if isinstance(p, np.ndarray) else p for p in probs],
        })

        # Add topic names
        topic_info = self.get_topic_info()
        topic_names = {t.topic_id: t.name for t in topic_info}
        result["topic_name"] = result["topic"].map(topic_names)

        return result

    def transform(
        self,
        documents: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> tuple[list[int], np.ndarray]:
        """
        Transform new documents to topics.

        Args:
            documents: List of document texts.
            embeddings: Optional pre-computed embeddings.

        Returns:
            Tuple of (topics, probabilities).
        """
        if self.topic_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.topic_model.transform(documents, embeddings=embeddings)

    def get_topic_info(self) -> list[TopicInfo]:
        """Get information about all topics."""
        if self.topic_model is None:
            return []

        topic_info = self.topic_model.get_topic_info()
        result = []

        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                continue  # Skip outliers

            # Get topic representation
            representation = self.topic_model.get_topic(topic_id)
            rep_words = [word for word, _ in representation[:10]]

            # Get representative docs
            rep_docs = []
            if hasattr(self.topic_model, "representative_docs_"):
                rep_docs = self.topic_model.representative_docs_.get(topic_id, [])[:3]

            result.append(TopicInfo(
                topic_id=topic_id,
                name=row.get("Name", f"Topic_{topic_id}"),
                count=row["Count"],
                representation=rep_words,
                representative_docs=rep_docs,
            ))

        return result

    def get_topic_embeddings(self) -> np.ndarray:
        """Get embeddings for each topic centroid."""
        if self.topic_model is None or self.embeddings is None:
            raise ValueError("Model not fitted or embeddings not available.")

        topics = np.array(self.topics)
        unique_topics = sorted(set(topics) - {-1})

        topic_embeddings = []
        for topic_id in unique_topics:
            mask = topics == topic_id
            topic_emb = self.embeddings[mask].mean(axis=0)
            topic_embeddings.append(topic_emb)

        return np.stack(topic_embeddings)

    def get_document_topic_matrix(self) -> np.ndarray:
        """Get document-topic probability matrix."""
        if self.probs is None:
            raise ValueError("Model not fitted.")

        if isinstance(self.probs, np.ndarray) and len(self.probs.shape) == 2:
            return self.probs

        # Convert to matrix if needed
        n_docs = len(self.topics)
        n_topics = len(set(self.topics)) - 1  # Exclude outliers

        matrix = np.zeros((n_docs, n_topics))
        for i, (topic, prob) in enumerate(zip(self.topics, self.probs)):
            if topic >= 0:
                matrix[i, topic] = prob if isinstance(prob, (int, float)) else prob.max()

        return matrix

    def visualize_topics(self, output_path: Optional[Path] = None):
        """Generate topic visualization."""
        if self.topic_model is None:
            raise ValueError("Model not fitted.")

        fig = self.topic_model.visualize_topics()

        if output_path:
            fig.write_html(str(output_path))
            logger.info(f"Saved topic visualization to {output_path}")

        return fig

    def visualize_barchart(self, top_n_topics: int = 10, output_path: Optional[Path] = None):
        """Generate topic barchart visualization."""
        if self.topic_model is None:
            raise ValueError("Model not fitted.")

        fig = self.topic_model.visualize_barchart(top_n_topics=top_n_topics)

        if output_path:
            fig.write_html(str(output_path))
            logger.info(f"Saved barchart to {output_path}")

        return fig

    def save(self, path: Union[str, Path]) -> None:
        """Save the topic model."""
        if self.topic_model is None:
            raise ValueError("No model to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.topic_model.save(str(path))
        logger.info(f"Saved topic model to {path}")

        # Save embeddings separately
        if self.embeddings is not None:
            np.save(path.parent / "topic_embeddings.npy", self.embeddings)

    def load(self, path: Union[str, Path]) -> None:
        """Load a saved topic model."""
        from bertopic import BERTopic

        path = Path(path)
        self.topic_model = BERTopic.load(str(path))
        logger.info(f"Loaded topic model from {path}")

        # Load embeddings if available
        emb_path = path.parent / "topic_embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)


def extract_venue_topics(
    reviews_df: pd.DataFrame,
    business_id_col: str = "business_id",
    text_col: str = "clean_text",
    use_kmeans: bool = True,
    n_clusters: int = 50,
    save_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, VenueTopicExtractor]:
    """
    Extract topics for venues from reviews.

    Args:
        reviews_df: DataFrame with reviews.
        business_id_col: Column name for business ID.
        text_col: Column name for review text.
        use_kmeans: Use K-Means clustering.
        n_clusters: Number of clusters for K-Means.
        save_path: Path to save the model.

    Returns:
        Tuple of (venue_topics DataFrame, extractor).
    """
    # Group reviews by venue
    venue_reviews = reviews_df.groupby(business_id_col)[text_col].apply(list).to_dict()

    # Configure and build extractor
    config = get_config().bertopic
    config.kmeans_n_clusters = n_clusters

    extractor = VenueTopicExtractor(config=config)
    extractor.build_model(use_kmeans=use_kmeans)

    # Fit and get venue topics
    venue_topics = extractor.fit_transform_venues(venue_reviews)

    # Save if path provided
    if save_path:
        extractor.save(save_path)
        venue_topics.to_parquet(save_path.parent / "venue_topics.parquet", index=False)

    return venue_topics, extractor
