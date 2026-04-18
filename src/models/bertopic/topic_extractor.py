"""
BERTopic-based Topic Extraction.

Extracts semantic topics from venue reviews using BERTopic with
custom embedding models and dimensionality reduction.
"""
import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from sklearn.feature_extraction.text import CountVectorizer
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
        use_mbti_embeddings: bool = False,
        mbti_checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the topic extractor.

        Args:
            config: BERTopic configuration.
            use_multimodal: Whether to use multimodal embeddings.
            use_mbti_embeddings: Use BERT-MBTI model for embeddings
                instead of generic sentence-transformers.
            mbti_checkpoint_path: Path to trained MBTI model checkpoint.
            device: Device for inference.
        """
        import torch

        self.config = config or get_config().bertopic
        self.use_multimodal = use_multimodal
        self.use_mbti_embeddings = use_mbti_embeddings
        self.mbti_checkpoint_path = mbti_checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.topic_model = None
        self.embeddings = None
        self.topics = None
        self.probs = None

    def _create_embedding_model(self):
        """Create the embedding model."""
        if self.use_mbti_embeddings:
            return self._create_mbti_embedding_model()
        elif self.use_multimodal:
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

    def _create_mbti_embedding_model(self):
        """Create BERT-MBTI embedding model for personality-informed topics."""
        from .mbti_embedder import MBTIEmbedder

        return MBTIEmbedder(
            model_path=self.mbti_checkpoint_path,
            device=self.device,
        )

    def _create_umap_model(self):
        """Create a more stable UMAP model."""
        from umap import UMAP
        return UMAP(
            n_neighbors=15,
            n_components=5,
            metric='cosine',
            low_memory=True, # Saves RAM to prevent kernel death
            random_state=42,
            transform_queue_size=4.0 # Helps with Windows threading
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
        """Limit vocabulary to prevent the '500-minute' CPU hang."""
        from sklearn.feature_extraction.text import CountVectorizer
        return CountVectorizer(
            stop_words="english",
            min_df=10, 
            max_features=5000 # Further reduced to ensure it finishes
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
        """Build the BERTopic model with an explicit embedding model."""
        from bertopic import BERTopic
        
        # 1. Create the embedding model explicitly
        # This gives KeyBERTInspired the 'engine' it needs to embed words
        embedding_model = self._create_embedding_model()

        # 2. Create your other components
        umap_model = self._create_umap_model()
        cluster_model = self._create_kmeans_model() if use_kmeans else self._create_hdbscan_model()
        vectorizer_model = self._create_vectorizer_model()
        ctfidf_model = self._create_ctfidf_model()
        representation_model = self._create_representation_model()

        # 3. Initialize BERTopic with the embedding_model passed in
        self.topic_model = BERTopic(
            embedding_model=embedding_model,  # THIS IS THE FIX
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            nr_topics=self.config.nr_topics,
            verbose=True,
        )

        logger.info("BERTopic model built with explicit embedding engine.")

    def find_sweet_spot(
        self,
        documents: list[str],
        embeddings: np.ndarray,
        target_outlier_pct: float = 10.0,
        candidates: Optional[list[int]] = None,
    ) -> int:
        """
        Grid search over HDBSCAN min_cluster_size to hit target outlier rate.

        Runs UMAP once, then fits HDBSCAN for each candidate. Picks the
        min_cluster_size closest to target_outlier_pct, preferring the
        5-15% band.

        Args:
            documents: Pre-built venue documents.
            embeddings: Pre-computed document embeddings.
            target_outlier_pct: Desired outlier percentage (default 10%).
            candidates: List of min_cluster_size values to try.

        Returns:
            Best min_cluster_size value.
        """
        from hdbscan import HDBSCAN
        from umap import UMAP

        if candidates is None:
            candidates = [5, 8, 10, 15, 20, 30, 50]

        logger.info(f"Searching for sweet spot (target: {target_outlier_pct}% outliers)")

        # Shared UMAP reduction (expensive, do once)
        umap_model = UMAP(
            n_neighbors=self.config.umap_n_neighbors,
            n_components=self.config.umap_n_components,
            metric=self.config.umap_metric,
            low_memory=True,
            random_state=42,
        )
        reduced = umap_model.fit_transform(embeddings)

        results = []
        for mcs in candidates:
            hdbscan_model = HDBSCAN(
                min_cluster_size=mcs,
                min_samples=max(1, mcs // 3),
                metric="euclidean",
                prediction_data=True,
            )
            labels = hdbscan_model.fit_predict(reduced)
            n_outliers = (labels == -1).sum()
            outlier_pct = 100.0 * n_outliers / len(labels)
            n_topics = len(set(labels) - {-1})

            results.append({
                "min_cluster_size": mcs,
                "outlier_pct": outlier_pct,
                "n_topics": n_topics,
            })
            logger.info(
                f"  min_cluster_size={mcs:3d}: "
                f"{outlier_pct:5.1f}% outliers, {n_topics} topics"
            )

        # Pick closest to target, preferring 5-15% band
        results_df = pd.DataFrame(results)
        in_band = results_df[
            (results_df["outlier_pct"] >= 5) & (results_df["outlier_pct"] <= 15)
        ]

        if len(in_band) > 0:
            best_idx = (in_band["outlier_pct"] - target_outlier_pct).abs().idxmin()
        else:
            best_idx = (results_df["outlier_pct"] - target_outlier_pct).abs().idxmin()

        best = results_df.loc[best_idx]
        best_mcs = int(best["min_cluster_size"])

        logger.info(
            f"Selected min_cluster_size={best_mcs}: "
            f"{best['outlier_pct']:.1f}% outliers, {int(best['n_topics'])} topics"
        )

        # Update config so subsequent build_model() uses the tuned value
        self.config.hdbscan_min_cluster_size = best_mcs
        return best_mcs

    def get_topic_confidence(
        self,
        outlier_penalty: Optional[float] = None,
    ) -> np.ndarray:
        """
        Build per-document topic confidence from fitted model.

        Venues with strong topic assignment get high confidence.
        Ex-outlier venues (originally topic -1 before reduction) get
        penalized so downstream models (XGBoost) learn to rely on
        GNN embeddings rather than topic features for those venues.

        Must be called AFTER fit() or fit_transform_venues().

        Args:
            outlier_penalty: Max confidence for ex-outlier venues.
                Defaults to config.outlier_confidence.

        Returns:
            Confidence array [n_documents, 1].
        """
        if self.topics is None or self.probs is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if outlier_penalty is None:
            outlier_penalty = self.config.outlier_confidence

        n = len(self.topics)
        confidence = np.ones((n, 1), dtype=np.float32)

        # Get the topic distribution matrix
        if isinstance(self.probs, np.ndarray) and len(self.probs.shape) == 2:
            topic_distr = self.probs
        else:
            topic_distr = None

        # Track which docs were originally outliers
        was_outlier = getattr(self, "_original_outlier_mask", None)

        for i in range(n):
            is_ex_outlier = was_outlier[i] if was_outlier is not None else False

            if is_ex_outlier:
                # Ex-outlier: use max topic probability, capped
                if topic_distr is not None and topic_distr.shape[1] > 0:
                    max_prob = float(topic_distr[i].max())
                    confidence[i, 0] = min(max_prob, outlier_penalty)
                else:
                    confidence[i, 0] = outlier_penalty
            else:
                # Original topic: use the assigned topic's probability
                t = self.topics[i]
                if topic_distr is not None and 0 <= t < topic_distr.shape[1]:
                    confidence[i, 0] = float(topic_distr[i, t])
                else:
                    confidence[i, 0] = 1.0

        return confidence

    def fit(
        self,
        documents: list[str],
        embeddings: Optional[np.ndarray] = None,
        reduce_outliers: bool = True,
    ) -> tuple[list[int], np.ndarray]:
        """
        Fit the topic model on documents with optional outlier reduction.

        When reduce_outliers=True (default), documents assigned to
        topic -1 are reassigned to their closest topic using
        BERTopic's built-in approximate_distribution strategy
        followed by c-TF-IDF-based reassignment.

        Args:
            documents: List of document texts.
            embeddings: Optional pre-computed embeddings.
            reduce_outliers: Whether to reassign outlier documents.

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

        n_outliers_before = sum(1 for t in self.topics if t == -1)
        n_total = len(self.topics)
        logger.info(
            f"Initial fit: {len(set(self.topics)) - 1} topics, "
            f"{n_outliers_before}/{n_total} outliers "
            f"({100 * n_outliers_before / n_total:.1f}%)"
        )

        # Outlier reduction: reassign -1 documents
        if reduce_outliers and n_outliers_before > 0:
            self.topics, self.probs = self._reduce_outliers(
                documents, embeddings
            )
        else:
            # No reduction - mark current outliers for get_topic_confidence()
            self._original_outlier_mask = [t == -1 for t in self.topics]

        return self.topics, self.probs

    def _reduce_outliers(
        self,
        documents: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> tuple[list[int], np.ndarray]:
        """
        Reassign outlier documents (topic -1) to their closest topic.

        Uses a two-stage strategy:
          1. Embedding-based: cosine similarity to topic centroids
          2. c-TF-IDF-based: word distribution overlap with topics

        Documents that remain outliers after both stages keep topic -1.

        Args:
            documents: Document texts.
            embeddings: Pre-computed document embeddings.

        Returns:
            Updated (topics, probabilities).
        """
        topics = list(self.topics)
        n_before = sum(1 for t in topics if t == -1)

        # Store original outlier mask for get_topic_confidence()
        self._original_outlier_mask = [t == -1 for t in topics]

        # Stage 1: embedding-based reassignment
        try:
            new_topics = self.topic_model.reduce_outliers(
                documents,
                topics,
                strategy="embeddings",
                embeddings=embeddings,
                threshold=0.0,  # Accept any positive similarity
            )
            n_after_emb = sum(1 for t in new_topics if t == -1)
            logger.info(
                f"Outlier reduction (embeddings): "
                f"{n_before} -> {n_after_emb} outliers"
            )
            topics = new_topics
        except Exception as e:
            logger.warning(f"Embedding-based outlier reduction failed: {e}")

        # Stage 2: c-TF-IDF-based reassignment for remaining outliers
        n_still_outliers = sum(1 for t in topics if t == -1)
        if n_still_outliers > 0:
            try:
                new_topics = self.topic_model.reduce_outliers(
                    documents,
                    topics,
                    strategy="c-tf-idf",
                    threshold=0.0,
                )
                n_after_ctfidf = sum(1 for t in new_topics if t == -1)
                logger.info(
                    f"Outlier reduction (c-TF-IDF): "
                    f"{n_still_outliers} -> {n_after_ctfidf} outliers"
                )
                topics = new_topics
            except Exception as e:
                logger.warning(f"c-TF-IDF outlier reduction failed: {e}")

        # Update topic model's internal state
        self.topic_model.update_topics(documents, topics=topics)

        # Recompute probabilities via approximate_distribution
        try:
            topic_distr, _ = self.topic_model.approximate_distribution(
                documents, min_similarity=0.0
            )
            probs = topic_distr
            logger.info("Recomputed topic probabilities via approximate_distribution")
        except Exception as e:
            logger.warning(
                f"approximate_distribution failed ({e}), using hard assignments"
            )
            probs = self.probs

        n_final = sum(1 for t in topics if t == -1)
        logger.info(
            f"Outlier reduction complete: {n_before} -> {n_final} outliers "
            f"({100 * n_final / len(topics):.1f}%)"
        )

        self.topics = topics
        self.probs = probs
        return self.topics, self.probs

    def fit_transform_venues(
        self,
        venue_reviews: dict[str, list[str]],
        embeddings: Optional[np.ndarray] = None,
        reduce_outliers: bool = True,
    ) -> pd.DataFrame:
        """
        Fit topics on venue reviews and return venue-topic mapping.

        Args:
            venue_reviews: Dict mapping venue_id to list of reviews.
            embeddings: Optional pre-computed venue embeddings.
            reduce_outliers: Reassign outlier venues to closest topic.

        Returns:
            DataFrame with venue_id, topic, topic_prob, topic_name,
            and was_outlier (bool indicating if originally topic -1).
        """
        # Truncate reviews to prevent massive documents
        documents = [
            " ".join([str(r)[:500] for r in reviews[:5]])
            for reviews in venue_reviews.values()
        ]

        # Fit (outlier reduction happens inside fit())
        topics_before_reduction, _ = self.topic_model.fit_transform(
            documents, embeddings=embeddings
        ) if not reduce_outliers else (None, None)

        # Track original outliers BEFORE reduction
        if reduce_outliers:
            # Do initial fit without reduction to capture original assignments
            if self.topic_model is None:
                self.build_model()

            self.topics, self.probs = self.topic_model.fit_transform(
                documents, embeddings=embeddings
            )
            if embeddings is None and hasattr(self.topic_model, "embedding_model"):
                self.embeddings = self.topic_model._extract_embeddings(
                    documents, method="document"
                )
            else:
                self.embeddings = embeddings

            original_outlier_mask = [t == -1 for t in self.topics]

            # Now reduce outliers
            self.topics, self.probs = self._reduce_outliers(
                documents, embeddings
            )
            topics = self.topics
            probs = self.probs
        else:
            topics, probs = self.fit(
                documents, embeddings, reduce_outliers=False
            )
            original_outlier_mask = [t == -1 for t in topics]

        if probs is None:
            probs = [1.0] * len(topics)

        # Create result DataFrame
        venue_ids = list(venue_reviews.keys())
        result = pd.DataFrame({
            "venue_id": venue_ids,
            "topic": topics,
            "topic_prob": [
                p.max() if isinstance(p, np.ndarray) else float(p)
                for p in probs
            ],
            "was_outlier": original_outlier_mask,
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
    reduce_outliers: bool = True,
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
        reduce_outliers: Reassign outlier venues to closest topic.
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

    # Fit and get venue topics (with outlier reduction)
    venue_topics = extractor.fit_transform_venues(
        venue_reviews, reduce_outliers=reduce_outliers
    )

    # Save if path provided
    if save_path:
        extractor.save(save_path)
        venue_topics.to_parquet(
            save_path.parent / "venue_topics.parquet", index=False
        )

    return venue_topics, extractor
