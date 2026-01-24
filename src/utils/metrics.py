"""
Evaluation Metrics for Recommendation Systems.

Includes precision, recall, NDCG, MRR, hit rate, and calibration metrics.
"""
import numpy as np
from typing import Optional
from collections import defaultdict


class RecommendationMetrics:
    """Calculate recommendation system metrics."""

    @staticmethod
    def precision_at_k(
        recommended: list,
        relevant: set,
        k: int,
    ) -> float:
        """
        Calculate Precision@K.

        Args:
            recommended: Ordered list of recommended items.
            relevant: Set of relevant (ground truth) items.
            k: Number of top items to consider.

        Returns:
            Precision@K score.
        """
        if k <= 0:
            return 0.0

        top_k = recommended[:k]
        hits = sum(1 for item in top_k if item in relevant)
        return hits / k

    @staticmethod
    def recall_at_k(
        recommended: list,
        relevant: set,
        k: int,
    ) -> float:
        """
        Calculate Recall@K.

        Args:
            recommended: Ordered list of recommended items.
            relevant: Set of relevant (ground truth) items.
            k: Number of top items to consider.

        Returns:
            Recall@K score.
        """
        if not relevant:
            return 0.0

        top_k = recommended[:k]
        hits = sum(1 for item in top_k if item in relevant)
        return hits / len(relevant)

    @staticmethod
    def ndcg_at_k(
        recommended: list,
        relevant: set,
        k: int,
        relevance_scores: Optional[dict] = None,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.

        Args:
            recommended: Ordered list of recommended items.
            relevant: Set of relevant items.
            k: Number of top items to consider.
            relevance_scores: Optional dict of item -> relevance score.

        Returns:
            NDCG@K score.
        """
        if not relevant or k <= 0:
            return 0.0

        # DCG
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
                dcg += rel / np.log2(i + 2)  # i+2 because positions are 1-indexed

        # Ideal DCG
        if relevance_scores:
            ideal_rels = sorted(
                [relevance_scores.get(item, 1.0) for item in relevant],
                reverse=True,
            )[:k]
        else:
            ideal_rels = [1.0] * min(len(relevant), k)

        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def mrr(recommended: list, relevant: set) -> float:
        """
        Calculate Mean Reciprocal Rank.

        Args:
            recommended: Ordered list of recommended items.
            relevant: Set of relevant items.

        Returns:
            Reciprocal rank (1/rank of first relevant item).
        """
        for i, item in enumerate(recommended):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def hit_rate_at_k(
        recommended: list,
        relevant: set,
        k: int,
    ) -> float:
        """
        Calculate Hit Rate@K (1 if any relevant item in top-k, else 0).

        Args:
            recommended: Ordered list of recommended items.
            relevant: Set of relevant items.
            k: Number of top items to consider.

        Returns:
            1.0 if hit, 0.0 otherwise.
        """
        top_k = set(recommended[:k])
        return 1.0 if top_k & relevant else 0.0

    @staticmethod
    def coverage(
        all_recommendations: list[list],
        catalog: set,
    ) -> float:
        """
        Calculate catalog coverage.

        Args:
            all_recommendations: List of recommendation lists for all users.
            catalog: Set of all items in the catalog.

        Returns:
            Fraction of catalog items recommended at least once.
        """
        if not catalog:
            return 0.0

        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)

        return len(recommended_items & catalog) / len(catalog)

    @staticmethod
    def intra_list_diversity(
        recommended: list,
        item_embeddings: dict,
    ) -> float:
        """
        Calculate intra-list diversity (average pairwise distance).

        Args:
            recommended: List of recommended items.
            item_embeddings: Dict of item -> embedding vector.

        Returns:
            Average pairwise cosine distance.
        """
        if len(recommended) < 2:
            return 0.0

        embeddings = []
        for item in recommended:
            if item in item_embeddings:
                embeddings.append(np.array(item_embeddings[item]))

        if len(embeddings) < 2:
            return 0.0

        # Calculate pairwise cosine distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                )
                distances.append(1 - sim)  # Convert similarity to distance

        return np.mean(distances)

    @staticmethod
    def expected_calibration_error(
        predictions: list[float],
        labels: list[int],
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        Args:
            predictions: List of predicted probabilities.
            labels: List of binary labels.
            n_bins: Number of bins for calibration.

        Returns:
            ECE score.
        """
        predictions = np.array(predictions)
        labels = np.array(labels)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            bin_count = bin_mask.sum()

            if bin_count > 0:
                bin_accuracy = labels[bin_mask].mean()
                bin_confidence = predictions[bin_mask].mean()
                ece += (bin_count / len(predictions)) * abs(bin_accuracy - bin_confidence)

        return ece

    def evaluate_all(
        self,
        user_recommendations: dict[str, list],
        user_relevant: dict[str, set],
        k_values: list[int] = [5, 10, 20],
        item_embeddings: Optional[dict] = None,
        catalog: Optional[set] = None,
    ) -> dict:
        """
        Evaluate all metrics for a set of recommendations.

        Args:
            user_recommendations: Dict of user_id -> recommended items.
            user_relevant: Dict of user_id -> relevant items.
            k_values: List of K values to evaluate.
            item_embeddings: Optional embeddings for diversity.
            catalog: Optional full catalog for coverage.

        Returns:
            Dictionary of metric results.
        """
        results = defaultdict(list)

        for user_id, recommended in user_recommendations.items():
            relevant = user_relevant.get(user_id, set())

            for k in k_values:
                results[f"precision@{k}"].append(
                    self.precision_at_k(recommended, relevant, k)
                )
                results[f"recall@{k}"].append(
                    self.recall_at_k(recommended, relevant, k)
                )
                results[f"ndcg@{k}"].append(
                    self.ndcg_at_k(recommended, relevant, k)
                )
                results[f"hit_rate@{k}"].append(
                    self.hit_rate_at_k(recommended, relevant, k)
                )

            results["mrr"].append(self.mrr(recommended, relevant))

            if item_embeddings:
                results["diversity"].append(
                    self.intra_list_diversity(recommended, item_embeddings)
                )

        # Average all metrics
        averaged = {k: np.mean(v) for k, v in results.items()}

        # Add coverage if catalog provided
        if catalog:
            all_recs = list(user_recommendations.values())
            averaged["coverage"] = self.coverage(all_recs, catalog)

        return averaged
