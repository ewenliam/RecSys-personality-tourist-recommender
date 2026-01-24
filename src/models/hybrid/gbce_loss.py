"""
Generalized Binary Cross-Entropy (gBCE) Loss.

Implements calibrated loss functions to reduce overconfidence in
recommendation predictions. Based on the RecSys 2023 best paper.

Reference:
    "Reducing Overconfidence in Sequential Recommendation
    Trained with Negative Sampling" RecSys 2023
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class gBCELoss:
    """
    Generalized Binary Cross-Entropy Loss.

    Adds temperature-based calibration to reduce overconfidence.
    Lower temperature = stronger penalty for overconfident predictions.
    """

    def __init__(self, t: float = 0.8, eps: float = 1e-7):
        """
        Initialize gBCE loss.

        Args:
            t: Temperature parameter (0 < t <= 1).
               t = 1.0: standard BCE
               t < 1.0: penalizes overconfidence
            eps: Small constant for numerical stability.
        """
        if not 0 < t <= 1:
            raise ValueError("Temperature t must be in (0, 1]")

        self.t = t
        self.eps = eps

    def __call__(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """
        Compute gBCE loss.

        Args:
            predictions: Predicted probabilities [N].
            targets: Binary targets [N].

        Returns:
            Scalar loss value.
        """
        # Clip predictions for stability
        predictions = np.clip(predictions, self.eps, 1 - self.eps)

        # Apply temperature scaling
        calibrated_preds = np.power(predictions, 1.0 / self.t)
        calibrated_preds = np.clip(calibrated_preds, self.eps, 1 - self.eps)

        # Standard BCE on calibrated predictions
        loss = -(
            targets * np.log(calibrated_preds) +
            (1 - targets) * np.log(1 - calibrated_preds)
        )

        return np.mean(loss)

    def gradient(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gradient of gBCE loss.

        Args:
            predictions: Predicted probabilities [N].
            targets: Binary targets [N].

        Returns:
            Gradient array [N].
        """
        predictions = np.clip(predictions, self.eps, 1 - self.eps)

        # Derivative includes temperature scaling
        calibrated = np.power(predictions, 1.0 / self.t)
        calibrated = np.clip(calibrated, self.eps, 1 - self.eps)

        # Chain rule: d(loss)/d(pred) = d(loss)/d(calibrated) * d(calibrated)/d(pred)
        d_calibrated = -(targets / calibrated - (1 - targets) / (1 - calibrated))
        d_pred = (1.0 / self.t) * np.power(predictions, 1.0 / self.t - 1)

        return d_calibrated * d_pred


class FocalLoss:
    """
    Focal Loss for handling class imbalance.

    Reduces loss contribution from easy examples, focusing
    training on hard negatives.

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-7):
        """
        Initialize focal loss.

        Args:
            gamma: Focusing parameter. Higher = more focus on hard examples.
            alpha: Weighting factor for positive class.
            eps: Small constant for numerical stability.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def __call__(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """
        Compute focal loss.

        Args:
            predictions: Predicted probabilities [N].
            targets: Binary targets [N].

        Returns:
            Scalar loss value.
        """
        predictions = np.clip(predictions, self.eps, 1 - self.eps)

        # Focal weighting
        pt = np.where(targets == 1, predictions, 1 - predictions)
        focal_weight = np.power(1 - pt, self.gamma)

        # Alpha weighting
        alpha_weight = np.where(targets == 1, self.alpha, 1 - self.alpha)

        # Cross-entropy
        ce = -(
            targets * np.log(predictions) +
            (1 - targets) * np.log(1 - predictions)
        )

        loss = alpha_weight * focal_weight * ce

        return np.mean(loss)


class CalibrationMetrics:
    """
    Metrics for evaluating prediction calibration.
    """

    @staticmethod
    def expected_calibration_error(
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Measures how well predicted probabilities match actual outcomes.

        Args:
            predictions: Predicted probabilities [N].
            targets: Binary targets [N].
            n_bins: Number of probability bins.

        Returns:
            ECE value (lower is better).
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(predictions)

        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            count = mask.sum()

            if count > 0:
                bin_accuracy = targets[mask].mean()
                bin_confidence = predictions[mask].mean()
                ece += (count / total) * abs(bin_accuracy - bin_confidence)

        return ece

    @staticmethod
    def maximum_calibration_error(
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        Maximum gap between predicted and actual probabilities.

        Args:
            predictions: Predicted probabilities [N].
            targets: Binary targets [N].
            n_bins: Number of probability bins.

        Returns:
            MCE value (lower is better).
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            count = mask.sum()

            if count > 0:
                bin_accuracy = targets[mask].mean()
                bin_confidence = predictions[mask].mean()
                mce = max(mce, abs(bin_accuracy - bin_confidence))

        return mce

    @staticmethod
    def reliability_diagram(
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute data for reliability diagram.

        Args:
            predictions: Predicted probabilities [N].
            targets: Binary targets [N].
            n_bins: Number of probability bins.

        Returns:
            Tuple of (bin_centers, bin_accuracies, bin_counts).
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        bin_accuracies = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            count = mask.sum()
            bin_counts[i] = count

            if count > 0:
                bin_accuracies[i] = targets[mask].mean()

        return bin_centers, bin_accuracies, bin_counts


def calibrate_predictions(
    predictions: np.ndarray,
    method: str = "temperature",
    temperature: float = 0.8,
    targets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calibrate prediction scores.

    Args:
        predictions: Raw prediction scores [N].
        method: Calibration method ("temperature", "platt", "isotonic").
        temperature: Temperature for temperature scaling.
        targets: Optional targets for fitting calibrators.

    Returns:
        Calibrated predictions.
    """
    if method == "temperature":
        # Temperature scaling
        calibrated = np.power(predictions, 1.0 / temperature)
        return np.clip(calibrated, 0, 1)

    elif method == "platt":
        # Platt scaling (logistic regression)
        if targets is None:
            raise ValueError("Platt scaling requires targets")

        from sklearn.linear_model import LogisticRegression

        # Fit logistic regression on logits
        logits = np.log(predictions / (1 - predictions + 1e-7))
        lr = LogisticRegression()
        lr.fit(logits.reshape(-1, 1), targets)

        calibrated = lr.predict_proba(logits.reshape(-1, 1))[:, 1]
        return calibrated

    elif method == "isotonic":
        # Isotonic regression
        if targets is None:
            raise ValueError("Isotonic calibration requires targets")

        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(predictions, targets)

        calibrated = ir.transform(predictions)
        return calibrated

    else:
        raise ValueError(f"Unknown calibration method: {method}")


def compute_overconfidence_penalty(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.9,
) -> float:
    """
    Compute penalty for overconfident predictions.

    Args:
        predictions: Predicted probabilities.
        targets: Binary targets.
        threshold: Confidence threshold.

    Returns:
        Overconfidence penalty score.
    """
    # Find overconfident predictions
    high_conf_mask = predictions > threshold
    low_conf_mask = predictions < (1 - threshold)

    # Check if overconfident predictions are wrong
    high_conf_wrong = high_conf_mask & (targets == 0)
    low_conf_wrong = low_conf_mask & (targets == 1)

    overconfident = high_conf_wrong | low_conf_wrong
    penalty = overconfident.sum() / (len(predictions) + 1e-8)

    return penalty
