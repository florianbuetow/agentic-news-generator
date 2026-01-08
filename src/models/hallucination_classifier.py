"""Hallucination classifier for detecting transcript hallucinations.

Loads SVM model configuration from Config object and provides classification
without requiring ML library dependencies at inference time.
"""

from src.config import Config


class HallucinationClassifier:
    """SVM-based hallucination classifier using trained model coefficients.

    Model coefficients are loaded from Config object.
    """

    def __init__(self, config: Config) -> None:
        """Initialize classifier from config.

        Args:
            config: Config object with hallucination detection settings.
        """
        # Load model coefficients from Config
        self.coef_repetitions, self.coef_sequence_length, self.intercept = config.getHallucinationClassifierModel()

    def predict(self, repetitions: float, sequence_length: float) -> bool:
        """Predict if input is a hallucination.

        Args:
            repetitions: Number of consecutive repetitions.
            sequence_length: Length of repeated phrase (k = words).

        Returns:
            True if classified as hallucination, False otherwise.
        """
        score = self.decision_function(repetitions, sequence_length)
        return score > 0

    def decision_function(self, repetitions: float, sequence_length: float) -> float:
        """Compute raw SVM decision function score.

        Args:
            repetitions: Number of consecutive repetitions.
            sequence_length: Length of repeated phrase (k = words).

        Returns:
            Decision score (positive = hallucination, negative = normal).
        """
        return self.coef_repetitions * repetitions + self.coef_sequence_length * sequence_length + self.intercept
