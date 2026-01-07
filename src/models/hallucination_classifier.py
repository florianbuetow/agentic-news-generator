"""Hallucination classifier for detecting transcript hallucinations.

Loads SVM model configuration from config.yaml and provides classification
without requiring ML library dependencies at inference time.
"""

from pathlib import Path

import yaml


class HallucinationClassifier:
    """SVM-based hallucination classifier using trained model coefficients.

    Model coefficients are loaded from config.yaml.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize classifier from config.

        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        # Load config
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        hallucination_config = config["hallucination_detection"]
        self.coef_repetitions: float = float(hallucination_config["coef_repetitions"])
        self.coef_sequence_length: float = float(hallucination_config["coef_sequence_length"])
        self.intercept: float = float(hallucination_config["intercept"])

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
