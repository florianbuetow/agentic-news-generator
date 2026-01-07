"""Standalone Hallucination Classifier - SVM Model

Auto-generated from trained Support Vector Machine (SVM) model.
No ML library dependencies required for inference.

Model Performance:
- Cross-validation accuracy: 94.3%
- Training accuracy: 100.0%
- Model type: Support Vector Machine (Linear, C=1.0)
- Support vectors: 3

Usage:
    from hallucination_classifier_svm import HallucinationClassifier

    classifier = HallucinationClassifier()
    is_hallucination = classifier.predict(repetitions=15, sequence_length=3)
    # Returns: True
"""

import math


class HallucinationClassifier:
    """Standalone hallucination classifier using SVM linear decision boundary.

    This classifier uses the decision function learned by a Support Vector Machine
    with soft margin (C=1.0). No external dependencies required.
    """

    def __init__(self) -> None:
        """Initialize classifier with trained coefficients."""
        # Coefficients from trained SVM model
        self.coef_repetitions = 0.8888461538
        self.coef_sequence_length = 0.6665384615
        self.intercept = -6.7770512821

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
        return (
            self.coef_repetitions * repetitions
            + self.coef_sequence_length * sequence_length
            + self.intercept
        )

    def predict_proba(self, repetitions: float, sequence_length: float) -> float:
        """Compute probability of hallucination using sigmoid function.

        Note: SVM doesn't natively provide probabilities, but we can approximate
        using sigmoid of the decision function.

        Args:
            repetitions: Number of consecutive repetitions.
            sequence_length: Length of repeated phrase (k = words).

        Returns:
            Probability between 0 and 1 (closer to 1 = more likely hallucination).
        """
        score = self.decision_function(repetitions, sequence_length)
        # Sigmoid function: 1 / (1 + e^(-score))
        return 1.0 / (1.0 + math.exp(-score))


# Example usage
if __name__ == "__main__":
    classifier = HallucinationClassifier()

    # Test cases
    test_cases = [
        (15, 3, "Massive loop: 'you know, like,' × 15"),
        (2, 3, "Natural stutter: 2x repetition"),
        (11, 1, "Single word × 11"),
        (5, 1, "Single word × 5 (borderline)"),
    ]

    print("Hallucination Classifier (SVM) - Test Cases")
    print("=" * 60)
    for reps, seq_len, description in test_cases:
        is_hall = classifier.predict(reps, seq_len)
        prob = classifier.predict_proba(reps, seq_len)
        score = classifier.decision_function(reps, seq_len)

        result = "✓ HALLUCINATION" if is_hall else "○ NORMAL"
        print(f"{result} | reps={reps:2d} len={seq_len:2d} | "
              f"prob={prob:.3f} score={score:+.2f} | {description}")
