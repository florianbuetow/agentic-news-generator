"""Standalone Hallucination Classifier - Quadratic Model

Auto-generated from trained logistic regression model with polynomial features (degree=2).
No ML library dependencies required for inference.

Model Performance:
- Cross-validation accuracy: 97.1%
- Training accuracy: 100.0%
- Model type: Logistic Regression with Polynomial Features (degree=2)

Usage:
    from hallucination_classifier_quadratic import HallucinationClassifier

    classifier = HallucinationClassifier()
    is_hallucination = classifier.predict(repetitions=15, sequence_length=3)
    # Returns: True
"""

import math


class HallucinationClassifier:
    """Standalone hallucination classifier using quadratic decision boundary.

    This classifier uses a quadratic function (polynomial degree=2) to determine if
    a pattern of repetitions constitutes a hallucination. The decision boundary is a
    curve rather than a straight line, allowing for more complex separation.

    No external dependencies required.
    """

    def __init__(self) -> None:
        """Initialize classifier with trained coefficients."""
        # Coefficients from trained quadratic model
        # Polynomial features: [1, rep, len, rep^2, rep*len, len^2]
        self.coef_bias = -0.0005429170
        self.coef_repetitions = 0.0380797988
        self.coef_sequence_length = 0.0280639033
        self.coef_repetitions_squared = 0.2609198746
        self.coef_interaction = 0.3383034829  # rep * len
        self.coef_sequence_length_squared = 0.1560897638
        self.intercept = -15.5181124275

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
        """Compute raw quadratic decision function score.

        Decision function:
            score = coef[3]*rep² + coef[5]*len² + coef[4]*rep*len
                  + coef[1]*rep + coef[2]*len + coef[0] + intercept

        Args:
            repetitions: Number of consecutive repetitions.
            sequence_length: Length of repeated phrase (k = words).

        Returns:
            Decision score (positive = hallucination, negative = normal).
        """
        # Compute polynomial features
        rep = repetitions
        len_val = sequence_length
        rep_squared = repetitions ** 2
        len_squared = sequence_length ** 2
        interaction = repetitions * sequence_length

        # Compute decision function
        score = (
            self.coef_bias
            + self.coef_repetitions * rep
            + self.coef_sequence_length * len_val
            + self.coef_repetitions_squared * rep_squared
            + self.coef_interaction * interaction
            + self.coef_sequence_length_squared * len_squared
            + self.intercept
        )

        return score

    def predict_proba(self, repetitions: float, sequence_length: float) -> float:
        """Compute probability of hallucination using sigmoid function.

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

    print("Hallucination Classifier (Quadratic) - Test Cases")
    print("=" * 60)
    for reps, seq_len, description in test_cases:
        is_hall = classifier.predict(reps, seq_len)
        prob = classifier.predict_proba(reps, seq_len)
        score = classifier.decision_function(reps, seq_len)

        result = "✓ HALLUCINATION" if is_hall else "○ NORMAL"
        print(f"{result} | reps={reps:2d} len={seq_len:2d} | "
              f"prob={prob:.3f} score={score:+.2f} | {description}")
