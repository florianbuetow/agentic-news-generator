#!/usr/bin/env python3
# ruff: noqa: N803, N806
"""Train a Support Vector Machine (SVM) classifier to distinguish hallucinations from normal speech.

This script:
1. Loads the test examples CSV dataset
2. Trains a soft-margin linear SVM model using repetitions and sequence_length
3. Performs cross-validation to evaluate model performance
4. Visualizes the decision boundary
5. Saves the trained model to disk
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset from CSV.

    Args:
        csv_path: Path to CSV file.

    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels.
    """
    df = pd.read_csv(csv_path)

    # Extract features: repetitions and sequence_length
    X = df[["repetitions", "sequence_length"]].values

    # Extract labels: is_hallucination (0 or 1)
    y = df["is_hallucination"].values

    return X, y


def train_model(X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> tuple[SVC, float, float]:
    """Train Support Vector Machine (SVM) model with cross-validation.

    Args:
        X: Feature matrix (n_samples, 2).
        y: Labels (n_samples,).
        cv_folds: Number of cross-validation folds.

    Returns:
        Tuple of (trained_model, mean_cv_score, std_cv_score).
    """
    # Initialize SVM (soft-margin linear classifier)
    # C=1.0 is the regularization parameter (smaller C = softer margin)
    model = SVC(kernel="linear", C=1.0, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")

    print(f"Cross-validation scores ({cv_folds} folds):")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.3f}")
    print(f"  Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print()

    # Train on full dataset
    model.fit(X, y)

    return model, cv_scores.mean(), cv_scores.std()


def plot_decision_boundary(
    model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
) -> None:
    """Plot decision boundary and data points.

    Args:
        model: Trained SVM model.
        X: Feature matrix (n_samples, 2).
        y: Labels (n_samples,).
        output_path: Path to save the plot.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Separate data by class
    X_normal = X[y == 0]
    X_hallucination = X[y == 1]

    # Plot data points
    ax.scatter(
        X_normal[:, 0],
        X_normal[:, 1],
        c="blue",
        marker="o",
        s=100,
        alpha=0.6,
        edgecolors="black",
        label="Normal Speech (0)",
    )
    ax.scatter(
        X_hallucination[:, 0],
        X_hallucination[:, 1],
        c="red",
        marker="^",
        s=150,
        alpha=0.8,
        edgecolors="black",
        label="Hallucination (1)",
    )

    # Create mesh for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary (contour line where prediction changes)
    ax.contourf(xx, yy, Z, alpha=0.2, levels=[-0.5, 0.5, 1.5], colors=["blue", "red"])
    ax.contour(
        xx,
        yy,
        Z,
        levels=[0.5],
        colors="black",
        linewidths=2,
        linestyles="--",
        label="Decision Boundary",
    )

    # Get model coefficients for annotation
    coef = model.coef_[0]
    intercept = model.intercept_[0]

    # Add model equation to plot
    equation_text = f"Decision Boundary:\n{coef[0]:.3f} × repetitions + {coef[1]:.3f} × sequence_length + {intercept:.3f} = 0"
    ax.text(
        0.02,
        0.98,
        equation_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Labels and title
    ax.set_xlabel("Repetitions (Number of Consecutive Repetitions)", fontsize=12)
    ax.set_ylabel("Sequence Length (k = words in repeated phrase)", fontsize=12)
    ax.set_title(
        "Support Vector Machine: Hallucination Detection Decision Boundary\nUsing Soft-Margin Linear SVM (C=1.0) with Cross-Validation",
        fontsize=14,
        fontweight="bold",
    )

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved: {output_path}")

    # Also display if running interactively
    # plt.show()


def generate_standalone_classifier(
    model: SVC,
    output_path: Path,
    model_type: str,
    cv_accuracy: float,
    train_accuracy: float,
    n_support_vectors: int,
) -> None:
    """Generate standalone classifier without ML dependencies.

    Args:
        model: Trained SVM model.
        output_path: Path to save the generated Python file.
        model_type: Type of model (for documentation).
        cv_accuracy: Cross-validation accuracy.
        train_accuracy: Training accuracy.
        n_support_vectors: Number of support vectors.
    """
    coef = model.coef_[0]
    intercept = model.intercept_[0]

    code = f'''"""Standalone Hallucination Classifier - SVM Model

Auto-generated from trained Support Vector Machine (SVM) model.
No ML library dependencies required for inference.

Model Performance:
- Cross-validation accuracy: {cv_accuracy:.1%}
- Training accuracy: {train_accuracy:.1%}
- Model type: {model_type}
- Support vectors: {n_support_vectors}

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
        self.coef_repetitions = {coef[0]:.10f}
        self.coef_sequence_length = {coef[1]:.10f}
        self.intercept = {intercept:.10f}

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
        print(f"{{result}} | reps={{reps:2d}} len={{seq_len:2d}} | "
              f"prob={{prob:.3f}} score={{score:+.2f}} | {{description}}")
'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"Standalone classifier generated: {output_path}")


def save_model(model: SVC, output_path: Path) -> None:
    """Save trained model to disk.

    Args:
        model: Trained model.
        output_path: Path to save the model.
    """
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {output_path}")


def main() -> int:
    """Main entry point."""
    # Paths
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "analyze" / "hallucinations" / "test_examples_analysis.csv"
    model_path = Path(__file__).parent / "hallucination_classifier_svm.pkl"
    plot_path = Path(__file__).parent / "hallucination_classifier_svm_decision_boundary.png"
    standalone_path = Path(__file__).parent / "hallucination_classifier_svm.py"

    print("=" * 70)
    print("Hallucination Classifier Training (SVM)")
    print("=" * 70)
    print()

    # Load dataset
    print(f"Loading dataset: {csv_path}")
    X, y = load_dataset(csv_path)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]} (repetitions, sequence_length)")
    print(f"  Hallucinations: {y.sum()} ({y.sum() / len(y) * 100:.1f}%)")
    print(f"  Normal speech: {(1 - y).sum()} ({(1 - y).sum() / len(y) * 100:.1f}%)")
    print()

    # Train model with cross-validation
    print("Training Support Vector Machine (SVM) model...")
    print("  Kernel: linear")
    print("  C parameter: 1.0 (soft margin)")
    model, mean_score, std_score = train_model(X, y, cv_folds=5)

    # Get training accuracy
    train_accuracy = model.score(X, y)
    print(f"Training accuracy: {train_accuracy:.3f}")
    print()

    # Display model parameters
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    print("Model Parameters:")
    print(f"  Coefficient (repetitions): {coef[0]:.6f}")
    print(f"  Coefficient (sequence_length): {coef[1]:.6f}")
    print(f"  Intercept: {intercept:.6f}")
    print()

    # Display support vectors info
    print("Support Vector Information:")
    print(f"  Number of support vectors: {model.n_support_.sum()}")
    print(f"  Support vectors per class: {model.n_support_}")
    print()

    # Decision boundary equation
    print("Decision Boundary Equation:")
    print(f"  {coef[0]:.6f} × repetitions + {coef[1]:.6f} × sequence_length + {intercept:.6f} = 0")
    print()

    # Interpretation
    print("Interpretation:")
    if coef[0] > 0:
        print("  • More repetitions → higher hallucination probability")
    else:
        print("  • More repetitions → lower hallucination probability")

    if coef[1] > 0:
        print("  • Longer sequences → higher hallucination probability")
    else:
        print("  • Longer sequences → lower hallucination probability")
    print()

    # Save model
    print("Saving model...")
    save_model(model, model_path)
    print()

    # Generate standalone classifier
    print("Generating standalone classifier (no ML dependencies)...")
    generate_standalone_classifier(
        model,
        standalone_path,
        "Support Vector Machine (Linear, C=1.0)",
        mean_score,
        train_accuracy,
        model.n_support_.sum(),
    )
    print()

    # Generate visualization
    print("Generating decision boundary visualization...")
    plot_decision_boundary(model, X, y, plot_path)
    print()

    # Summary
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Cross-validation accuracy: {mean_score:.3f} (+/- {std_score:.3f})")
    print(f"Training accuracy: {train_accuracy:.3f}")
    print()
    print("Output files:")
    print(f"  • Model: {model_path}")
    print(f"  • Standalone Classifier: {standalone_path}")
    print(f"  • Visualization: {plot_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
