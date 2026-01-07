#!/usr/bin/env python3
# ruff: noqa: N803, N806
"""Train a quadratic classifier to distinguish hallucinations from normal speech.

This script:
1. Loads the test examples CSV dataset
2. Transforms features using polynomial features (degree=2) for quadratic decision boundary
3. Trains a logistic regression model on the transformed features
4. Performs cross-validation to evaluate model performance
5. Visualizes the quadratic decision boundary
6. Saves the trained model to disk
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


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


def train_model(X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> tuple[PolynomialFeatures, LogisticRegression, float, float]:
    """Train quadratic classifier with cross-validation.

    Args:
        X: Feature matrix (n_samples, 2).
        y: Labels (n_samples,).
        cv_folds: Number of cross-validation folds.

    Returns:
        Tuple of (poly_transformer, trained_model, mean_cv_score, std_cv_score).
    """
    # Create polynomial features (degree=2 for quadratic)
    # This creates: 1, x1, x2, x1^2, x1*x2, x2^2
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)

    # Initialize logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)

    # Perform cross-validation on transformed features
    cv_scores = cross_val_score(model, X_poly, y, cv=cv_folds, scoring="accuracy")

    print(f"Cross-validation scores ({cv_folds} folds):")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.3f}")
    print(f"  Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print()

    # Train on full dataset with polynomial features
    model.fit(X_poly, y)

    return poly, model, cv_scores.mean(), cv_scores.std()


def plot_decision_boundary(
    poly: PolynomialFeatures,
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
) -> None:
    """Plot quadratic decision boundary and data points.

    Args:
        poly: Polynomial feature transformer.
        model: Trained logistic regression model.
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

    # Transform mesh to polynomial features and predict
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_poly = poly.transform(mesh_points)
    Z = model.predict(mesh_poly)
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
    )

    # Get model coefficients for annotation
    coef = model.coef_[0]
    intercept = model.intercept_[0]

    # Polynomial features order: [1, x1, x2, x1^2, x1*x2, x2^2]
    # coef[0] = bias term coefficient (usually ~0 since we have intercept)
    # coef[1] = x1 (repetitions)
    # coef[2] = x2 (sequence_length)
    # coef[3] = x1^2
    # coef[4] = x1*x2
    # coef[5] = x2^2

    # Add model equation to plot
    equation_text = (
        f"Quadratic Decision Boundary:\n"
        f"{coef[3]:.3f}·rep² + {coef[5]:.3f}·len² + {coef[4]:.3f}·rep·len\n"
        f"+ {coef[1]:.3f}·rep + {coef[2]:.3f}·len + {intercept:.3f} = 0"
    )
    ax.text(
        0.02,
        0.98,
        equation_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )

    # Labels and title
    ax.set_xlabel("Repetitions (Number of Consecutive Repetitions)", fontsize=12)
    ax.set_ylabel("Sequence Length (k = words in repeated phrase)", fontsize=12)
    ax.set_title(
        "Quadratic Classifier: Hallucination Detection Decision Boundary\nUsing Polynomial Features (degree=2) with Logistic Regression",
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


def generate_standalone_classifier(
    poly: PolynomialFeatures,
    model: LogisticRegression,
    output_path: Path,
    model_type: str,
    cv_accuracy: float,
    train_accuracy: float,
) -> None:
    """Generate standalone classifier without ML dependencies.

    Args:
        poly: Polynomial feature transformer.
        model: Trained model.
        output_path: Path to save the generated Python file.
        model_type: Type of model (for documentation).
        cv_accuracy: Cross-validation accuracy.
        train_accuracy: Training accuracy.
    """
    coef = model.coef_[0]
    intercept = model.intercept_[0]

    # Polynomial features order: [1, x1, x2, x1^2, x1*x2, x2^2]
    # coef[0] = bias term coefficient
    # coef[1] = x1 (repetitions)
    # coef[2] = x2 (sequence_length)
    # coef[3] = x1^2
    # coef[4] = x1*x2
    # coef[5] = x2^2

    code = f'''"""Standalone Hallucination Classifier - Quadratic Model

Auto-generated from trained logistic regression model with polynomial features (degree=2).
No ML library dependencies required for inference.

Model Performance:
- Cross-validation accuracy: {cv_accuracy:.1%}
- Training accuracy: {train_accuracy:.1%}
- Model type: {model_type}

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
        self.coef_bias = {coef[0]:.10f}
        self.coef_repetitions = {coef[1]:.10f}
        self.coef_sequence_length = {coef[2]:.10f}
        self.coef_repetitions_squared = {coef[3]:.10f}
        self.coef_interaction = {coef[4]:.10f}  # rep * len
        self.coef_sequence_length_squared = {coef[5]:.10f}
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
        print(f"{{result}} | reps={{reps:2d}} len={{seq_len:2d}} | "
              f"prob={{prob:.3f}} score={{score:+.2f}} | {{description}}")
'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"Standalone classifier generated: {output_path}")


def save_model(poly: PolynomialFeatures, model: LogisticRegression, output_path: Path) -> None:
    """Save trained model and polynomial transformer to disk.

    Args:
        poly: Polynomial feature transformer.
        model: Trained model.
        output_path: Path to save the model.
    """
    # Save both the transformer and model together
    model_data = {"poly": poly, "model": model}
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model saved: {output_path}")


def main() -> int:
    """Main entry point."""
    # Paths
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "analyze" / "hallucinations" / "test_examples_analysis.csv"
    model_path = Path(__file__).parent / "hallucination_classifier_quadratic.pkl"
    plot_path = Path(__file__).parent / "hallucination_classifier_quadratic_decision_boundary.png"
    standalone_path = Path(__file__).parent / "hallucination_classifier_quadratic.py"

    print("=" * 70)
    print("Hallucination Classifier Training (Quadratic)")
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
    print("Training quadratic classifier...")
    print("  Polynomial degree: 2 (quadratic)")
    print("  Features: rep², len², rep·len, rep, len, bias")
    poly, model, mean_score, std_score = train_model(X, y, cv_folds=5)

    # Transform features for evaluation
    X_poly = poly.transform(X)

    # Get training accuracy
    train_accuracy = model.score(X_poly, y)
    print(f"Training accuracy: {train_accuracy:.3f}")
    print()

    # Display model parameters
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    print("Model Parameters:")
    print(f"  Polynomial features: {poly.n_output_features_}")
    print(f"  Feature names: {poly.get_feature_names_out(['rep', 'len'])}")
    print()
    print("Coefficients:")
    for i, name in enumerate(poly.get_feature_names_out(["rep", "len"])):
        print(f"  {name:12s}: {coef[i]:10.6f}")
    print(f"  {'intercept':12s}: {intercept:10.6f}")
    print()

    # Decision boundary equation
    print("Quadratic Decision Boundary Equation:")
    print(f"  {coef[3]:.6f}·rep² + {coef[5]:.6f}·len² + {coef[4]:.6f}·rep·len")
    print(f"  + {coef[1]:.6f}·rep + {coef[2]:.6f}·len + {intercept:.6f} = 0")
    print()

    # Interpretation
    print("Interpretation:")
    print("  • Quadratic terms allow curved decision boundary")
    print("  • Can capture non-linear relationships between features")
    print(f"  • rep² coefficient: {coef[3]:.6f}")
    print(f"  • len² coefficient: {coef[5]:.6f}")
    print(f"  • rep·len interaction: {coef[4]:.6f}")
    print()

    # Save model
    print("Saving model and polynomial transformer...")
    save_model(poly, model, model_path)
    print()

    # Generate standalone classifier
    print("Generating standalone classifier (no ML dependencies)...")
    generate_standalone_classifier(
        poly,
        model,
        standalone_path,
        "Logistic Regression with Polynomial Features (degree=2)",
        mean_score,
        train_accuracy,
    )
    print()

    # Generate visualization
    print("Generating quadratic decision boundary visualization...")
    plot_decision_boundary(poly, model, X, y, plot_path)
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
