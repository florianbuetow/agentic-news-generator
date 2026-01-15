#!/usr/bin/env python3
"""
Model Performance Analysis for Shell Script Classification

This script analyzes the performance of different LLM models on shell script
classification tasks. Models classify scripts as PASS (no violations) or FAIL
(has environment variable violations or forbidden constructs).

The analysis generates comprehensive visualizations including:
- Performance metrics tables (accuracy, precision, recall, F1)
- Confusion matrices
- Misclassification patterns
- Model agreement heatmaps
- Additional insights (inter-model agreement, test difficulty, etc.)

All outputs are saved as high-resolution JPEG files for easy sharing.
"""

# =============================================================================
# SECTION: Imports and Configuration
# =============================================================================
# Import necessary libraries for data analysis and visualization

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Dict, List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration constants
REPORTS_DIR = Path("reports")
OUTPUT_DIR = Path("reports/visualizations")
CSV_PATTERN = "*.csv"
JPEG_DPI = 300
FIGSIZE_LARGE = (16, 10)
FIGSIZE_WIDE = (20, 8)
FIGSIZE_XLARGE = (24, 12)

# Modern Color Palette (consistent across all visualizations)
# Inspired by modern design systems with good contrast and accessibility
COLOR_PALETTE = {
    # Primary colors for main elements
    "primary": "#6366f1",      # Indigo
    "secondary": "#8b5cf6",    # Purple
    "accent": "#ec4899",       # Pink

    # Semantic colors
    "success": "#10b981",      # Green
    "error": "#ef4444",        # Red
    "warning": "#f59e0b",      # Amber
    "info": "#3b82f6",         # Blue

    # Chart-specific colors
    "false_positive": "#f87171",   # Light red (for FP bars)
    "false_negative": "#60a5fa",   # Light blue (for FN bars)

    # Gradient colors for heatmaps (light to dark)
    "heatmap_sequential": ["#ede9fe", "#c7d2fe", "#a5b4fc", "#818cf8", "#6366f1", "#4f46e5"],
    "heatmap_diverging": ["#ef4444", "#fca5a5", "#fef3c7", "#a7f3d0", "#34d399", "#10b981"],

    # Categorical palette for multiple models
    "categorical": ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#3b82f6", "#06b6d4", "#14b8a6"],
}

# Set seaborn style for better-looking plots
sns.set_theme(style="whitegrid")
sns.set_palette(COLOR_PALETTE["categorical"])
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = JPEG_DPI
plt.rcParams["font.size"] = 10


# =============================================================================
# SECTION: CSV Loading and Preprocessing
# =============================================================================
# Functions to load, validate, and preprocess model benchmark CSV files


def scan_csv_files(reports_dir: Path) -> List[Path]:
    """
    Scan the reports directory for CSV files containing model benchmarks.

    Args:
        reports_dir: Path to the reports directory

    Returns:
        List of Path objects for CSV files found

    Note:
        Excludes any CSV files that don't match the expected naming pattern
    """
    csv_files = list(reports_dir.glob(CSV_PATTERN))
    # Filter out any non-model CSV files (e.g., summary files, temp files)
    model_csvs = [
        f
        for f in csv_files
        if f.stem not in ["summary", "temp", "shell_env_detector_test_results"]
    ]
    return sorted(model_csvs)


def extract_model_name(filepath: Path) -> str:
    """
    Extract the model name from the CSV filename.

    Args:
        filepath: Path to the CSV file

    Returns:
        Model name (filename without .csv extension)

    Example:
        'qwen2.5-7b-instruct-mlx.csv' -> 'qwen2.5-7b-instruct-mlx'
    """
    return filepath.stem


def extract_timing_from_log(model_name: str, reports_dir: Path) -> int:
    """
    Extract total elapsed time from the model's log file.

    Args:
        model_name: Name of the model
        reports_dir: Path to the reports directory

    Returns:
        Total elapsed time in seconds, or -1 if not found

    The log file should contain a line like:
        "Total elapsed time: 112 seconds"
    """
    log_file = reports_dir / f"{model_name}.log"

    if not log_file.exists():
        return -1

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if "Total elapsed time:" in line:
                    # Extract the number from "Total elapsed time: 112 seconds"
                    parts = line.strip().split()
                    if len(parts) >= 4 and parts[-1] == "seconds":
                        return int(parts[-2])
    except Exception:
        return -1

    return -1


def validate_csv_format(df: pd.DataFrame, filepath: Path) -> None:
    """
    Validate that the CSV has the expected 5-column format.

    Args:
        df: DataFrame to validate
        filepath: Path to the file (for error messages)

    Raises:
        ValueError: If the CSV doesn't have the expected format

    Expected format: Script, Expected, Predicted, Match, Error
    Old format (4 columns) is no longer supported and must be converted.
    """
    expected_columns = ["Script", "Expected", "Predicted", "Match", "Error"]
    actual_columns = df.columns.tolist()

    if len(actual_columns) == 4:
        raise ValueError(
            f"ERROR: {filepath.name} uses the old 4-column format.\n"
            f"Expected columns: {expected_columns}\n"
            f"Actual columns: {actual_columns}\n\n"
            f"To convert, please:\n"
            f"1. Rename 'Actual' column to 'Predicted'\n"
            f"2. Add an 'Error' column (can be empty)\n"
            f"3. Ensure Script column contains full path to test files\n"
        )

    if actual_columns != expected_columns:
        raise ValueError(
            f"ERROR: {filepath.name} has unexpected column format.\n"
            f"Expected: {expected_columns}\n"
            f"Actual: {actual_columns}\n"
        )


def extract_script_basename(script_path: str) -> str:
    """
    Extract the basename from a script path.

    Args:
        script_path: Full or partial path to script

    Returns:
        Script basename (e.g., 'bad_export_cli_driven.sh')
    """
    return Path(script_path).name


def determine_test_type(script_name: str) -> str:
    """
    Determine if a test script is a 'bad' or 'good' example.

    Args:
        script_name: Name of the test script

    Returns:
        'bad' if script starts with 'bad_', 'good' if it starts with 'good_'

    Note:
        bad_*.sh files are expected to FAIL (have violations)
        good_*.sh files are expected to PASS (no violations)
    """
    if script_name.startswith("bad_"):
        return "bad"
    elif script_name.startswith("good_"):
        return "good"
    else:
        return "unknown"


def load_all_models() -> Dict[str, pd.DataFrame]:
    """
    Load all model CSV files from the reports directory.

    Returns:
        Dictionary mapping model names to their DataFrames

    The function:
    - Scans for CSV files in the reports directory
    - Validates the format of each CSV
    - Extracts test metadata (basename, test type)
    - Assigns test IDs for plotting
    - Returns a dictionary of DataFrames ready for analysis

    Raises:
        ValueError: If no CSV files are found or if format validation fails
    """
    csv_files = scan_csv_files(REPORTS_DIR)

    if not csv_files:
        raise ValueError(f"No CSV files found in {REPORTS_DIR}")

    print(f"\nFound {len(csv_files)} model CSV file(s) in {REPORTS_DIR}")

    model_data = {}

    for csv_file in csv_files:
        model_name = extract_model_name(csv_file)
        df = pd.read_csv(csv_file)

        # Validate format
        validate_csv_format(df, csv_file)

        # Add metadata columns
        df["script_basename"] = df["Script"].apply(extract_script_basename)
        df["test_type"] = df["script_basename"].apply(determine_test_type)

        # Assign test IDs (0-49 for bad tests, 50-99 for good tests)
        bad_mask = df["test_type"] == "bad"
        good_mask = df["test_type"] == "good"
        df.loc[bad_mask, "test_id"] = range(bad_mask.sum())
        df.loc[good_mask, "test_id"] = range(50, 50 + good_mask.sum())
        df["test_id"] = df["test_id"].astype(int)

        model_data[model_name] = df
        print(f"  Loaded: {model_name} ({len(df)} tests)")

    # Validate that all models have the same test scripts
    first_model = list(model_data.keys())[0]
    reference_scripts = set(model_data[first_model]["script_basename"])

    for model_name, df in model_data.items():
        model_scripts = set(df["script_basename"])
        if model_scripts != reference_scripts:
            missing = reference_scripts - model_scripts
            extra = model_scripts - reference_scripts
            print(f"\nWARNING: {model_name} has different test scripts:")
            if missing:
                print(f"  Missing: {missing}")
            if extra:
                print(f"  Extra: {extra}")

    return model_data


# =============================================================================
# SECTION: Metric Calculation
# =============================================================================
# Functions to calculate performance metrics (accuracy, precision, recall, F1)
# and confusion matrix values (TP, TN, FP, FN)


def calculate_confusion_matrix_values(df: pd.DataFrame) -> Tuple[int, int, int, int]:
    """
    Calculate confusion matrix values for a single model.

    Args:
        df: DataFrame with Expected and Predicted columns

    Returns:
        Tuple of (TP, TN, FP, FN) where:
        - TP (True Positive): Expected=PASS, Predicted=PASS (correctly identified clean script)
        - TN (True Negative): Expected=FAIL, Predicted=FAIL (correctly identified bad script)
        - FP (False Positive): Expected=FAIL, Predicted=PASS (incorrectly cleared a bad script)
        - FN (False Negative): Expected=PASS, Predicted=FAIL (incorrectly flagged a good script)

    Note:
        In our classification:
        - PASS = Positive class (no violations)
        - FAIL = Negative class (has violations)
    """
    # True Positives: Expected PASS, Predicted PASS
    tp = ((df["Expected"] == "PASS") & (df["Predicted"] == "PASS")).sum()

    # True Negatives: Expected FAIL, Predicted FAIL
    tn = ((df["Expected"] == "FAIL") & (df["Predicted"] == "FAIL")).sum()

    # False Positives: Expected FAIL, Predicted PASS (more concerning!)
    fp = ((df["Expected"] == "FAIL") & (df["Predicted"] == "PASS")).sum()

    # False Negatives: Expected PASS, Predicted FAIL
    fn = ((df["Expected"] == "PASS") & (df["Predicted"] == "FAIL")).sum()

    return tp, tn, fp, fn


def calculate_metrics(model_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate performance metrics for all models.

    Args:
        model_data: Dictionary mapping model names to DataFrames

    Returns:
        DataFrame with metrics for each model, sorted by F1 Score (descending)

    Metrics calculated:
    - Accuracy: (TP + TN) / Total
    - Precision: TP / (TP + FP) - how many predicted PASS are actually PASS
    - Recall: TP / (TP + FN) - how many actual PASS were predicted as PASS
    - F1 Score: harmonic mean of precision and recall
    - TP, TN, FP, FN: confusion matrix values

    Ranking by F1 Score:
    F1 is the balanced harmonic mean of precision and recall. While we're more
    concerned about false positives (FP), F1 gives us a single metric that
    considers both types of errors. A high F1 score means the model performs
    well at both identifying clean scripts (precision) and catching violations
    (recall).
    """
    metrics_list = []

    for model_name, df in model_data.items():
        # Filter out ERROR predictions for metric calculation
        df_clean = df[df["Predicted"] != "ERROR"].copy()

        if len(df_clean) == 0:
            print(f"WARNING: {model_name} has no valid predictions (all ERROR)")
            continue

        # Calculate confusion matrix values
        tp, tn, fp, fn = calculate_confusion_matrix_values(df_clean)

        # Calculate metrics using sklearn
        y_true = (df_clean["Expected"] == "PASS").astype(int)
        y_pred = (df_clean["Predicted"] == "PASS").astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0.0)
        recall = recall_score(y_true, y_pred, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)

        # Extract timing information from log file
        time_seconds = extract_timing_from_log(model_name, REPORTS_DIR)

        metrics_list.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Total": len(df_clean),
                "Time_s": time_seconds,
            }
        )

    # Create DataFrame and sort by F1 Score (best to worst)
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values("F1_Score", ascending=False).reset_index(
        drop=True
    )

    return metrics_df


# =============================================================================
# SECTION: Helper Functions for Efficiency Analysis
# =============================================================================
# Helper functions for Pareto front calculation and time formatting


def format_time_mmss(seconds: float) -> str:
    """
    Format seconds as mm:ss string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "2:15" for 2 minutes 15 seconds

    Example:
        >>> format_time_mmss(112)
        '1:52'
        >>> format_time_mmss(1491)
        '24:51'
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def calculate_pareto_front(metrics_df: pd.DataFrame) -> List[str]:
    """
    Identify models on the Pareto frontier for speed vs accuracy.

    A model is Pareto optimal if no other model is both faster AND more accurate.
    In other words, for a model to be on the Pareto front, any faster model must
    have lower accuracy, and any more accurate model must be slower.

    Args:
        metrics_df: DataFrame with Time_s and F1_Score columns

    Returns:
        List of model names on the Pareto front

    Example:
        If Model A is faster and more accurate than Model B, then Model B is
        dominated and NOT on the Pareto front.
    """
    df = metrics_df[metrics_df["Time_s"] > 0].copy()

    if len(df) == 0:
        return []

    pareto_models = []

    for i, row in df.iterrows():
        is_dominated = False
        for j, other_row in df.iterrows():
            if i != j:
                # Other model is better if it's both faster AND more accurate
                if other_row["Time_s"] < row["Time_s"] and other_row["F1_Score"] > row["F1_Score"]:
                    is_dominated = True
                    break

        if not is_dominated:
            pareto_models.append(row["Model"])

    return pareto_models


# =============================================================================
# SECTION: Visualization 1 - Performance Metrics Table
# =============================================================================
# Create a comprehensive table showing all performance metrics for each model


def generate_metrics_table(metrics_df: pd.DataFrame) -> None:
    """
    Print a formatted table showing performance metrics for all models.

    Args:
        metrics_df: DataFrame containing metrics for all models

    Output:
        Prints formatted DataFrame to console

    The table shows:
    - Model name
    - Accuracy, Precision, Recall, F1 Score (formatted to 3 decimal places)
    - TP, TN, FP, FN counts
    - Total tests and Time (seconds) from log file
    - Rows sorted by F1 Score (best model at top)
    """
    # Note: The metrics table is already printed in the main() function
    # This function is kept for consistency but doesn't generate an image anymore
    pass


# =============================================================================
# SECTION: Visualization 2 - Confusion Matrices
# =============================================================================
# Generate confusion matrices for each model in a grid layout


def generate_confusion_matrices(
    model_data: Dict[str, pd.DataFrame], metrics_df: pd.DataFrame
) -> None:
    """
    Generate a grid of confusion matrices for all models.

    Args:
        model_data: Dictionary mapping model names to DataFrames
        metrics_df: DataFrame containing metrics (for F1 scores in titles)

    Output:
        Saves grid as JPEG: reports/visualizations/02_confusion_matrices_grid.jpg

    Each confusion matrix shows:
    - 2x2 heatmap with counts
    - Percentages in cells
    - Model name and F1 score in title
    - Consistent color scale across all matrices
    """
    print("\nGenerating Visualization 2: Confusion Matrices...")

    # Sort models by F1 score (same order as metrics table)
    model_order = metrics_df["Model"].tolist()
    n_models = len(model_order)

    # Determine grid size
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Find global max for consistent color scale
    max_count = 0
    for model_name in model_order:
        df = model_data[model_name]
        df_clean = df[df["Predicted"] != "ERROR"]
        tp, tn, fp, fn = calculate_confusion_matrix_values(df_clean)
        max_count = max(max_count, tp, tn, fp, fn)

    # Generate confusion matrix for each model
    for idx, model_name in enumerate(model_order):
        ax = axes[idx]
        df = model_data[model_name]
        df_clean = df[df["Predicted"] != "ERROR"]

        # Calculate confusion matrix
        tp, tn, fp, fn = calculate_confusion_matrix_values(df_clean)
        cm = np.array([[tn, fp], [fn, tp]])

        # Get F1 score for title
        f1 = metrics_df[metrics_df["Model"] == model_name]["F1_Score"].values[0]

        # Create heatmap using modern color palette
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("custom", COLOR_PALETTE["heatmap_sequential"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            ax=ax,
            vmin=0,
            vmax=max_count,
            xticklabels=["FAIL", "PASS"],
            yticklabels=["FAIL", "PASS"],
        )

        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = (cm[i, j] / total) * 100
                ax.text(
                    j + 0.5,
                    i + 0.7,
                    f"({pct:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="gray",
                )

        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Expected", fontweight="bold")
        ax.set_title(f"{model_name}\nF1: {f1:.3f}", fontweight="bold")

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Confusion Matrices by Model", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save
    output_path = OUTPUT_DIR / "02_confusion_matrices_grid.jpg"
    plt.savefig(output_path, bbox_inches="tight", dpi=JPEG_DPI)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


# =============================================================================
# SECTION: Visualization 3 - Misclassification Analysis by Test
# =============================================================================
# Identify which tests are most frequently misclassified across all models


def generate_misclassification_analysis(model_data: Dict[str, pd.DataFrame]) -> None:
    """
    Generate side-by-side bar plots showing misclassification patterns.

    Args:
        model_data: Dictionary mapping model names to DataFrames

    Output:
        Saves plot as JPEG: reports/visualizations/03_misclassification_by_test.jpg

    The visualization shows:
    - Left: FAIL tests (bad_*.sh) - False positives
    - Right: PASS tests (good_*.sh) - False negatives
    - X-axis: Test ID
    - Y-axis: % of models that misclassified the test
    - Top 5 most misclassified tests are labeled with their names
    """
    print("\nGenerating Visualization 3: Misclassification Analysis...")

    # Aggregate misclassifications across all models
    all_data = []
    for model_name, df in model_data.items():
        df_copy = df.copy()
        df_copy["model"] = model_name
        df_copy["misclassified"] = df_copy["Match"] == "N"
        all_data.append(df_copy)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Count misclassifications per test
    n_models = len(model_data)
    misclass_counts = (
        combined_df.groupby(["test_id", "script_basename", "test_type", "Expected"])[
            "misclassified"
        ]
        .sum()
        .reset_index()
    )
    misclass_counts.columns = [
        "test_id",
        "script_basename",
        "test_type",
        "expected",
        "misclass_count",
    ]

    # Convert counts to percentages
    misclass_counts["misclass_pct"] = (
        misclass_counts["misclass_count"] / n_models
    ) * 100

    # Separate FAIL and PASS tests
    fail_tests = misclass_counts[misclass_counts["test_type"] == "bad"].copy()
    pass_tests = misclass_counts[misclass_counts["test_type"] == "good"].copy()

    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    # Plot 1: FAIL tests (False Positives)
    if not fail_tests.empty:
        ax1.bar(
            fail_tests["test_id"],
            fail_tests["misclass_pct"],
            color=COLOR_PALETTE["false_positive"],
            alpha=0.8,
            edgecolor="black",
        )
        ax1.set_xlabel("Test ID", fontweight="bold")
        ax1.set_ylabel("% of Models", fontweight="bold")
        ax1.set_title(
            "False Positives: Bad Scripts Predicted as PASS", fontweight="bold"
        )
        ax1.set_ylim(0, 100)
        ax1.grid(axis="y", alpha=0.3)

        # Label top 5 most misclassified
        top5_fail = fail_tests.nlargest(5, "misclass_pct")
        for _, row in top5_fail.iterrows():
            ax1.text(
                row["test_id"],
                row["misclass_pct"] + 2,
                row["script_basename"][:20],
                rotation=45,
                ha="left",
                va="bottom",
                fontsize=7,
            )

    # Plot 2: PASS tests (False Negatives)
    if not pass_tests.empty:
        ax2.bar(
            pass_tests["test_id"],
            pass_tests["misclass_pct"],
            color=COLOR_PALETTE["false_negative"],
            alpha=0.8,
            edgecolor="black",
        )
        ax2.set_xlabel("Test ID", fontweight="bold")
        ax2.set_ylabel("% of Models", fontweight="bold")
        ax2.set_title(
            "False Negatives: Good Scripts Predicted as FAIL", fontweight="bold"
        )
        ax2.set_ylim(0, 100)
        ax2.grid(axis="y", alpha=0.3)

        # Label top 5 most misclassified
        top5_pass = pass_tests.nlargest(5, "misclass_pct")
        for _, row in top5_pass.iterrows():
            ax2.text(
                row["test_id"],
                row["misclass_pct"] + 2,
                row["script_basename"][:20],
                rotation=45,
                ha="left",
                va="bottom",
                fontsize=7,
            )

    plt.suptitle(
        "Misclassification Patterns Across Models", fontsize=14, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = OUTPUT_DIR / "03_misclassification_by_test.jpg"
    plt.savefig(output_path, bbox_inches="tight", dpi=JPEG_DPI)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


# =============================================================================
# SECTION: Visualization 4 - Model-Test Agreement Heatmaps
# =============================================================================
# Show which models correctly/incorrectly classified each test


def generate_agreement_heatmaps(model_data: Dict[str, pd.DataFrame]) -> None:
    """
    Generate heatmaps showing model agreement on each test.

    Args:
        model_data: Dictionary mapping model names to DataFrames

    Output:
        Saves two heatmaps as JPEGs:
        - reports/visualizations/04a_heatmap_fail_tests.jpg
        - reports/visualizations/04b_heatmap_pass_tests.jpg

    Each heatmap shows:
    - X-axis: Test IDs
    - Y-axis: Model names
    - Color: Green = Correct, Red = Incorrect
    """
    print("\nGenerating Visualization 4: Model-Test Agreement Heatmaps...")

    # Create a matrix for each model's predictions
    models = sorted(model_data.keys())
    first_model_df = model_data[models[0]]

    # Separate FAIL and PASS tests
    fail_tests = first_model_df[first_model_df["test_type"] == "bad"].sort_values(
        "test_id"
    )
    pass_tests = first_model_df[first_model_df["test_type"] == "good"].sort_values(
        "test_id"
    )

    def create_heatmap(test_subset: pd.DataFrame, title: str, filename: str) -> None:
        """Helper function to create a single heatmap"""
        # Create matrix: models x tests
        matrix = []
        for model_name in models:
            df = model_data[model_name]
            # Match test order
            test_basenames = test_subset["script_basename"].tolist()
            model_tests = df[df["script_basename"].isin(test_basenames)].sort_values(
                "test_id"
            )
            # 1 = correct, 0 = incorrect
            row = [1 if m == "Y" else 0 for m in model_tests["Match"]]
            matrix.append(row)

        matrix = np.array(matrix)

        # Calculate figure size to make cells square
        # Each cell should be square, so height = (n_models / n_tests) * width
        n_models = len(models)
        n_tests = len(test_subset)
        cell_size = 0.35  # Size of each cell in inches
        fig_width = n_tests * cell_size
        fig_height = n_models * cell_size
        # Add extra space for labels and legend
        fig_width = max(fig_width, 12)  # Minimum width
        fig_height = max(fig_height, 3)  # Minimum height

        # Create figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create heatmap using modern color palette
        sns.heatmap(
            matrix,
            cmap=[COLOR_PALETTE["error"], COLOR_PALETTE["success"]],  # Red for incorrect, Green for correct
            cbar=False,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            xticklabels=test_subset["test_id"].tolist(),
            yticklabels=models,
            square=True,  # Make cells square
        )

        ax.set_xlabel("Test ID", fontweight="bold")
        ax.set_ylabel("Model", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=20)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=COLOR_PALETTE["success"], label="Correct"),
            Patch(facecolor=COLOR_PALETTE["error"], label="Incorrect"),
        ]
        ax.legend(
            handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1
        )

        plt.tight_layout()

        # Save
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, bbox_inches="tight", dpi=JPEG_DPI)
        plt.close()
        print(f"  ✓ Saved: {output_path.name}")

    # Generate both heatmaps
    create_heatmap(
        fail_tests,
        "Model-Test Agreement: FAIL Tests (bad_*.sh)",
        "04a_heatmap_fail_tests.jpg",
    )
    create_heatmap(
        pass_tests,
        "Model-Test Agreement: PASS Tests (good_*.sh)",
        "04b_heatmap_pass_tests.jpg",
    )


# =============================================================================
# SECTION: Visualization 5 - Additional Insights
# =============================================================================
# Additional visualizations for deeper analysis


def generate_additional_insights(
    model_data: Dict[str, pd.DataFrame], metrics_df: pd.DataFrame
) -> None:
    """
    Generate additional visualizations for deeper analysis.

    Args:
        model_data: Dictionary mapping model names to DataFrames
        metrics_df: DataFrame containing metrics for all models

    Output:
        Saves multiple JPEGs with additional insights
    """
    print("\nGenerating Visualization 5: Additional Insights...")

    # 5A: Inter-Model Agreement Matrix
    generate_model_agreement_matrix(model_data)

    # 5B: Precision-Recall Scatter Plot
    generate_precision_recall_scatter(metrics_df)

    # 5C: Performance Distribution
    generate_performance_distribution(model_data, metrics_df)


def generate_model_agreement_matrix(model_data: Dict[str, pd.DataFrame]) -> None:
    """
    Generate a heatmap showing pairwise agreement between models.

    Output:
        Saves as: reports/visualizations/05_model_agreement_matrix.jpg
    """
    models = sorted(model_data.keys())
    n_models = len(models)

    # Create agreement matrix
    agreement_matrix = np.zeros((n_models, n_models))

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i == j:
                agreement_matrix[i, j] = 100.0
            else:
                df1 = model_data[model1]
                df2 = model_data[model2]

                # Merge on script basename
                merged = pd.merge(
                    df1[["script_basename", "Predicted"]],
                    df2[["script_basename", "Predicted"]],
                    on="script_basename",
                    suffixes=("_1", "_2"),
                )

                # Calculate agreement percentage
                agreement = (merged["Predicted_1"] == merged["Predicted_2"]).sum()
                total = len(merged)
                agreement_pct = (agreement / total) * 100
                agreement_matrix[i, j] = agreement_pct

    # Create heatmap using modern color palette
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("custom", COLOR_PALETTE["heatmap_sequential"])

    fig, ax = plt.subplots(figsize=(max(10, n_models), max(8, n_models * 0.8)))
    sns.heatmap(
        agreement_matrix,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        xticklabels=models,
        yticklabels=models,
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Agreement (%)"},
        ax=ax,
    )
    ax.set_title("Inter-Model Agreement Matrix", fontweight="bold", pad=20)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "05_model_agreement_matrix.jpg"
    plt.savefig(output_path, bbox_inches="tight", dpi=JPEG_DPI)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def generate_precision_recall_scatter(metrics_df: pd.DataFrame) -> None:
    """
    Generate a scatter plot showing precision vs recall trade-offs.

    Output:
        Saves as: reports/visualizations/06_precision_recall_scatter.jpg
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot with F1 score as size
    # Use modern color palette for the scatter plot
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("custom", COLOR_PALETTE["heatmap_sequential"])

    sizes = metrics_df["F1_Score"] * 500  # Scale for visibility

    scatter = ax.scatter(
        metrics_df["Precision"],
        metrics_df["Recall"],
        s=sizes,
        alpha=0.6,
        c=metrics_df["F1_Score"],
        cmap=cmap,
        edgecolors="black",
        linewidth=1.5,
    )

    # Add model labels - positioned to the top left of each point
    for _, row in metrics_df.iterrows():
        ax.annotate(
            row["Model"],
            (row["Precision"], row["Recall"]),
            xytext=(-5, 5),
            textcoords="offset points",
            fontsize=8,
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
        )

    # Add diagonal line (precision = recall)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.3, label="Precision = Recall")

    ax.set_xlabel("Precision", fontweight="bold")
    ax.set_ylabel("Recall", fontweight="bold")
    ax.set_title("Precision vs Recall Trade-off", fontweight="bold", pad=20)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("F1 Score", fontweight="bold")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "06_precision_recall_scatter.jpg"
    plt.savefig(output_path, bbox_inches="tight", dpi=JPEG_DPI)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def generate_performance_distribution(
    model_data: Dict[str, pd.DataFrame], metrics_df: pd.DataFrame
) -> None:
    """
    Generate box plots showing performance distribution.

    Output:
        Saves as: reports/visualizations/07_performance_distribution.jpg
    """
    # Prepare data for box plot
    plot_data = []
    for model_name, df in model_data.items():
        df_clean = df[df["Predicted"] != "ERROR"]
        correct_pct = (df_clean["Match"] == "Y").sum() / len(df_clean) * 100
        plot_data.append({"Model": model_name, "Correct_Pct": correct_pct})

    plot_df = pd.DataFrame(plot_data)

    # Sort by F1 score order
    model_order = metrics_df["Model"].tolist()
    plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values("Model")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar plot using categorical colors from modern palette
    # Cycle through the categorical palette if we have more models than colors
    n_models = len(plot_df)
    colors = [COLOR_PALETTE["categorical"][i % len(COLOR_PALETTE["categorical"])] for i in range(n_models)]

    bars = ax.bar(
        range(len(plot_df)), plot_df["Correct_Pct"], color=colors, edgecolor="black"
    )

    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["Model"], rotation=45, ha="right")
    ax.set_ylabel("Correct Predictions (%)", fontweight="bold")
    ax.set_title("Model Performance Distribution", fontweight="bold", pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, plot_df["Correct_Pct"])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "07_performance_distribution.jpg"
    plt.savefig(output_path, bbox_inches="tight", dpi=JPEG_DPI)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


# =============================================================================
# SECTION: Efficiency Analysis Visualizations
# =============================================================================
# Visualizations focused on speed vs accuracy trade-offs


def generate_efficiency_scatter(metrics_df: pd.DataFrame, efficiency_df: pd.DataFrame) -> None:
    """
    Generate speed vs accuracy scatter plot with Pareto frontier.

    Args:
        metrics_df: DataFrame with all metrics
        efficiency_df: DataFrame with efficiency metrics (same as metrics_df in this case)

    Output:
        Saves as: reports/visualizations/08_efficiency_scatter.jpg

    The visualization shows:
    - Scatter plot of time vs F1 score
    - Pareto-optimal models marked with stars
    - Dashed line connecting Pareto front
    - Quadrant lines dividing fast/slow and accurate/less-accurate regions
    """
    df = efficiency_df.copy()
    pareto_models = calculate_pareto_front(df)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    for idx, row in df.iterrows():
        is_pareto = row["Model"] in pareto_models
        marker = "*" if is_pareto else "o"
        size = 300 if is_pareto else 150

        ax.scatter(
            row["Time_s"],
            row["F1_Score"],
            s=size,
            marker=marker,
            color=COLOR_PALETTE["categorical"][idx % len(COLOR_PALETTE["categorical"])],
            edgecolors="black",
            linewidth=2 if is_pareto else 1,
            alpha=0.8,
            label=row["Model"],
            zorder=10 if is_pareto else 5
        )

    # Draw Pareto front line
    pareto_df = df[df["Model"].isin(pareto_models)].sort_values("Time_s")
    if len(pareto_df) > 1:
        ax.plot(
            pareto_df["Time_s"],
            pareto_df["F1_Score"],
            "k--",
            alpha=0.4,
            linewidth=2,
            label="Pareto Front"
        )

    # Reference lines for quadrants
    mean_time = df["Time_s"].mean()
    mean_f1 = df["F1_Score"].mean()

    ax.axvline(mean_time, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(mean_f1, color="gray", linestyle=":", alpha=0.5)

    # Quadrant labels
    ax.text(mean_time * 0.5, df["F1_Score"].max() * 0.98,
            "Fast & Accurate", ha="center", fontsize=9, style="italic", alpha=0.6)
    ax.text(mean_time * 1.5, df["F1_Score"].max() * 0.98,
            "Slow & Accurate", ha="center", fontsize=9, style="italic", alpha=0.6)

    # Labels and formatting
    ax.set_xlabel("Execution Time (seconds)", fontweight="bold")
    ax.set_ylabel("F1 Score", fontweight="bold")
    ax.set_title("Speed vs Accuracy Trade-off Analysis", fontweight="bold", pad=20)

    # Use log scale if time range is large
    if df["Time_s"].max() / df["Time_s"].min() > 10:
        ax.set_xscale("log")
        ax.set_xlabel("Execution Time (seconds, log scale)", fontweight="bold")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "08_efficiency_scatter.jpg"
    plt.savefig(output_path, bbox_inches="tight", dpi=JPEG_DPI)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def generate_speed_comparison(metrics_df: pd.DataFrame) -> None:
    """
    Generate grouped bar chart comparing model execution speeds.

    Args:
        metrics_df: DataFrame with Time_s column

    Output:
        Saves as: reports/visualizations/09_speed_comparison.jpg

    The visualization shows:
    - Two bars per model (time per test and total time)
    - Y-axis in minutes
    - Annotations in mm:ss format
    - Models sorted by speed (fastest first)
    """
    # Filter models with timing data
    df = metrics_df[metrics_df["Time_s"] > 0].copy()

    if len(df) == 0:
        print("  ⚠ No timing data available for speed comparison")
        return

    # Sort by total time (fastest first)
    df = df.sort_values("Time_s")

    # Calculate metrics
    df["Time_Per_Test_Min"] = (df["Time_s"] / 100) / 60  # Convert to minutes
    df["Total_Time_Min"] = df["Time_s"] / 60  # Convert to minutes

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    x_positions = np.arange(len(df))
    bar_width = 0.35

    # Create bars
    bars1 = ax.bar(
        x_positions - bar_width/2,
        df["Time_Per_Test_Min"],
        bar_width,
        label="Time per Test",
        color=COLOR_PALETTE["primary"],
        edgecolor="black",
        alpha=0.9
    )

    bars2 = ax.bar(
        x_positions + bar_width/2,
        df["Total_Time_Min"],
        bar_width,
        label="Total Time (100 tests)",
        color=COLOR_PALETTE["secondary"],
        edgecolor="black",
        alpha=0.7
    )

    # Annotate bars with mm:ss format
    for i, (idx, row) in enumerate(df.iterrows()):
        # Time per test annotation
        time_per_test_str = format_time_mmss(row["Time_s"] / 100)
        ax.text(
            i - bar_width/2,
            row["Time_Per_Test_Min"] + max(df["Time_Per_Test_Min"]) * 0.02,
            time_per_test_str,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0
        )

        # Total time annotation
        total_time_str = format_time_mmss(row["Time_s"])
        ax.text(
            i + bar_width/2,
            row["Total_Time_Min"] + max(df["Total_Time_Min"]) * 0.02,
            total_time_str,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0
        )

    # Formatting
    ax.set_xlabel("Model", fontweight="bold", fontsize=12)
    ax.set_ylabel("Time (minutes)", fontweight="bold", fontsize=12)
    ax.set_title("Model Speed Comparison", fontweight="bold", fontsize=14, pad=20)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(df["Model"], rotation=45, ha="right")

    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "09_speed_comparison.jpg"
    plt.savefig(output_path, bbox_inches="tight", dpi=JPEG_DPI)
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


# =============================================================================
# SECTION: Main Execution
# =============================================================================
# Main function to orchestrate the entire analysis pipeline


def main() -> None:
    """
    Main execution function.

    This function orchestrates the entire analysis pipeline:
    1. Setup output directory
    2. Load all model CSV files
    3. Calculate performance metrics
    4. Generate all visualizations
    5. Print summary

    All outputs are saved to reports/visualizations/
    """
    print("=" * 80)
    print("Model Performance Analysis for Shell Script Classification")
    print("=" * 80)

    # Step 1: Setup
    print("\nStep 1: Setting up output directory...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Output directory: {OUTPUT_DIR}")

    # Step 2: Load data
    print("\nStep 2: Loading model data...")
    model_data = load_all_models()

    # Step 3: Calculate metrics
    print("\nStep 3: Calculating performance metrics...")
    metrics_df = calculate_metrics(model_data)

    print("\n" + "=" * 80)
    print("Performance Summary (Ranked by F1 Score)")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print()

    # Step 4: Generate visualizations
    print("\nStep 4: Generating visualizations...")
    print("-" * 80)

    generate_confusion_matrices(model_data, metrics_df)
    generate_misclassification_analysis(model_data)
    generate_agreement_heatmaps(model_data)
    generate_additional_insights(model_data, metrics_df)

    # Efficiency-focused visualizations (if timing data available)
    if (metrics_df["Time_s"] > 0).any():
        print("\nGenerating Efficiency Visualizations...")
        generate_efficiency_scatter(metrics_df, metrics_df)
        generate_speed_comparison(metrics_df)
    else:
        print("\n⚠ Skipping efficiency visualizations - no timing data available")

    # Summary
    print("\n" + "=" * 80)
    print(f"✓ Analysis complete! All visualizations saved to: {OUTPUT_DIR}/")
    print("=" * 80)
    print("\nGenerated files:")
    for output_file in sorted(OUTPUT_DIR.glob("*.jpg")):
        print(f"  - {output_file.name}")
    print()


if __name__ == "__main__":
    main()
