#!/usr/bin/env python3
"""Visualize vector similarity at different word distances from embeddings JSON files.

Creates a stacked visualization showing cosine similarity between embedding vectors
at various word distances (2, 4, 8, 16, 32, 64, 128, 256, 512 words).

Usage:
    uv run python scripts/visualize-embeddings.py --file path/to/embeddings.json
    uv run python scripts/visualize-embeddings.py  # Process all embeddings files
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Word distances to visualize
DISTANCES = [2, 4, 8, 16, 32, 64, 128, 256, 512]


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_similarities_at_distance(
    embeddings: list[NDArray[np.float32]],
    offsets: list[int],
    target_distance: int,
    stride: int,
) -> tuple[list[int], list[float]]:
    """Compute cosine similarities between embeddings at approximately target_distance words apart.

    Args:
        embeddings: List of embedding vectors.
        offsets: List of word offsets for each embedding.
        target_distance: Target word distance between compared embeddings.
        stride: Word stride between consecutive windows.

    Returns:
        Tuple of (positions, similarities) where positions are word offsets.
    """
    # Calculate window skip needed for target distance
    window_skip = max(1, round(target_distance / stride))

    positions: list[int] = []
    similarities: list[float] = []

    for i in range(len(embeddings) - window_skip):
        j = i + window_skip
        sim = cosine_similarity(embeddings[i], embeddings[j])
        # Use midpoint between the two compared windows as position
        midpoint = (offsets[i] + offsets[j]) // 2
        positions.append(midpoint)
        similarities.append(sim)

    return positions, similarities


def load_embeddings(json_path: Path) -> tuple[list[NDArray[np.float32]], list[int], int, str]:
    """Load embeddings from JSON file.

    Returns:
        Tuple of (embeddings, offsets, stride, model_name).
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    stride = data["stride"]
    model_name = data["embedding_model"]
    windows = data["windows"]

    embeddings = [np.array(w["embed"], dtype=np.float32) for w in windows]
    offsets = [w["offset"] for w in windows]

    return embeddings, offsets, stride, model_name


def create_visualization(
    embeddings: list[NDArray[np.float32]],
    offsets: list[int],
    stride: int,
    model_name: str,
    title: str,
    output_path: Path,
) -> None:
    """Create stacked visualization of similarities at different distances."""
    # Filter distances that are feasible given the data
    max_offset = offsets[-1] if offsets else 0
    feasible_distances = [d for d in DISTANCES if d < max_offset // 2]

    if not feasible_distances:
        print(f"  Warning: Not enough data for visualization (max offset: {max_offset})")
        return

    n_plots = len(feasible_distances)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    # Color gradient from green (high similarity) to red (low similarity)
    for idx, distance in enumerate(feasible_distances):
        ax = axes[idx]
        positions, similarities = compute_similarities_at_distance(embeddings, offsets, distance, stride)

        if not positions:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(f"{distance}w")
            continue

        # Plot as filled area
        ax.fill_between(positions, similarities, alpha=0.3, color="steelblue")
        ax.plot(positions, similarities, color="steelblue", linewidth=0.8)

        # Add horizontal reference lines
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axhline(y=0.8, color="green", linestyle=":", linewidth=0.5, alpha=0.5)

        # Actual window skip used
        actual_skip = max(1, round(distance / stride))
        actual_distance = actual_skip * stride

        ax.set_ylabel(f"{distance}w\n(≈{actual_distance}w)", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlim(offsets[0], offsets[-1])

        # Add statistics
        mean_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        ax.text(
            0.98,
            0.95,
            f"μ={mean_sim:.2f} min={min_sim:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    # Labels
    axes[-1].set_xlabel("Word Position")
    fig.suptitle(f"Embedding Similarity at Different Word Distances\n{title}\nModel: {model_name}", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", format="jpg")
    plt.close(fig)


def process_embeddings_file(json_path: Path, output_dir: Path) -> bool:
    """Process a single embeddings JSON file and create visualization.

    Args:
        json_path: Path to the embeddings JSON file.
        output_dir: Directory to save the visualization.

    Returns:
        True if successful, False otherwise.
    """
    print(f"Processing: {json_path.name}")

    embeddings, offsets, stride, model_name = load_embeddings(json_path)
    print(f"  Loaded {len(embeddings)} embeddings, stride={stride}")

    if len(embeddings) < 10:
        print("  Warning: Too few embeddings for meaningful visualization")
        return False

    # Create output path
    output_path = output_dir / json_path.with_suffix(".similarity.jpg").name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract title from filename
    title = json_path.stem.replace(".embeddings", "")

    create_visualization(embeddings, offsets, stride, model_name, title, output_path)
    print(f"  Saved: {output_path}")

    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize embedding similarity at different word distances")
    parser.add_argument(
        "--file",
        type=Path,
        help="Process a single embeddings JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for visualizations (default: same as input)",
    )
    args = parser.parse_args()

    # Determine files to process
    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
        files = [args.file]
        default_output_dir = args.file.parent
    else:
        # Find all embeddings files in output/topics
        from src.config import Config

        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        config = Config(config_path)
        td_config = config.get_topic_detection_config()
        topics_dir = config.getDataDir() / td_config.output_dir

        if not topics_dir.exists():
            print(f"Error: Topics directory not found: {topics_dir}", file=sys.stderr)
            return 1

        files = sorted(topics_dir.rglob("*.embeddings.json"))
        default_output_dir = topics_dir

    output_dir = args.output_dir or default_output_dir

    if not files:
        print("No embeddings files found to process.")
        return 0

    print(f"Found {len(files)} embeddings file(s) to process")
    print(f"Output directory: {output_dir}")
    print()

    success_count = 0
    for f in files:
        if process_embeddings_file(f, output_dir):
            success_count += 1
        print()

    print("=" * 50)
    print(f"Completed: {success_count}/{len(files)} visualizations created")

    return 0


if __name__ == "__main__":
    sys.exit(main())
