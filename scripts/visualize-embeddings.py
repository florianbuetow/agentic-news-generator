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
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
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


def parse_srt_timestamp(ts: str) -> float:
    """Parse SRT timestamp (HH:MM:SS,mmm) to seconds."""
    parts = ts.replace(",", ".").split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


@dataclass
class EmbeddingsData:
    """Container for embeddings data loaded from JSON."""

    embeddings: list[NDArray[np.float32]]
    offsets: list[int]
    timestamps: list[float]  # Start timestamp in seconds for each window
    stride: int
    model_name: str


def load_embeddings(json_path: Path) -> EmbeddingsData:
    """Load embeddings from JSON file.

    Returns:
        EmbeddingsData containing embeddings, offsets, timestamps, stride, and model_name.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    config = data["config"]
    stride = config["stride"]
    model_name = config["embedding_model"]
    windows = data["windows"]
    srt_entries = data.get("srt_entries", [])

    embeddings = [np.array(w["embedding"], dtype=np.float32) for w in windows]
    offsets = [w["offset"] for w in windows]

    # Build timestamp mapping from srt_entries (word_start -> start_timestamp)
    # Windows no longer contain timestamps directly; map from word position
    timestamps: list[float] = []
    if srt_entries:
        for w in windows:
            offset = w["offset"]
            # Find the SRT entry that contains this word position
            timestamp = 0.0
            for entry in srt_entries:
                if entry["word_start"] <= offset < entry["word_end"]:
                    timestamp = parse_srt_timestamp(entry["start_timestamp"])
                    break
            timestamps.append(timestamp)
    else:
        timestamps = [0.0] * len(windows)

    return EmbeddingsData(
        embeddings=embeddings,
        offsets=offsets,
        timestamps=timestamps,
        stride=stride,
        model_name=model_name,
    )


@dataclass
class SegmentInfo:
    """Information about a segment including position and topics."""

    start_word: int
    end_word: int
    high_level_topics: list[str]
    mid_level_topics: list[str]
    specific_topics: list[str]


def load_segments(json_path: Path) -> tuple[list[int], float, list[SegmentInfo]]:
    """Load segments JSON to get boundary positions, threshold, and segment info.

    Args:
        json_path: Path to the segments JSON file.

    Returns:
        Tuple of (boundary_positions, threshold_value, segment_infos).
        boundary_positions are word offsets where topic changes occur.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Get threshold from config
    threshold = data.get("config", {}).get("threshold_value", 0.4)

    # Boundaries are at end_word of each segment except the last
    segments = data.get("segments", [])
    boundaries: list[int] = []
    segment_infos: list[SegmentInfo] = []

    for i, seg in enumerate(segments):
        segment_infos.append(
            SegmentInfo(
                start_word=seg["start_word"],
                end_word=seg["end_word"],
                high_level_topics=[],  # Will be filled from topics JSON
                mid_level_topics=[],  # Will be filled from topics JSON
                specific_topics=[],  # Will be filled from topics JSON
            )
        )
        if i < len(segments) - 1:
            boundaries.append(seg["end_word"])

    return boundaries, threshold, segment_infos


@dataclass
class TopicLevels:
    """Container for all topic levels for a segment."""

    high_level_topics: list[str]
    mid_level_topics: list[str]
    specific_topics: list[str]


def load_topics(json_path: Path) -> list[TopicLevels]:
    """Load topics JSON to get all topic levels for each segment.

    Args:
        json_path: Path to the topics JSON file.

    Returns:
        List of TopicLevels, one per segment.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    return [
        TopicLevels(
            high_level_topics=seg.get("high_level_topics", []),
            mid_level_topics=seg.get("mid_level_topics", []),
            specific_topics=seg.get("specific_topics", []),
        )
        for seg in data.get("segments", [])
    ]


def _draw_boundary_markers(
    ax: Axes,
    boundaries: list[int],
) -> None:
    """Draw vertical red lines at boundary positions on the plot."""
    for boundary_pos in boundaries:
        ax.axvline(
            x=boundary_pos,
            color="red",
            linestyle="-",
            linewidth=1.5,
            alpha=0.7,
            zorder=5,
        )


def _wrap_topic_text(topics: list[str], max_width: int = 20) -> str:
    """Wrap topics into multi-line text with max_width chars per line.

    Args:
        topics: List of topic strings.
        max_width: Maximum characters per line.

    Returns:
        Multi-line string with wrapped topics.
    """
    if not topics:
        return ""
    lines: list[str] = []
    current_line = ""
    for topic in topics:
        if current_line and len(current_line) + len(topic) + 2 > max_width:
            lines.append(current_line)
            current_line = topic
        else:
            current_line = f"{current_line}, {topic}" if current_line else topic
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


def _draw_segment_backgrounds(
    ax: Axes,
    segment_infos: list[SegmentInfo],
    x_min: int,
    x_max: int,
    graph_index: int,
) -> None:
    """Draw alternating colored backgrounds for segments and topic labels.

    Topic labels are distributed by graph index:
    - graph_index 0: high_level_topics
    - graph_index 1: mid_level_topics
    - graph_index 2: specific_topics
    - graph_index 3+: no labels
    """
    colors = ["#f0f0ff", "#fff0f0"]  # Alternating light blue and light red

    for i, seg in enumerate(segment_infos):
        # Clamp segment bounds to visible range
        seg_start = max(seg.start_word, x_min)
        seg_end = min(seg.end_word, x_max)

        if seg_start >= seg_end:
            continue

        # Draw background span
        ax.axvspan(seg_start, seg_end, alpha=0.3, color=colors[i % 2], zorder=0)

        # Select topics based on graph_index
        if graph_index == 0:
            topics = seg.high_level_topics
        elif graph_index == 1:
            topics = seg.mid_level_topics
        elif graph_index == 2:
            topics = seg.specific_topics
        else:
            topics = []

        # Draw topic labels for first 3 graphs only
        if topics:
            topic_text = _wrap_topic_text(topics)

            # Position label at segment center
            label_x = (seg_start + seg_end) / 2
            ax.text(
                label_x,
                0.92,
                topic_text,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6,
                color="darkblue",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=10,
            )


def _plot_single_distance(
    ax: Axes,
    embeddings: list[NDArray[np.float32]],
    offsets: list[int],
    distance: int,
    stride: int,
    boundaries: list[int],
    threshold: float,
    segment_infos: list[SegmentInfo],
    graph_index: int,
) -> None:
    """Plot similarity curve for a single distance on given axes."""
    positions, similarities = compute_similarities_at_distance(embeddings, offsets, distance, stride)

    if not positions:
        raise ValueError(f"Insufficient data for distance {distance}")

    # Draw segment backgrounds first (behind everything)
    _draw_segment_backgrounds(ax, segment_infos, offsets[0], offsets[-1], graph_index)

    # Plot as filled area
    ax.fill_between(positions, similarities, alpha=0.3, color="steelblue")
    ax.plot(positions, similarities, color="steelblue", linewidth=0.8)

    # Add horizontal reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.8, color="green", linestyle=":", linewidth=0.5, alpha=0.5)

    # Draw threshold line
    ax.axhline(y=threshold, color="red", linestyle="-", linewidth=1.0, alpha=0.7)

    # Draw vertical boundary lines
    _draw_boundary_markers(ax, boundaries)

    # Axis labels and limits
    actual_skip = max(1, round(distance / stride))
    actual_distance = actual_skip * stride
    ax.set_ylabel(f"{distance}w\n(≈{actual_distance}w)", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(offsets[0], offsets[-1])

    # Add statistics text
    mean_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    stats_text = f"μ={mean_sim:.2f} min={min_sim:.2f} bounds={len(boundaries)}"
    ax.text(
        0.98,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )


def _create_offset_to_timestamp_mapper(offsets: list[int], timestamps: list[float]) -> tuple[Callable[[float], float], float]:
    """Create a function that maps word offsets to timestamps via interpolation.

    Returns:
        Tuple of (mapper_function, max_timestamp).
    """
    if not timestamps:
        raise ValueError("timestamps list cannot be empty")

    offsets_arr = np.array(offsets, dtype=np.float64)
    timestamps_arr = np.array(timestamps, dtype=np.float64)

    def mapper(offset: float) -> float:
        return float(np.interp(offset, offsets_arr, timestamps_arr))

    return mapper, timestamps[-1]


def _format_timestamp(seconds: float) -> str:
    """Format seconds as absolute minutes."""
    minutes = int(seconds // 60)
    return f"{minutes}m"


def create_visualization(
    embeddings: list[NDArray[np.float32]],
    offsets: list[int],
    stride: int,
    model_name: str,
    title: str,
    output_path: Path,
    boundaries: list[int],
    threshold: float,
    segment_infos: list[SegmentInfo],
    timestamps: list[float],
) -> None:
    """Create stacked visualization of similarities at different distances."""
    # Filter distances that are feasible given the data
    if not offsets:
        raise ValueError("offsets list cannot be empty")
    max_offset = offsets[-1]
    feasible_distances = [d for d in DISTANCES if d < max_offset // 2]

    if not feasible_distances:
        raise ValueError(f"Not enough data for visualization (max offset: {max_offset})")

    # Create offset-to-timestamp mapper
    has_timestamps = len(timestamps) > 0 and any(t > 0 for t in timestamps)
    format_xaxis: Callable[[float, int], str] | None = None
    if has_timestamps:
        offset_to_time, _ = _create_offset_to_timestamp_mapper(offsets, timestamps)

        def format_xaxis_fn(x: float, _pos: int) -> str:
            return _format_timestamp(offset_to_time(x))

        format_xaxis = format_xaxis_fn

    n_plots = len(feasible_distances)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    for idx, distance in enumerate(feasible_distances):
        _plot_single_distance(
            axes[idx],
            embeddings,
            offsets,
            distance,
            stride,
            boundaries,
            threshold,
            segment_infos,
            graph_index=idx,
        )

    # Set timestamp formatter on x-axis if timestamps available
    if format_xaxis is not None:
        axes[-1].xaxis.set_major_formatter(FuncFormatter(format_xaxis))
        axes[-1].set_xlabel("Time (minutes)")
    else:
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

    emb_data = load_embeddings(json_path)
    print(f"  Loaded {len(emb_data.embeddings)} embeddings, stride={emb_data.stride}")

    if len(emb_data.embeddings) < 10:
        print("  Warning: Too few embeddings for meaningful visualization")
        return False

    # Load segmentation JSON for boundaries and threshold (required)
    # File naming: input_embeddings.json -> input_segmentation.json
    segments_path = Path(str(json_path).replace("_embeddings.json", "_segmentation.json"))
    if not segments_path.exists():
        raise FileNotFoundError(f"Required segmentation file not found: {segments_path}")
    boundaries, threshold, segment_infos = load_segments(segments_path)
    print(f"  Loaded {len(boundaries)} boundaries, threshold={threshold}")

    # Load topics JSON and merge into segment_infos (required)
    # File naming: input_embeddings.json -> input_topics.json
    topics_path = Path(str(json_path).replace("_embeddings.json", "_topics.json"))
    if not topics_path.exists():
        raise FileNotFoundError(f"Required topics file not found: {topics_path}")
    topics_list = load_topics(topics_path)
    if len(topics_list) != len(segment_infos):
        raise ValueError(f"Topics count ({len(topics_list)}) doesn't match segments ({len(segment_infos)})")
    for seg_info, topics in zip(segment_infos, topics_list, strict=True):
        seg_info.high_level_topics = topics.high_level_topics
        seg_info.mid_level_topics = topics.mid_level_topics
        seg_info.specific_topics = topics.specific_topics
    print(f"  Loaded topics for {len(segment_infos)} segments")

    # Create output path
    output_filename = json_path.name.replace("_embeddings.json", "_similarity.jpg")
    output_path = output_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract title from filename (remove _embeddings suffix)
    # Normalize fullwidth Unicode characters (e.g. ： → :) to prevent missing glyph warnings
    title = unicodedata.normalize("NFKC", json_path.stem.replace("_embeddings", ""))

    create_visualization(
        emb_data.embeddings,
        emb_data.offsets,
        emb_data.stride,
        emb_data.model_name,
        title,
        output_path,
        boundaries,
        threshold,
        segment_infos,
        emb_data.timestamps,
    )
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
        # Find all topics files first, then derive embeddings file paths
        from src.config import Config

        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        config = Config(config_path)
        td_config = config.get_topic_detection_config()
        topics_dir = config.getDataDir() / td_config.output_dir

        if not topics_dir.exists():
            print(f"Error: Topics directory not found: {topics_dir}", file=sys.stderr)
            return 1

        # Find topics files and derive corresponding embeddings file paths
        topics_files = sorted(f for f in topics_dir.rglob("*_topics.json") if not f.name.startswith("._"))
        files = []
        for tf in topics_files:
            emb_path = Path(str(tf).replace("_topics.json", "_embeddings.json"))
            if emb_path.exists():
                files.append(emb_path)
            else:
                print(f"Warning: Embeddings file not found for {tf.name}")
        default_output_dir = topics_dir

    output_dir = args.output_dir or default_output_dir

    if not files:
        print("No files with complete topic data found to process.")
        return 0

    print(f"Found {len(files)} file(s) with complete topic data to process")
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
