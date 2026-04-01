#!/usr/bin/env python3
"""Display processing status of downloaded content."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config


def count_files_by_suffix(directory: Path, suffix: str) -> int:
    """Count files with a specific suffix in a directory (recursive, excludes macOS metadata)."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob(f"*{suffix}") if f.is_file() and not f.name.startswith("._"))


def count_files_by_pattern(directory: Path, pattern: str) -> int:
    """Count files matching a glob pattern in a directory (excludes macOS metadata)."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob(pattern) if f.is_file() and not f.name.startswith("._"))


def get_channel_stats(  # noqa: C901
    downloads_dir: Path,
    archive_dir: Path,
    transcripts_hallucinations_dir: Path,
    transcripts_cleaned_dir: Path,
    topics_dir: Path,
) -> dict[str, dict[str, int | float]]:
    """Get statistics for each channel across all pipeline stages.

    Returns dict mapping channel names to stats dicts with counts for each stage.
    """
    stats: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "videos_active": 0,
            "videos_archived": 0,
            "audio": 0,
            "transcripts": 0,
            "hall_analysis": 0,
            "cleaned_transcripts": 0,
            "topics_embeddings": 0,
            "topics_segmentations": 0,
            "topics_extracted": 0,
            "topics_visualizations": 0,
            "total_size_bytes": 0,
        }
    )

    video_extensions = {".mp4", ".webm", ".m4a", ".mov", ".m4v", ".avi", ".mkv", ".flv"}

    # Count active videos and their sizes
    videos_dir = downloads_dir / "videos"
    if videos_dir.exists():
        for channel_dir in videos_dir.iterdir():
            if channel_dir.is_dir():
                video_files = [
                    f for f in channel_dir.iterdir() if f.is_file() and f.suffix in video_extensions and not f.name.startswith("._")
                ]
                stats[channel_dir.name]["videos_active"] = len(video_files)
                stats[channel_dir.name]["total_size_bytes"] += sum(f.stat().st_size for f in video_files)

    # Count archived videos and their sizes
    archive_videos_dir = archive_dir / "videos"
    if archive_videos_dir.exists():
        for channel_dir in archive_videos_dir.iterdir():
            if channel_dir.is_dir():
                video_files = [
                    f for f in channel_dir.iterdir() if f.is_file() and f.suffix in video_extensions and not f.name.startswith("._")
                ]
                stats[channel_dir.name]["videos_archived"] = len(video_files)
                stats[channel_dir.name]["total_size_bytes"] += sum(f.stat().st_size for f in video_files)

    # Count audio files
    audio_dir = downloads_dir / "audio"
    if audio_dir.exists():
        for channel_dir in audio_dir.iterdir():
            if channel_dir.is_dir():
                count = sum(1 for f in channel_dir.iterdir() if f.suffix == ".wav" and not f.name.startswith("._"))
                stats[channel_dir.name]["audio"] = count

    # Count transcripts (.srt files)
    transcripts_dir = downloads_dir / "transcripts"
    if transcripts_dir.exists():
        for channel_dir in transcripts_dir.iterdir():
            if channel_dir.is_dir():
                count = sum(1 for f in channel_dir.iterdir() if f.suffix == ".srt" and not f.name.startswith("._"))
                stats[channel_dir.name]["transcripts"] = count

    # Count hallucination analysis files
    if transcripts_hallucinations_dir.exists():
        for channel_dir in transcripts_hallucinations_dir.iterdir():
            if channel_dir.is_dir():
                count = sum(1 for f in channel_dir.iterdir() if f.suffix == ".json" and not f.name.startswith("._"))
                stats[channel_dir.name]["hall_analysis"] = count

    # Count cleaned transcripts
    if transcripts_cleaned_dir.exists():
        for channel_dir in transcripts_cleaned_dir.iterdir():
            if channel_dir.is_dir():
                count = sum(1 for f in channel_dir.iterdir() if f.suffix == ".srt" and not f.name.startswith("._"))
                stats[channel_dir.name]["cleaned_transcripts"] = count

    # Count topic detection outputs
    if topics_dir.exists():
        for channel_dir in topics_dir.iterdir():
            if channel_dir.is_dir():
                channel_name = channel_dir.name

                # Embeddings
                emb_count = sum(1 for f in channel_dir.iterdir() if f.name.endswith("_embeddings.json") and not f.name.startswith("._"))
                stats[channel_name]["topics_embeddings"] = emb_count

                # Segmentations
                seg_count = sum(1 for f in channel_dir.iterdir() if f.name.endswith("_segmentation.json") and not f.name.startswith("._"))
                stats[channel_name]["topics_segmentations"] = seg_count

                # Topics extracted
                topics_count = sum(1 for f in channel_dir.iterdir() if f.name.endswith("_topics.json") and not f.name.startswith("._"))
                stats[channel_name]["topics_extracted"] = topics_count

                # Visualizations
                viz_count = sum(1 for f in channel_dir.iterdir() if f.name.endswith("_similarity.jpg") and not f.name.startswith("._"))
                stats[channel_name]["topics_visualizations"] = viz_count

    return dict(stats)


def print_two_row_header(columns: list[tuple[str, str, int]]) -> tuple[str, str]:
    """Generate two header rows from column definitions.

    Args:
        columns: List of (row1_text, row2_text, width) tuples

    Returns:
        Tuple of (header_row1, header_row2) strings
    """
    row1_parts: list[str] = []
    row2_parts: list[str] = []

    for row1_text, row2_text, width in columns:
        row1_parts.append(f"{row1_text:>{width}}")
        row2_parts.append(f"{row2_text:>{width}}")

    return " ".join(row1_parts), " ".join(row2_parts)


STAT_KEYS = [
    "videos_active",
    "videos_archived",
    "audio",
    "transcripts",
    "hall_analysis",
    "cleaned_transcripts",
    "topics_embeddings",
    "topics_segmentations",
    "topics_extracted",
    "topics_visualizations",
]


def _fmt_cell(value: int, prev: int | None, num_width: int) -> str:
    """Format number right-aligned to num_width + fixed 3-char delta suffix (always 3 chars)."""
    if value == 0:
        num = f"\033[90m{'-':>{num_width}}\033[0m"
    else:
        num = f"{value:>{num_width}}"
    if prev is None or value == prev:
        return num + "   "
    diff = value - prev
    if diff > 0:
        tag = "+99" if diff > 99 else f"+{diff}"
        return num + f"\033[0;32m{tag:<3}\033[0m"
    tag = "-99" if diff < -99 else str(diff)
    return num + f"\033[0;31m{tag:<3}\033[0m"


def _print_stat_row(
    label: str,
    stats: dict[str, int | float],
    prev: dict[str, int] | None,
    completion_pct: float,
    size_gb: float,
    channel_width: int,
    col_width: int,
) -> None:
    """Print a single stats row (channel or TOTAL) with optional inline deltas."""
    num_w = col_width - 3
    parts = [f"{label:<{channel_width}}"]
    for key in STAT_KEYS:
        prev_val = prev.get(key) if prev else None
        parts.append(_fmt_cell(int(stats[key]), prev_val, num_w))
    parts.append(f"{completion_pct:>{col_width - 1}.1f}%")
    parts.append(f"{size_gb:>{col_width}.1f}")
    print(" ".join(parts))


def main() -> int:
    """Main entry point."""
    # Load configuration
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    config = Config(config_path)

    # Get directories from Config
    downloads_dir = config.getDataDownloadsDir()
    archive_dir = config.getDataArchiveDir()
    transcripts_hallucinations_dir = config.getDataDownloadsTranscriptsHallucinationsDir()
    transcripts_cleaned_dir = config.getDataDownloadsTranscriptsCleanedDir()

    # Topics output directory
    td_config = config.get_topic_detection_config()
    topics_dir = config.getDataDir() / td_config.output_dir

    if not downloads_dir.exists():
        print(f"Error: Downloads directory not found: {downloads_dir}")
        return 1

    # Get channel statistics
    channel_stats = get_channel_stats(
        downloads_dir,
        archive_dir,
        transcripts_hallucinations_dir,
        transcripts_cleaned_dir,
        topics_dir,
    )

    if not channel_stats:
        print("No data found.")
        return 0

    # Calculate totals
    totals: dict[str, int | float] = {
        "videos_active": sum(int(s["videos_active"]) for s in channel_stats.values()),
        "videos_archived": sum(int(s["videos_archived"]) for s in channel_stats.values()),
        "audio": sum(int(s["audio"]) for s in channel_stats.values()),
        "transcripts": sum(int(s["transcripts"]) for s in channel_stats.values()),
        "hall_analysis": sum(int(s["hall_analysis"]) for s in channel_stats.values()),
        "cleaned_transcripts": sum(int(s["cleaned_transcripts"]) for s in channel_stats.values()),
        "topics_embeddings": sum(int(s["topics_embeddings"]) for s in channel_stats.values()),
        "topics_segmentations": sum(int(s["topics_segmentations"]) for s in channel_stats.values()),
        "topics_extracted": sum(int(s["topics_extracted"]) for s in channel_stats.values()),
        "topics_visualizations": sum(int(s["topics_visualizations"]) for s in channel_stats.values()),
        "total_size_bytes": sum(float(s["total_size_bytes"]) for s in channel_stats.values()),
    }

    total_videos = totals["videos_active"] + totals["videos_archived"]

    # Print summary
    print("=" * 120)
    print("PROCESSING STATUS")
    print("=" * 120)
    print()
    print(f"Total Videos: {total_videos} (Active: {totals['videos_active']}, Archived: {totals['videos_archived']})")
    print(f"Total Channels: {len(channel_stats)}")
    print()

    # Calculate overall completion (topics extracted / total videos)
    if total_videos > 0:
        overall_pct = (totals["topics_extracted"] / total_videos) * 100
        print(f"Overall Pipeline Completion: {overall_pct:.1f}%")
        print()

    # Load previous stats from cache
    cache_file = project_root / ".cache" / "stats_previous.json"
    previous: dict[str, Any] | None = None
    if cache_file.exists():
        previous = json.loads(cache_file.read_text())

    col_width = 8
    channel_width = 40

    columns = [
        ("", "Videos", col_width),
        ("Arch.", "Videos", col_width),
        ("", "Audio", col_width),
        ("Tran-", "scripts", col_width),
        ("Hall.", "Analysis", col_width),
        ("Cleaned", "Trans.", col_width),
        ("Topics", "Embed.", col_width),
        ("Topics", "Segment.", col_width),
        ("Topics", "Extract.", col_width),
        ("Topics", "Visual.", col_width),
        ("", "%", col_width),
        ("", "GB", col_width),
    ]

    # Calculate total line width: channel + space + (col_width + space) * num_columns
    num_columns = len(columns)
    line_width = channel_width + 1 + (col_width + 1) * num_columns

    # Generate header rows
    header_row1, header_row2 = print_two_row_header(columns)

    # Print table
    print("=" * line_width)
    print("PIPELINE STATUS BY CHANNEL")
    print("=" * line_width)
    print()

    # Print two-row header
    print(f"{'':>{channel_width}} {header_row1}")
    print(f"{'Channel':<{channel_width}} {header_row2}")
    print("-" * line_width)

    prev_channels: dict[str, dict[str, int]] = previous.get("channels", {}) if previous else {}

    for channel_name in sorted(channel_stats.keys()):
        s = channel_stats[channel_name]
        videos_total = s["videos_active"] + s["videos_archived"]
        completion_pct = (s["topics_extracted"] / videos_total * 100) if videos_total > 0 else 0.0
        size_gb = float(s["total_size_bytes"]) / (1024**3)
        display_name = channel_name[:channel_width] if len(channel_name) > channel_width else channel_name
        prev_ch: dict[str, int] | None = prev_channels.get(channel_name)
        _print_stat_row(display_name, s, prev_ch, completion_pct, size_gb, channel_width, col_width)

    # Print totals row
    print("-" * line_width)
    overall_pct = (totals["topics_extracted"] / total_videos * 100) if total_videos > 0 else 0.0
    total_size_gb = float(totals["total_size_bytes"]) / (1024**3)
    prev_totals: dict[str, int] | None = previous.get("totals") if previous else None
    _print_stat_row("TOTAL", totals, prev_totals, overall_pct, total_size_gb, channel_width, col_width)

    # Save current stats for next run
    cache_data = {
        "totals": {k: int(totals[k]) for k in STAT_KEYS},
        "channels": {name: {k: int(s[k]) for k in STAT_KEYS} for name, s in channel_stats.items()},
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache_data, indent=2) + "\n")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
