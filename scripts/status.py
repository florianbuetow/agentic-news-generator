#!/usr/bin/env python3
"""Display processing status of downloaded content."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config

TIMESTAMP_RE = re.compile(r"(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)[,.](?P<millis>\d{3})")


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


def _parse_bool_flag(raw: str | None) -> bool:
    """Parse common truthy/falsey flag values."""
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    return False


def _timestamp_to_seconds(timestamp: str) -> int | None:
    """Convert `HH:MM:SS,mmm` (or dot millis) to integer seconds."""
    match = TIMESTAMP_RE.fullmatch(timestamp.strip())
    if match is None:
        return None
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    millis = int(match.group("millis"))
    return (hours * 3600) + (minutes * 60) + seconds + (millis // 1000)


def _extract_last_timestamp_seconds_from_srt(path: Path, tail_line_count: int = 30) -> int | None:
    """Extract the last timestamp from the final lines of an SRT file."""
    tail_lines: deque[str] = deque(maxlen=tail_line_count)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                tail_lines.append(line)
    except OSError:
        return None

    last_seconds: int | None = None
    for line in tail_lines:
        for match in TIMESTAMP_RE.finditer(line):
            parsed_seconds = _timestamp_to_seconds(match.group(0))
            if parsed_seconds is not None:
                last_seconds = parsed_seconds

    return last_seconds


def _sum_channel_cleaned_srt_seconds(channel_dir: Path) -> int:
    """Sum per-file end timestamps from cleaned SRT files for one channel."""
    if not channel_dir.exists():
        return 0

    total_seconds = 0
    for srt_file in channel_dir.rglob("*.srt"):
        if not srt_file.is_file() or srt_file.name.startswith("._"):
            continue
        last_seconds = _extract_last_timestamp_seconds_from_srt(srt_file)
        if last_seconds is None or last_seconds < 0:
            continue
        total_seconds += last_seconds
    return total_seconds


def _format_seconds_as_dhm(total_seconds: int) -> str:
    """Format absolute seconds as `<days>d<hours>h<minutes>m`."""
    if total_seconds < 0:
        total_seconds = 0
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    return f"{days}d{hours}h{minutes}m"


def _compute_channel_time_seconds_map(transcripts_cleaned_dir: Path, show_progress: bool = False) -> dict[str, int]:
    """Compute total cleaned transcript duration (seconds) for each channel."""
    per_channel_seconds: dict[str, int] = {}
    if not transcripts_cleaned_dir.exists():
        return per_channel_seconds

    channel_dirs = sorted((channel_dir for channel_dir in transcripts_cleaned_dir.iterdir() if channel_dir.is_dir()), key=lambda p: p.name)
    if show_progress:
        print(f"Computing transcript time totals for {len(channel_dirs)} channel(s)...", flush=True)

    for index, channel_dir in enumerate(channel_dirs, start=1):
        if show_progress:
            print(f"[{index}/{len(channel_dirs)}] Analyzing channel: {channel_dir.name}", flush=True)
        if not channel_dir.is_dir():
            continue
        per_channel_seconds[channel_dir.name] = _sum_channel_cleaned_srt_seconds(channel_dir)

    if show_progress:
        print("Finished transcript time totals.", flush=True)
    return per_channel_seconds


def get_channel_stats(  # noqa: C901
    downloads_dir: Path,
    archive_dir: Path,
    transcripts_hallucinations_dir: Path,
    transcripts_cleaned_dir: Path,
    transcripts_summaries_dir: Path,
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
            "summaries": 0,
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

    # Count summaries
    if transcripts_summaries_dir.exists():
        for channel_dir in transcripts_summaries_dir.iterdir():
            if channel_dir.is_dir():
                count = sum(1 for f in channel_dir.iterdir() if f.suffix == ".md" and not f.name.startswith("._"))
                stats[channel_dir.name]["summaries"] = count

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
    "summaries",
]


def _fmt_cell(value: int, prev: int | None, num_width: int) -> str:
    """Format number right-aligned to num_width + fixed 3-char delta suffix (always 3 chars)."""
    num = f"\033[90m{'-':>{num_width}}\033[0m" if value == 0 else f"{value:>{num_width}}"
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
    show_time: bool,
    time_value: str,
    time_col_width: int,
    channel_width: int,
    col_width: int,
) -> None:
    """Print a single stats row (channel or TOTAL) with optional inline deltas."""
    num_w = col_width - 3
    parts = [f"{label:<{channel_width}}"]
    for key in STAT_KEYS:
        prev_val = prev.get(key) if prev else None
        parts.append(_fmt_cell(int(stats[key]), prev_val, num_w))
        if show_time and key == "audio":
            parts.append(f"{time_value:>{time_col_width}}")
    parts.append(f"{completion_pct:>{col_width - 1}.1f}%")
    parts.append(f"{size_gb:>{col_width}.1f}")
    print(" ".join(parts))


def main() -> int:  # noqa: C901
    """Main entry point."""
    update_cache = "--no-update-cache" not in sys.argv
    show_time = _parse_bool_flag(os.environ.get("SHOW_TIME"))

    # Load configuration
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    config = Config(config_path)

    # Get directories from Config
    downloads_dir = config.getDataDownloadsDir()
    archive_dir = config.getDataArchiveDir()
    transcripts_hallucinations_dir = config.getDataDownloadsTranscriptsHallucinationsDir()
    transcripts_cleaned_dir = config.getDataDownloadsTranscriptsCleanedDir()
    transcripts_summaries_dir = config.getDataDownloadsTranscriptsSummariesDir()

    if not downloads_dir.exists():
        print(f"Error: Downloads directory not found: {downloads_dir}")
        return 1

    # Get channel statistics
    channel_stats = get_channel_stats(
        downloads_dir,
        archive_dir,
        transcripts_hallucinations_dir,
        transcripts_cleaned_dir,
        transcripts_summaries_dir,
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
        "summaries": sum(int(s["summaries"]) for s in channel_stats.values()),
        "total_size_bytes": sum(float(s["total_size_bytes"]) for s in channel_stats.values()),
    }

    channel_time_seconds: dict[str, int] = {}
    if show_time:
        channel_time_seconds = _compute_channel_time_seconds_map(transcripts_cleaned_dir, show_progress=True)
    total_time_seconds = sum(channel_time_seconds.values()) if show_time else 0

    total_videos = totals["videos_active"] + totals["videos_archived"]

    # Print summary
    print("=" * 120)
    print("PROCESSING STATUS")
    print("=" * 120)
    print()
    print(f"Total Videos: {total_videos} (Active: {totals['videos_active']}, Archived: {totals['videos_archived']})")
    print(f"Total Channels: {len(channel_stats)}")
    print()

    total_transcripts = int(totals["transcripts"])
    if total_transcripts > 0:
        overall_pct = (totals["summaries"] / total_transcripts) * 100
        print(f"Overall Pipeline Completion: {overall_pct:.1f}%")
        print()

    # Load previous stats from cache
    cache_file = project_root / ".cache" / "stats_previous.json"
    previous: dict[str, Any] | None = None
    if cache_file.exists():
        previous = json.loads(cache_file.read_text())

    col_width = 8
    time_col_width = 10
    channel_width = 40

    columns = [
        ("", "Videos", col_width),
        ("Arch.", "Videos", col_width),
        ("", "Audio", col_width),
    ]
    if show_time:
        formatted_time_values = [_format_seconds_as_dhm(seconds) for seconds in channel_time_seconds.values()]
        formatted_total_time = _format_seconds_as_dhm(total_time_seconds)
        max_time_len = max([len(formatted_total_time), *(len(value) for value in formatted_time_values)], default=time_col_width)
        time_col_width = max(time_col_width, max_time_len)
        columns.append(("", "Time", time_col_width))
    else:
        formatted_total_time = ""

    columns.extend(
        [
            ("Tran-", "scripts", col_width),
            ("Hall.", "Analysis", col_width),
            ("Cleaned", "Trans.", col_width),
            ("", "Summ.", col_width),
            ("", "%", col_width),
            ("", "GB", col_width),
        ]
    )

    # Calculate total line width from effective column widths
    line_width = channel_width + 1 + sum(width + 1 for _, _, width in columns)

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
        transcripts_total = int(s["transcripts"])
        completion_pct = (s["summaries"] / transcripts_total * 100) if transcripts_total > 0 else 0.0
        size_gb = float(s["total_size_bytes"]) / (1024**3)
        display_name = channel_name[:channel_width] if len(channel_name) > channel_width else channel_name
        prev_ch: dict[str, int] | None = prev_channels.get(channel_name)
        channel_time_label = _format_seconds_as_dhm(channel_time_seconds.get(channel_name, 0)) if show_time else ""
        _print_stat_row(
            display_name,
            s,
            prev_ch,
            completion_pct,
            size_gb,
            show_time,
            channel_time_label,
            time_col_width,
            channel_width,
            col_width,
        )

    # Print totals row
    print("-" * line_width)
    overall_pct = (totals["summaries"] / total_transcripts * 100) if total_transcripts > 0 else 0.0
    total_size_gb = float(totals["total_size_bytes"]) / (1024**3)
    prev_totals: dict[str, int] | None = previous.get("totals") if previous else None
    _print_stat_row(
        "TOTAL",
        totals,
        prev_totals,
        overall_pct,
        total_size_gb,
        show_time,
        formatted_total_time,
        time_col_width,
        channel_width,
        col_width,
    )

    # Print percentage row: each column as % of total transcripts
    pct_keys = set(STAT_KEYS[STAT_KEYS.index("transcripts") :])
    num_w = col_width - 3
    pct_parts = [f"{'% OF TRANSCRIPTS':<{channel_width}}"]
    for key in STAT_KEYS:
        if key in pct_keys and total_transcripts > 0:
            pct = int(totals[key]) / total_transcripts * 100
            pct_str = f"{pct:.0f}%"
            pct_parts.append(f"{pct_str:>{num_w}}   ")
        else:
            pct_parts.append(f"{'':>{num_w}}   ")
        if show_time and key == "audio":
            pct_parts.append(f"{'':>{time_col_width}}")
    pct_parts.append(f"{'':>{col_width}}")
    pct_parts.append(f"{'':>{col_width}}")
    print(" ".join(pct_parts))

    if update_cache:
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
