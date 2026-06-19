#!/usr/bin/env python3
"""Render a terminal histogram of pending transcript sizes in tokens."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.util.fs_util import FSUtil

DEFAULT_BUCKET_WIDTH_TOKENS = 8_000
DEFAULT_MAX_TOKENS = 256_000
DEFAULT_BUCKET_COUNT = 32
DEFAULT_CHART_HEIGHT = 20
DEFAULT_CELL_WIDTH = 2
MIN_POWER_OF_TWO_BUCKET_WIDTH_TOKENS = 1_024
FULL_BLOCK = "█"
HALF_BLOCK = "▄"


@dataclass(frozen=True)
class HistogramResult:
    """Computed token histogram for pending transcript summaries."""

    bucket_counts: list[int]
    total_files: int
    overflow_count: int
    total_tokens: int
    max_observed_tokens: int


def load_summarize_transcripts_module() -> ModuleType:
    """Load the existing summarize-transcripts script as a module."""
    script_path = Path(__file__).with_name("summarize-transcripts.py")
    spec = importlib.util.spec_from_file_location("summarize_transcripts_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bucket_count(max_tokens: int, bucket_width: int) -> int:
    """Return the number of buckets needed for the token range."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    if bucket_width <= 0:
        raise ValueError("bucket_width must be greater than zero")
    if max_tokens % bucket_width != 0:
        raise ValueError("max_tokens must be evenly divisible by bucket_width")
    return max_tokens // bucket_width


def choose_bucket_width(max_observed_tokens: int, bucket_count_value: int) -> int:
    """Choose the smallest power-of-two bucket width that includes the largest file."""
    if bucket_count_value <= 0:
        raise ValueError("bucket_count must be greater than zero")

    bucket_width = MIN_POWER_OF_TWO_BUCKET_WIDTH_TOKENS
    while bucket_width * bucket_count_value < max_observed_tokens:
        bucket_width *= 2
    return bucket_width


def bucket_index(token_count: int, max_tokens: int, bucket_width: int) -> int | None:
    """Return the histogram bucket index, or None when the token count overflows."""
    if token_count < 0:
        raise ValueError("token_count must be greater than or equal to zero")

    count = bucket_count(max_tokens, bucket_width)
    if token_count > max_tokens:
        return None
    if token_count == max_tokens:
        return count - 1
    return token_count // bucket_width


def build_histogram(token_counts: list[int], max_tokens: int, bucket_width: int) -> HistogramResult:
    """Build fixed-width histogram buckets from token counts."""
    counts = [0] * bucket_count(max_tokens, bucket_width)
    overflow = 0

    for token_count in token_counts:
        index = bucket_index(token_count, max_tokens, bucket_width)
        if index is None:
            overflow += 1
        else:
            counts[index] += 1

    return HistogramResult(
        bucket_counts=counts,
        total_files=len(token_counts),
        overflow_count=overflow,
        total_tokens=sum(token_counts),
        max_observed_tokens=max(token_counts, default=0),
    )


def token_counts_for_pending_files(
    pending_files: list[tuple[Path, Path]],
    encoder: tiktoken.Encoding,
    show_progress: bool,
) -> list[int]:
    """Read pending transcript files and estimate their token counts."""
    token_counts: list[int] = []
    current_channel = ""
    for txt_file, _output_file in pending_files:
        channel_name = txt_file.parent.name
        if show_progress and channel_name != current_channel:
            print(f"Analyzing: {channel_name}")
            current_channel = channel_name
        transcript = FSUtil.read_text_file(txt_file)
        token_counts.append(len(encoder.encode(transcript, disallowed_special=())))
    return token_counts


def format_k_tokens(tokens: int) -> str:
    """Format tokens as a compact K-token label."""
    if tokens == 0:
        return "0K"
    if tokens % 1_000 == 0:
        return f"{tokens // 1_000}K"
    if tokens % 1_024 == 0:
        return f"{tokens // 1_024}K"
    return f"{tokens:,}"


def place_label(row: list[str], position: int, label: str) -> None:
    """Place a label into a row without exceeding row bounds."""
    if len(label) > len(row):
        return
    start = min(max(position, 0), len(row) - len(label))
    for offset, char in enumerate(label):
        row[start + offset] = char


def render_x_axis(bucket_count_value: int, bucket_width: int, max_tokens: int, cell_width: int) -> list[str]:
    """Render staggered token labels for bucket boundaries."""
    plot_width = bucket_count_value * cell_width
    tick_row = [" "] * plot_width
    labels = [boundary for boundary in range(0, max_tokens + bucket_width, bucket_width * 2) if boundary <= max_tokens]

    for boundary in labels:
        position = min((boundary // bucket_width) * cell_width, plot_width - 1)
        tick_row[position] = "|"

    label_rows = [[" "] * plot_width for _ in range(4)]
    for label_number, boundary in enumerate(labels):
        label = format_k_tokens(boundary)
        position = min((boundary // bucket_width) * cell_width, plot_width - len(label))
        place_label(label_rows[label_number % len(label_rows)], position, label)

    return ["      " + "".join(tick_row), *["      " + "".join(row).rstrip() for row in label_rows]]


def render_bucket_cell(count: int, percentage: float, row_floor_pct: float, row_ceiling_pct: float, row_height_pct: float) -> str:
    """Render one histogram cell for a bucket at a specific percentage row."""
    if percentage >= row_ceiling_pct:
        return FULL_BLOCK
    if percentage > row_floor_pct:
        fill_fraction = (percentage - row_floor_pct) / row_height_pct
        return FULL_BLOCK if fill_fraction >= 0.5 else HALF_BLOCK
    if count > 0 and row_floor_pct == 0:
        return HALF_BLOCK
    return " "


def render_histogram(result: HistogramResult, max_tokens: int, bucket_width: int, chart_height: int, cell_width: int) -> str:
    """Render histogram buckets as terminal text."""
    if chart_height <= 0:
        raise ValueError("chart_height must be greater than zero")
    if cell_width <= 0:
        raise ValueError("cell_width must be greater than zero")

    if result.total_files == 0:
        return "No pending non-empty transcript files found."

    plot_width = len(result.bucket_counts) * cell_width
    major_ticks = {100: "100%", 75: "75%", 50: "50%", 25: "25%", 0: "0%"}
    tick_rows = {round(percent / 100 * chart_height): (label, percent) for percent, label in major_ticks.items()}
    lines = [
        "Pending transcript token histogram",
        f"Files: {result.total_files:,} | Bucket width: {format_k_tokens(bucket_width)} tokens | "
        f"Range: 0-{format_k_tokens(max_tokens)} tokens",
        "",
        f"{'pct':>4} |{' ' * plot_width}| {'files':>7}",
    ]

    bucket_percentages = [(count / result.total_files) * 100 for count in result.bucket_counts]

    row_height_pct = 100 / chart_height
    for row in range(chart_height, 0, -1):
        row_floor_pct = (row - 1) * row_height_pct
        row_ceiling_pct = row * row_height_pct
        bars = []
        for count, percentage in zip(result.bucket_counts, bucket_percentages, strict=True):
            bars.append(render_bucket_cell(count, percentage, row_floor_pct, row_ceiling_pct, row_height_pct) * cell_width)

        tick = tick_rows.get(row)
        label = tick[0] if tick else ""
        count_label = ""
        if tick:
            count_at_tick = round(result.total_files * tick[1] / 100)
            count_label = f"{count_at_tick:,}"
        lines.append(f"{label:>4} |{''.join(bars)}| {count_label:>7}")

    baseline_count = tick_rows.get(0, ("", 0))[0]
    lines.append(f"{baseline_count:>4} +{'-' * plot_width}+ {0:>7}")
    lines.extend(render_x_axis(len(result.bucket_counts), bucket_width, max_tokens, cell_width))

    if result.overflow_count > 0:
        lines.append("")
        lines.append(f"Overflow above {format_k_tokens(max_tokens)} tokens: {result.overflow_count:,} file(s)")

    average = result.total_tokens / result.total_files
    lines.append("")
    lines.append(f"Average pending transcript size: {average:,.0f} tokens")
    lines.append(f"Largest pending transcript size: {result.max_observed_tokens:,} tokens")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="", help="Only include pending transcripts from this channel directory")
    parser.add_argument(
        "--bucket-width",
        type=int,
        default=None,
        help=f"Histogram bucket width in tokens; defaults to the smallest power-of-two width for {DEFAULT_BUCKET_COUNT} buckets",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum token value shown on the x-axis; defaults to bucket-width multiplied by bucket-count",
    )
    parser.add_argument("--bucket-count", type=int, default=DEFAULT_BUCKET_COUNT, help="Number of histogram buckets")
    parser.add_argument("--height", type=int, default=DEFAULT_CHART_HEIGHT, help="Histogram height in terminal rows")
    return parser.parse_args(argv)


def resolve_histogram_range(
    max_observed_tokens: int,
    bucket_count_value: int,
    bucket_width: int | None,
    max_tokens: int | None,
) -> tuple[int, int]:
    """Resolve bucket width and x-axis maximum from explicit options or observed data."""
    if bucket_count_value <= 0:
        raise ValueError("bucket_count must be greater than zero")

    resolved_bucket_width = bucket_width if bucket_width is not None else choose_bucket_width(max_observed_tokens, bucket_count_value)
    if resolved_bucket_width <= 0:
        raise ValueError("bucket_width must be greater than zero")

    resolved_max_tokens = max_tokens if max_tokens is not None else resolved_bucket_width * bucket_count_value
    if resolved_max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    if resolved_max_tokens % resolved_bucket_width != 0:
        raise ValueError("max_tokens must be evenly divisible by bucket_width")
    if resolved_max_tokens < max_observed_tokens:
        raise ValueError("max_tokens must include the largest pending transcript")

    return resolved_bucket_width, resolved_max_tokens


def main(argv: list[str] | None = None) -> int:
    """Run the token histogram report."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    config = Config(config_path)
    cleaned_dir = config.get_data_downloads_transcripts_cleaned_dir()
    summaries_dir = config.get_data_downloads_transcripts_summaries_dir()

    if not cleaned_dir.exists():
        print(f"Error: Cleaned transcripts directory not found: {cleaned_dir}", file=sys.stderr)
        return 1

    summarize_module = load_summarize_transcripts_module()
    pending, total, already_done, empty_files = summarize_module.collect_pending_files(
        cleaned_dir,
        summaries_dir,
        args.channel.strip(),
    )

    if total == 0:
        print(f"Error: No .txt files found in: {cleaned_dir}", file=sys.stderr)
        return 1

    encoder = tiktoken.get_encoding(config.get_encoding_name())
    if args.channel.strip():
        print(f"Channel filter: {args.channel.strip()}")
    print(f"Found {total:,} transcript(s), {already_done:,} already summarized, {empty_files:,} empty, {len(pending):,} pending")
    print(f"Encoding: {config.get_encoding_name()}")
    print()
    token_counts = token_counts_for_pending_files(pending, encoder, show_progress=True)
    max_observed_tokens = max(token_counts, default=0)
    bucket_width, max_tokens = resolve_histogram_range(
        max_observed_tokens,
        args.bucket_count,
        args.bucket_width,
        args.max_tokens,
    )
    result = build_histogram(token_counts, max_tokens, bucket_width)

    print()
    print(render_histogram(result, max_tokens, bucket_width, args.height, DEFAULT_CELL_WIDTH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
