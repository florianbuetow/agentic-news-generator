#!/usr/bin/env python3
"""Detect hallucinations in SRT transcripts using repetition detection."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import srt
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.processing.repetition_detector import RepetitionDetector
from src.util.fs_util import FSUtil


def timedelta_to_srt_timestamp(td: timedelta) -> str:
    """Convert timedelta to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        td: Timedelta object.

    Returns:
        Timestamp in SRT format (00:02:45,120).
    """
    total_seconds = int(td.total_seconds())
    milliseconds = int(td.total_seconds() * 1000) % 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def detect_hallucinations_in_file(
    srt_file: Path,
    detector: RepetitionDetector,
    min_window_size: int,
    overlap_percent: float,
) -> dict[str, Any]:
    """Process SRT file using SRT-entry-based sliding windows.

    Args:
        srt_file: Path to SRT file.
        detector: RepetitionDetector instance (uses SVM classifier for detection).
        min_window_size: Minimum window size in words.
        overlap_percent: Overlap percentage.

    Returns:
        Dictionary with detection results.
    """
    from collections import deque

    # Read and parse SRT
    srt_content = FSUtil.read_text_file(srt_file)
    subtitles = list(srt.parse(srt_content))

    # Prepare all SRT entries
    parts = [(subtitle.start, subtitle.end, subtitle.content) for subtitle in subtitles]

    # Initialize sliding window
    window = deque()
    words_in_window = 0
    hallucinations = []
    window_idx = 0

    # Process entries ONE AT A TIME
    for i in range(len(parts)):
        start_ts, end_ts, text = parts[i]
        words = detector.prepare(text).split()
        window.append((start_ts, end_ts, text, words))
        words_in_window += len(words)

        # Detect when window is full OR at end of file
        if i == len(parts) - 1 or words_in_window >= min_window_size:
            # DETECT: Concatenate window text and run detection
            window_text = " ".join([" ".join(words) for _, _, _, words in window])

            repetitions = detector.detect_hallucinations(window_text)

            # Extract timestamps and text directly from window
            for rep_start, rep_end, rep_k, rep_count in repetitions:
                word_offset = 0
                start_ts = None
                end_ts = None
                texts = []

                for entry_start_ts, entry_end_ts, entry_text, entry_words in window:
                    entry_word_count = len(entry_words)

                    # Find start timestamp
                    if start_ts is None and word_offset <= rep_start < word_offset + entry_word_count:
                        start_ts = entry_start_ts

                    # Collect text from entries with hallucination
                    if start_ts is not None and end_ts is None:
                        texts.append(entry_text)

                    # Find end timestamp
                    if word_offset < rep_end <= word_offset + entry_word_count:
                        end_ts = entry_end_ts
                        break

                    word_offset += entry_word_count

                # Build cleaned text for this hallucination
                cleaned_repetition = " ".join(window_text.split()[rep_start:rep_end])

                hallucinations.append(
                    {
                        "window_index": window_idx,
                        "window_word_count": words_in_window,
                        "start_timestamp": timedelta_to_srt_timestamp(start_ts or window[0][0]),
                        "end_timestamp": timedelta_to_srt_timestamp(end_ts or window[-1][1]),
                        "original_text": " ".join(texts).strip(),
                        "cleaned_text": cleaned_repetition,
                        "repetition_length": rep_k,
                        "repetition_count": rep_count,
                        "score": rep_k * rep_count,
                        "repetition_pattern": cleaned_repetition,
                        "window_text": window_text,
                    }
                )

            # SHRINK: Remove entries until window is at overlap size
            while words_in_window > min_window_size * (overlap_percent / 100):
                _, _, _, words = window.popleft()
                words_in_window -= len(words)

            window_idx += 1

    # Build result
    total_words = sum(len(detector.prepare(text).split()) for _, _, text in parts)

    return {
        "source_file": str(srt_file),
        "processed_at": datetime.now().isoformat(),
        "min_window_size": min_window_size,
        "overlap_percent": overlap_percent,
        "total_windows": window_idx + 1,
        "hallucinations_detected": hallucinations,
        "summary": {
            "total_hallucinations": len(hallucinations),
            "total_words_processed": total_words,
        },
    }


def process_srt_file(
    srt_file: Path,
    output_base_dir: Path,
    detector: RepetitionDetector,
    min_window_size: int,
    overlap_percent: float,
) -> tuple[bool, str | None, int]:
    """Process a single SRT file.

    Args:
        srt_file: Path to SRT file.
        output_base_dir: Base directory for output.
        detector: RepetitionDetector instance (uses SVM classifier for detection).
        min_window_size: Minimum window size in words.
        overlap_percent: Overlap percentage.

    Returns:
        Tuple of (success, error_message, hallucination_count).
    """
    try:
        result = detect_hallucinations_in_file(srt_file, detector, min_window_size, overlap_percent)

        # Create output directory (channel_name/transcript-analysis/)
        channel_name = srt_file.parent.name
        analysis_dir = output_base_dir / channel_name / "transcript-analysis"
        FSUtil.ensure_directory_exists(analysis_dir)

        # Write JSON output
        output_file = analysis_dir / f"{srt_file.stem}.json"
        FSUtil.write_json_file(output_file, result, create_parents=True)

        return (True, None, result["summary"]["total_hallucinations"])

    except srt.SRTParseError as e:
        return (False, f"SRT parse error: {e}", 0)
    except FileNotFoundError as e:
        return (False, f"File not found: {e}", 0)
    except Exception as e:
        return (False, f"Unexpected error: {e}", 0)


def print_file_result(filename: str, hallucination_count: int, example: dict[str, Any] | None = None) -> None:
    """Print console output for a file.

    Args:
        filename: Name of the SRT file.
        hallucination_count: Number of hallucinations detected.
        example: Example hallucination to display (if any).
    """
    if hallucination_count == 0:
        print(f"✅ {filename}: No hallucinations detected")
    else:
        print(f"🛑 {filename}: Found {hallucination_count} potential hallucination(s)")

        if example:
            print(f"  Example at {example['start_timestamp']}:")
            preview = example["original_text"][:100]
            if len(example["original_text"]) > 100:
                preview += "..."
            print(f"  {preview}")


def main() -> int:  # noqa: C901
    """Main entry point."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    # Load raw YAML to access hallucination_detection config
    with open(config_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Get hallucination detection config
    hallucination_config = config_data.get("hallucination_detection", {})
    config_min_window_size = hallucination_config.get("min_window_size")
    config_overlap_percent = hallucination_config.get("overlap_percent")

    if config_min_window_size is None or config_overlap_percent is None:
        print(
            "Error: hallucination_detection.min_window_size and overlap_percent must be configured in config.yaml",
            file=sys.stderr,
        )
        return 1

    # Parse arguments (allow CLI overrides)
    parser = argparse.ArgumentParser(
        description="Detect hallucinations in SRT transcripts using repetition detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                  # Use config.yaml settings
  %(prog)s --min-window-size 1000           # Override window size
  %(prog)s --overlap-percent 50             # Override overlap percentage
        """,
    )
    parser.add_argument(
        "--min-window-size",
        type=int,
        default=None,
        help=f"Minimum sliding window size in words (default: {config_min_window_size} from config)",
    )
    parser.add_argument(
        "--overlap-percent",
        type=float,
        default=None,
        help=f"Overlap percentage between windows (default: {config_overlap_percent} from config)",
    )
    args = parser.parse_args()

    # Use CLI args if provided, otherwise use config
    min_window_size = args.min_window_size if args.min_window_size is not None else config_min_window_size
    overlap_percent = args.overlap_percent if args.overlap_percent is not None else config_overlap_percent

    # Validate parameters
    if min_window_size < 1:
        print("Error: min-window-size must be >= 1", file=sys.stderr)
        return 1
    if not 0 <= overlap_percent < 100:
        print("Error: overlap-percent must be in [0, 100)", file=sys.stderr)
        return 1

    # Find all SRT files
    transcripts_dir = Path(__file__).parent.parent / "data" / "downloads" / "transcripts"
    if not transcripts_dir.exists():
        print(f"Error: Transcripts directory not found: {transcripts_dir}", file=sys.stderr)
        return 1

    srt_files = FSUtil.find_files_by_extension(transcripts_dir, ".srt", recursive=True)

    # Filter out macOS resource fork files (._* files)
    srt_files = [f for f in srt_files if not f.name.startswith("._")]

    if not srt_files:
        print("No SRT files found in transcripts directory", file=sys.stderr)
        return 1

    print(f"Found {len(srt_files)} SRT file(s) to process\n")

    # Initialize detector (SVM classifier decides what's a hallucination)
    config = Config(config_path)
    detector = RepetitionDetector(
        min_k=config.getRepetitionMinK(),
        min_repetitions=config.getRepetitionMinRepetitions(),
        config_path=config.getConfigPath(),
    )

    # Process each file
    total_files = 0
    files_with_hallucinations = 0
    total_hallucinations = 0
    failed_files = []
    all_hallucinations = []  # Collect all for score distribution

    for srt_file in srt_files:
        # Display relative path for cleaner output
        relative_path = srt_file.relative_to(transcripts_dir)
        print(f"Processing: {relative_path}")

        success, error_msg, hallucination_count = process_srt_file(
            srt_file,
            transcripts_dir,
            detector,
            min_window_size,
            overlap_percent,
        )

        total_files += 1

        if not success:
            print(f"  ❌ Failed: {error_msg}")
            failed_files.append((str(relative_path), error_msg))
        else:
            # Load the result to get an example if hallucinations were found
            example = None
            if hallucination_count > 0:
                channel_name = srt_file.parent.name
                analysis_dir = transcripts_dir / channel_name / "transcript-analysis"
                output_file = analysis_dir / f"{srt_file.stem}.json"

                # Read back the JSON to get an example and collect all hallucinations
                try:
                    import json

                    with output_file.open("r", encoding="utf-8") as f:
                        result_data = json.load(f)
                        if result_data["hallucinations_detected"]:
                            example = result_data["hallucinations_detected"][0]
                            # Collect all hallucinations for score distribution
                            all_hallucinations.extend(result_data["hallucinations_detected"])
                except Exception:
                    pass  # If we can't load the example, just skip it

                files_with_hallucinations += 1
                total_hallucinations += hallucination_count

            print_file_result(str(relative_path), hallucination_count, example)

        print()  # Blank line between files

    # Print summary
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Total files processed: {total_files}")
    print(f"Files with hallucinations: {files_with_hallucinations}")
    print(f"Total hallucinations detected: {total_hallucinations}")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")

    # Print score distribution if we have hallucinations
    if all_hallucinations:
        print("\n" + "=" * 50)
        print("Score Distribution (for analysis)")
        print("=" * 50)

        # Define score ranges for grouping
        ranges = [(11, 20, "Low"), (21, 50, "Medium"), (51, 100, "High"), (101, float("inf"), "Very High")]

        # Group hallucinations by score range
        for min_score, max_score, label in ranges:
            range_hallucinations = [
                h
                for h in all_hallucinations
                if min_score <= h.get("score", h.get("repetition_length", 0) * h.get("repetition_count", 1)) < max_score
            ]

            if not range_hallucinations:
                continue

            print(f"\n{label} Score ({min_score}-{max_score - 1 if max_score != float('inf') else '∞'}):")
            print(f"  Total patterns: {len(range_hallucinations)}")

            # Show top 3 unique examples by score
            seen_patterns = set()
            examples_shown = 0
            for h in sorted(range_hallucinations, key=lambda x: x.get("score", 0), reverse=True):
                pattern = h["repetition_pattern"][:60]
                if pattern not in seen_patterns and examples_shown < 3:
                    seen_patterns.add(pattern)
                    k = h.get("repetition_length", 0)
                    reps = h.get("repetition_count", 1)
                    score = h.get("score", k * reps)
                    ellipsis = "..." if len(h["repetition_pattern"]) > 60 else ""
                    print(f'  • Score {score:3d} = k={k:2d} × {reps:2d} reps: "{pattern}{ellipsis}"')
                    examples_shown += 1

        print("\nUsing SVM classifier for hallucination detection")
        print("Patterns are classified based on repetition count and sequence length.")

    return 1 if failed_files else 0


if __name__ == "__main__":
    sys.exit(main())
