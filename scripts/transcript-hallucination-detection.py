#!/usr/bin/env python3
"""Detect hallucinations in SRT transcripts using repetition detection."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import srt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.util.fs_util import FSUtil
from src.utils.repetition_detector import RepetitionDetector


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


def create_sliding_windows(
    total_words: int, window_size: int, overlap_percent: float
) -> list[tuple[int, int]]:
    """Create sliding window boundaries.

    Args:
        total_words: Total number of words in the text.
        window_size: Number of words per window.
        overlap_percent: Percentage of overlap (0-100).

    Returns:
        List of (start_word_idx, end_word_idx) tuples.
    """
    if total_words < window_size:
        # If text is shorter than window, return single window
        return [(0, total_words)]

    step_size = int(window_size * (1 - overlap_percent / 100))
    step_size = max(1, step_size)  # Ensure at least 1 word step

    windows = []
    start = 0
    while start < total_words:
        end = min(start + window_size, total_words)
        windows.append((start, end))

        if end >= total_words:
            break

        start += step_size

    return windows


class WordToSubtitleMapper:
    """Maps word positions in cleaned text to SRT subtitle entries."""

    def __init__(self, subtitles: list[srt.Subtitle], detector: RepetitionDetector) -> None:
        """Initialize mapper with SRT subtitles and repetition detector.

        Args:
            subtitles: List of parsed SRT subtitle objects.
            detector: RepetitionDetector instance for text cleaning.
        """
        self.subtitles = subtitles
        self.detector = detector
        # Build the mapping: word_index -> (subtitle_index, word_offset_in_subtitle)
        self.word_to_subtitle_map: list[tuple[int, int]] = []
        self._build_mapping()

    def _build_mapping(self) -> None:
        """Build mapping from word position to subtitle entry."""
        for sub_idx, subtitle in enumerate(self.subtitles):
            # Clean the subtitle text the same way RepetitionDetector does
            cleaned = self.detector.prepare(subtitle.content)
            words = cleaned.split()

            # Each word gets mapped to (subtitle_index, word_offset_in_subtitle)
            for word_offset in range(len(words)):
                self.word_to_subtitle_map.append((sub_idx, word_offset))

    def get_timestamp_range(
        self, start_word_idx: int, end_word_idx: int
    ) -> tuple[timedelta, timedelta]:
        """Get timestamp range for a word range.

        Args:
            start_word_idx: Starting word index in the full cleaned text.
            end_word_idx: Ending word index in the full cleaned text (exclusive).

        Returns:
            Tuple of (start_timestamp, end_timestamp).
        """
        # Get the subtitle containing the start word
        start_sub_idx, _ = self.word_to_subtitle_map[start_word_idx]
        start_timestamp = self.subtitles[start_sub_idx].start

        # Get the subtitle containing the end word
        end_sub_idx, _ = self.word_to_subtitle_map[end_word_idx - 1]
        end_timestamp = self.subtitles[end_sub_idx].end

        return (start_timestamp, end_timestamp)

    def get_original_text(self, start_word_idx: int, end_word_idx: int) -> str:
        """Get original text (uncleaned) for a word range.

        Args:
            start_word_idx: Starting word index.
            end_word_idx: Ending word index (exclusive).

        Returns:
            Original text from the SRT subtitles.
        """
        start_sub_idx, _ = self.word_to_subtitle_map[start_word_idx]
        end_sub_idx, _ = self.word_to_subtitle_map[end_word_idx - 1]

        # Collect text from all subtitles in the range
        texts = [
            self.subtitles[i].content for i in range(start_sub_idx, end_sub_idx + 1)
        ]
        return " ".join(texts)


def detect_hallucinations_in_file(
    srt_file: Path,
    detector: RepetitionDetector,
    window_size: int,
    overlap_percent: float,
) -> dict[str, Any]:
    """Process a single SRT file to detect hallucinations.

    Args:
        srt_file: Path to SRT file.
        detector: RepetitionDetector instance (uses SVM classifier for detection).
        window_size: Window size in words.
        overlap_percent: Overlap percentage.

    Returns:
        Dictionary with detection results.
    """
    # Read and parse SRT
    srt_content = FSUtil.read_text_file(srt_file)
    subtitles = list(srt.parse(srt_content))

    # Build full text and mapper
    full_text_parts = [subtitle.content for subtitle in subtitles]
    full_original_text = " ".join(full_text_parts)
    cleaned_text = detector.prepare(full_original_text)

    mapper = WordToSubtitleMapper(subtitles, detector)

    # Create sliding windows
    words = cleaned_text.split()
    windows = create_sliding_windows(len(words), window_size, overlap_percent)

    # Process each window
    hallucinations = []

    for window_idx, (start_idx, end_idx) in enumerate(windows):
        window_text = " ".join(words[start_idx:end_idx])

        # Detect hallucinations (consecutive repetitions) in this window
        # Uses SVM classifier to determine what's a hallucination
        repetitions = detector.detect_hallucinations(window_text)

        for rep_start, rep_end, rep_k, rep_count in repetitions:
            # Map window-local indices to global indices
            global_start = start_idx + rep_start
            global_end = start_idx + rep_end

            # Get timestamps and text
            start_ts, end_ts = mapper.get_timestamp_range(global_start, global_end)
            original_text = mapper.get_original_text(global_start, global_end)
            cleaned_repetition = " ".join(words[global_start:global_end])

            # Calculate score
            score = rep_k * rep_count

            hallucinations.append(
                {
                    "window_index": window_idx,
                    "start_timestamp": timedelta_to_srt_timestamp(start_ts),
                    "end_timestamp": timedelta_to_srt_timestamp(end_ts),
                    "original_text": original_text.strip(),
                    "cleaned_text": cleaned_repetition,
                    "repetition_length": rep_k,
                    "repetition_count": rep_count,
                    "score": score,
                    "repetition_pattern": cleaned_repetition,
                    "window_text": window_text,
                }
            )

    # Build result
    result = {
        "source_file": str(srt_file),
        "processed_at": datetime.now().isoformat(),
        "window_size": window_size,
        "overlap_percent": overlap_percent,
        "total_windows": len(windows),
        "hallucinations_detected": hallucinations,
        "summary": {
            "total_hallucinations": len(hallucinations),
            "total_words_processed": len(words),
        },
    }

    return result


def process_srt_file(
    srt_file: Path,
    output_base_dir: Path,
    detector: RepetitionDetector,
    window_size: int,
    overlap_percent: float,
) -> tuple[bool, str | None, int]:
    """Process a single SRT file.

    Args:
        srt_file: Path to SRT file.
        output_base_dir: Base directory for output.
        detector: RepetitionDetector instance (uses SVM classifier for detection).
        window_size: Window size in words.
        overlap_percent: Overlap percentage.

    Returns:
        Tuple of (success, error_message, hallucination_count).
    """
    try:
        result = detect_hallucinations_in_file(
            srt_file, detector, window_size, overlap_percent
        )

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


def print_file_result(
    filename: str, hallucination_count: int, example: dict[str, Any] | None = None
) -> None:
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


def main() -> int:
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Detect hallucinations in SRT transcripts using repetition detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Use defaults (SVM classifier decides)
  %(prog)s --window-size 200            # Larger window for longer transcripts
  %(prog)s --overlap-percent 50         # More overlap between windows
        """,
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Sliding window size in words (default: 100)",
    )
    parser.add_argument(
        "--overlap-percent",
        type=float,
        default=30.0,
        help="Overlap percentage between windows (default: 30)",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.window_size < 1:
        print("Error: window-size must be >= 1", file=sys.stderr)
        return 1
    if not 0 <= args.overlap_percent < 100:
        print("Error: overlap-percent must be in [0, 100)", file=sys.stderr)
        return 1

    # Find all SRT files
    transcripts_dir = Path(__file__).parent.parent / "data" / "downloads" / "transcripts"
    if not transcripts_dir.exists():
        print(f"Error: Transcripts directory not found: {transcripts_dir}", file=sys.stderr)
        return 1

    srt_files = FSUtil.find_files_by_extension(transcripts_dir, ".srt", recursive=True)

    if not srt_files:
        print("No SRT files found in transcripts directory", file=sys.stderr)
        return 1

    print(f"Found {len(srt_files)} SRT file(s) to process\n")

    # Initialize detector with defaults (SVM classifier decides what's a hallucination)
    detector = RepetitionDetector()

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
            args.window_size,
            args.overlap_percent,
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
        ranges = [
            (11, 20, "Low"),
            (21, 50, "Medium"),
            (51, 100, "High"),
            (101, float('inf'), "Very High")
        ]

        # Group hallucinations by score range
        for min_score, max_score, label in ranges:
            range_hallucinations = [
                h for h in all_hallucinations
                if min_score <= h.get("score", h.get("repetition_length", 0) * h.get("repetition_count", 1)) < max_score
            ]

            if not range_hallucinations:
                continue

            print(f"\n{label} Score ({min_score}-{max_score-1 if max_score != float('inf') else '∞'}):")
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
                    print(f"  • Score {score:3d} = k={k:2d} × {reps:2d} reps: \"{pattern}{'...' if len(h['repetition_pattern']) > 60 else ''}\"")
                    examples_shown += 1

        print("\nUsing SVM classifier for hallucination detection")
        print("Patterns are classified based on repetition count and sequence length.")

    return 1 if failed_files else 0


if __name__ == "__main__":
    sys.exit(main())
