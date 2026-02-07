#!/usr/bin/env python3
"""Remove hallucinations from SRT transcripts using deterministic string replacement."""

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import srt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.processing.repetition_detector import RepetitionDetector
from src.processing.srt_converter import srt_to_simplified_format
from src.util.fs_util import FSUtil

# Whitespace normalization pattern (same as RepetitionDetector)
_WS = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and remove invisible Unicode characters.

    Same normalization as used in RepetitionDetector.prepare().
    """
    # Remove invisible Unicode formatting characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t ")
    # Normalize whitespace to single spaces
    return _WS.sub(" ", text).strip()


def remove_consecutive_repetitions(text: str, pattern: str, repetition_count: int) -> str:
    """Remove consecutive repetitions of a pattern using simple string replacement.

    Args:
        text: Text containing repetitions.
        pattern: The repeating pattern to remove.
        repetition_count: How many times the pattern repeats consecutively.

    Returns:
        Cleaned text with pattern reduced to single occurrence.
    """
    # Normalize the text and pattern (same as hallucination detection does)
    normalized_text = normalize_whitespace(text)
    normalized_pattern = normalize_whitespace(pattern)

    # Build the repeated pattern string (pattern repeated N times with spaces)
    repeated_pattern = (normalized_pattern + " ") * repetition_count
    repeated_pattern = repeated_pattern.strip()

    # Replace the repeated pattern with single occurrence in normalized text
    cleaned_normalized = normalized_text.replace(repeated_pattern, normalized_pattern)

    return cleaned_normalized


def setup_environment() -> tuple[Config, Path, Path, Path, Path, RepetitionDetector] | int:
    """Load configuration and initialize environment.

    Returns:
        Tuple of (config, data_dir, transcripts_dir, hallucinations_dir, output_dir, detector)
        or error code if setup fails.
    """
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    config = Config(config_path)

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / config.getDataDir()
    transcripts_dir = base_dir / config.getDataDownloadsTranscriptsDir()
    hallucinations_dir = base_dir / config.getDataDownloadsTranscriptsHallucinationsDir()
    output_dir = base_dir / config.getDataDownloadsTranscriptsCleanedDir()

    if not transcripts_dir.exists():
        print(f"Error: Transcripts directory not found: {transcripts_dir}", file=sys.stderr)
        return 1
    if not hallucinations_dir.exists():
        print(f"Error: Hallucinations directory not found: {hallucinations_dir}", file=sys.stderr)
        return 1

    try:
        detector = RepetitionDetector(
            min_k=config.getRepetitionMinK(),
            min_repetitions=config.getRepetitionMinRepetitions(),
            config=config,
        )
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return config, data_dir, transcripts_dir, hallucinations_dir, output_dir, detector


def process_single_file(
    srt_file: Path,
    data_dir: Path,
    output_dir: Path,
    hallucinations_lookup: dict,
    detector: RepetitionDetector,
) -> tuple[str, bool, str | None]:
    """Process a single SRT file.

    Returns:
        Tuple of (file_type, success, error_message) where:
        - file_type: "cleaned", "copied", or "skipped"
        - success: True if successful
        - error_message: Error message if failed, None otherwise
    """
    relative_path = srt_file.relative_to(data_dir)
    lookup_key = str(relative_path)

    channel_name = srt_file.parent.name
    output_channel_dir = output_dir / channel_name
    output_file = output_channel_dir / srt_file.name

    # Skip empty SRT files
    if srt_file.stat().st_size == 0:
        print(f"  ⚠ Skipping empty file: {srt_file.name}")
        return "skipped", True, None

    # Skip if output already exists and is up-to-date
    if output_file.exists() and output_file.stat().st_mtime >= srt_file.stat().st_mtime:
        return "skipped", True, None

    if lookup_key in hallucinations_lookup:
        print(f"  Cleaning: {srt_file.name}")
        hallucinations = hallucinations_lookup[lookup_key]
        success = process_srt_with_hallucinations(srt_file, output_file, hallucinations, detector)
        if success:
            print(f"    ✓ Cleaned {len(hallucinations)} hallucination(s)")
            return "cleaned", True, None
        else:
            print("    ✗ Failed to clean hallucinations")
            return "cleaned", False, "Failed to clean hallucinations"
    else:
        print(f"  Copying: {srt_file.name}")
        copy_srt_unchanged(srt_file, output_file)
        return "copied", True, None


def print_summary(
    total_files: int,
    cleaned_files: int,
    copied_files: int,
    skipped_files: int,
    failed_files: list[tuple[str, str]],
) -> None:
    """Print processing summary."""
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Total files: {total_files}")
    print(f"Cleaned: {cleaned_files}")
    print(f"Copied: {copied_files}")
    print(f"Skipped (up-to-date): {skipped_files}")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")


def main() -> int:
    """Main entry point.

    Returns:
        0 if all files processed successfully, 1 if any failures.
    """
    # Setup environment
    setup_result = setup_environment()
    if isinstance(setup_result, int):
        return setup_result
    config, data_dir, transcripts_dir, hallucinations_dir, output_dir, detector = setup_result

    # Phase 1: Build hallucinations lookup
    hallucinations_lookup = build_hallucinations_lookup(hallucinations_dir, data_dir)

    print(f"Phase 1: Found {len(hallucinations_lookup)} file(s) with hallucinations\n")

    # Phase 2: Process transcripts
    srt_files = FSUtil.find_files_by_extension(transcripts_dir, ".srt", recursive=True)
    srt_files = [f for f in srt_files if not f.name.startswith("._")]

    if not srt_files:
        print("No SRT files found in transcripts directory", file=sys.stderr)
        return 1

    print(f"Phase 2: Processing {len(srt_files)} transcript file(s)\n")

    failed_files = []
    cleaned_files = 0
    copied_files = 0
    skipped_files = 0

    for srt_file in srt_files:
        try:
            file_type, success, error = process_single_file(srt_file, data_dir, output_dir, hallucinations_lookup, detector)

            if success:
                if file_type == "cleaned":
                    cleaned_files += 1
                elif file_type == "copied":
                    copied_files += 1
                elif file_type == "skipped":
                    skipped_files += 1
            else:
                relative_path = srt_file.relative_to(data_dir)
                failed_files.append((str(relative_path), error or "Unknown error"))

        except Exception as e:
            relative_path = srt_file.relative_to(data_dir)
            failed_files.append((str(relative_path), str(e)))
            print(f"  ✗ Error: {e}\n")

    print_summary(len(srt_files), cleaned_files, copied_files, skipped_files, failed_files)

    # Phase 3: Convert cleaned SRT files to simplified text format
    print("\nPhase 3: SRT → TXT conversion\n")
    txt_converted, txt_skipped, txt_failed = convert_srt_to_txt(output_dir)
    print(f"\n  Converted: {txt_converted}")
    print(f"  Skipped (up-to-date): {txt_skipped}")
    if txt_failed:
        print(f"  Failed: {txt_failed}")

    return 1 if failed_files else 0


def build_hallucinations_lookup(hallucinations_dir: Path, data_dir: Path) -> dict[str, list[dict]]:
    """Build lookup dictionary from hallucination reports.

    Args:
        hallucinations_dir: Directory containing hallucination reports.
        data_dir: Data directory to use as base for relative paths.

    Returns:
        Dictionary mapping source file paths (relative to data_dir) to lists of hallucination objects.
    """
    hallucinations_lookup = defaultdict(list)

    # Find all JSON files in hallucinations directory
    json_files = FSUtil.find_files_by_extension(hallucinations_dir, ".json", recursive=True)
    json_files = [f for f in json_files if not f.name.startswith("._")]

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as fh:
                data = json.load(fh)

                if data["summary"]["total_hallucinations"] > 0:
                    # source_file in JSON is already relative to data_dir (e.g., "downloads/transcripts/...")
                    # Use it directly as the key
                    key = data["source_file"]

                    value = data["hallucinations_detected"]
                    hallucinations_lookup[key] = value
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse {json_file}: {e}", file=sys.stderr)
            continue

    return dict(hallucinations_lookup)


def process_srt_with_hallucinations(
    srt_file: Path,
    output_file: Path,
    hallucinations: list[dict],
    detector: RepetitionDetector,
) -> bool:
    """Process SRT file with hallucinations and write cleaned version.

    Args:
        srt_file: Input SRT file path.
        output_file: Output SRT file path.
        hallucinations: List of hallucination objects.
        detector: Repetition detector for validation.

    Returns:
        True if successful, False otherwise.
    """
    # Parse SRT file
    srt_content = FSUtil.read_text_file(srt_file)
    subtitles = list(srt.parse(srt_content))

    # Process each hallucination
    for idx, hallucination in enumerate(hallucinations, start=1):
        # Extract timestamps
        start_timestamp = hallucination["start_timestamp"]
        end_timestamp = hallucination["end_timestamp"]

        print(f"    Cleaning hallucination {idx}/{len(hallucinations)} at {start_timestamp}—{end_timestamp}...")

        # Convert to timedelta
        start_td = srt.srt_timestamp_to_timedelta(start_timestamp)
        end_td = srt.srt_timestamp_to_timedelta(end_timestamp)

        # Collect text segments within window
        window_text_parts = []
        affected_indices = []

        for subtitle_idx, subtitle in enumerate(subtitles):
            # Check if subtitle overlaps with hallucination window
            if subtitle.start >= start_td and subtitle.end <= end_td:
                window_text_parts.append(subtitle.content)
                affected_indices.append(subtitle_idx)

        if not affected_indices:
            # No segments found in window (edge case)
            print(f"    ⚠ Warning: No segments found in hallucination window {start_timestamp}—{end_timestamp}")
            continue

        # Clean the text
        window_content = " ".join(window_text_parts)

        # Validate: Check hallucination exists BEFORE cleaning
        hallucinations_before = detector.detect_hallucinations(window_content)
        if not hallucinations_before:
            print("    ⚠ Warning: No hallucination detected in window before cleaning (false positive?)")

        # Get hallucination info for deterministic cleaning
        hallucination_pattern = hallucination.get("repetition_pattern", "")
        hallucination_count = hallucination.get("repetition_count", 0)

        # Use deterministic regex replacement instead of LLM
        if hallucination_pattern and hallucination_count >= 2:
            window_content_clean = remove_consecutive_repetitions(window_content, hallucination_pattern, hallucination_count)

            # Validate: Check hallucination is GONE AFTER cleaning
            hallucinations_after = detector.detect_hallucinations(window_content_clean)
            if hallucinations_after:
                print(f"    ⚠ Warning: Hallucination still detected after cleaning ({len(hallucinations_after)} pattern(s))")
            else:
                print("    ✓ Cleaned successfully (verified: no hallucinations detected)")
        else:
            print("    ⚠ Warning: Invalid hallucination data - skipping")
            continue

        # Replace text in affected segments
        # Strategy: Put all cleaned text in first segment, clear others
        subtitles[affected_indices[0]] = srt.Subtitle(
            index=subtitles[affected_indices[0]].index,
            start=subtitles[affected_indices[0]].start,
            end=subtitles[affected_indices[0]].end,
            content=window_content_clean,
        )

        # Clear remaining segments in window
        for idx in affected_indices[1:]:
            subtitles[idx] = srt.Subtitle(
                index=subtitles[idx].index,
                start=subtitles[idx].start,
                end=subtitles[idx].end,
                content="",  # Empty content
            )

    # Remove empty subtitles and reindex
    subtitles = [s for s in subtitles if s.content.strip()]
    for new_idx, subtitle in enumerate(subtitles, start=1):
        subtitles[new_idx - 1] = srt.Subtitle(
            index=new_idx,
            start=subtitle.start,
            end=subtitle.end,
            content=subtitle.content,
        )

    # Write cleaned SRT
    cleaned_srt_content = srt.compose(subtitles)
    FSUtil.write_text_file(output_file, cleaned_srt_content, create_parents=True)

    return True


def convert_srt_to_txt(output_dir: Path) -> tuple[int, int, int]:
    """Convert all cleaned SRT files in output_dir to simplified text format.

    Skips SRT files whose .txt is already up-to-date (newer than the .srt).

    Args:
        output_dir: Directory containing cleaned SRT files.

    Returns:
        Tuple of (converted_count, skipped_count, failed_count).
    """
    srt_files = FSUtil.find_files_by_extension(output_dir, ".srt", recursive=True)
    srt_files = [f for f in srt_files if not f.name.startswith("._")]

    converted = 0
    skipped = 0
    failed = 0

    for srt_file in srt_files:
        txt_file = srt_file.with_suffix(".txt")
        if txt_file.exists() and txt_file.stat().st_mtime >= srt_file.stat().st_mtime:
            skipped += 1
            continue

        if srt_file.stat().st_size == 0:
            print(f"  ⚠ Skipping empty file: {srt_file.name}")
            failed += 1
            continue

        try:
            srt_content = FSUtil.read_text_file(srt_file)
            txt_content = srt_to_simplified_format(srt_content)
            FSUtil.write_text_file(txt_file, txt_content, create_parents=True)
            print(f"  ✓ {srt_file.name} → .txt")
            converted += 1
        except Exception as e:
            print(f"  ✗ {srt_file.name}: {e}", file=sys.stderr)
            failed += 1

    return converted, skipped, failed


def copy_srt_unchanged(srt_file: Path, output_file: Path) -> None:
    """Copy SRT file unchanged to output directory.

    Args:
        srt_file: Input SRT file path.
        output_file: Output SRT file path.
    """
    content = FSUtil.read_text_file(srt_file)
    FSUtil.write_text_file(output_file, content, create_parents=True)


if __name__ == "__main__":
    sys.exit(main())
