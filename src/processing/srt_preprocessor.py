"""SRT file preprocessing utilities.

This module provides functionality to preprocess SRT content,
converting it from SRT format to simplified format.
"""

from pathlib import Path

from src.processing.srt_converter import srt_to_simplified_format
from src.util import FSUtil


class SRTPreprocessor:
    """Preprocessor for SRT transcript content."""

    def process(self, srt_content: str) -> str:
        """Process SRT content and convert it to simplified format.

        Args:
            srt_content: Raw SRT format content.

        Returns:
            Simplified format content with timestamp prefixes.

        Raises:
            ValueError: If SRT content cannot be parsed or is invalid.
        """
        return srt_to_simplified_format(srt_content)


def process_srt_file(
    srt_file: Path,
    channel_output_dir: Path,
    preprocessor: SRTPreprocessor,
) -> tuple[bool, str | None]:
    """Process a single SRT file.

    Args:
        srt_file: Path to the SRT file.
        channel_output_dir: Output directory for this channel.
        preprocessor: SRT preprocessor instance.

    Returns:
        Tuple of (success, error_message).
    """
    try:
        # Read SRT content
        srt_content = FSUtil.read_text_file(srt_file)

        # Process with preprocessor
        simplified_content = preprocessor.process(srt_content)

        # Generate output filename (use original filename with .txt extension)
        output_filename = srt_file.stem + ".txt"
        output_path = channel_output_dir / output_filename

        # Write preprocessed content
        FSUtil.write_text_file(output_path, simplified_content, create_parents=True)

        print(f"    ✓ {srt_file.name} -> {output_filename}")
        return (True, None)

    except Exception as e:
        error_msg = str(e)
        print(f"    ✗ {srt_file.name}: {error_msg}")
        return (False, error_msg)


def process_channel_srt_files(
    channel_folder: Path,
    output_dir: Path,
    preprocessor: SRTPreprocessor,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Process all SRT files in a channel.

    Args:
        channel_folder: Path to the channel folder.
        output_dir: Base output directory.
        preprocessor: SRT preprocessor instance.

    Returns:
        Tuple of (processed_count, failed_count, failures).
    """
    channel_name = channel_folder.name
    print(f"\nProcessing channel: {channel_name}")

    # Find all .srt files recursively
    try:
        srt_files = FSUtil.find_files_by_extension(channel_folder, ".srt", recursive=True)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"  Error finding files: {e}")
        return (0, 0, [])

    if not srt_files:
        print(f"  No .srt files found in {channel_name}")
        return (0, 0, [])

    print(f"  Found {len(srt_files)} .srt file(s)")

    # Create output directory for this channel
    channel_output_dir = output_dir / channel_name
    FSUtil.ensure_directory_exists(channel_output_dir)

    # Process each SRT file
    processed_count = 0
    failed_count = 0
    failures: list[tuple[str, str]] = []

    for srt_file in srt_files:
        success, error = process_srt_file(srt_file, channel_output_dir, preprocessor)
        if success:
            processed_count += 1
        else:
            failed_count += 1
            if error:
                failures.append((f"{channel_name}/{srt_file.name}", error))

    return (processed_count, failed_count, failures)


def main() -> None:
    """Main function to preprocess all SRT files."""
    import sys

    # Define paths
    project_root = Path(__file__).parent.parent.parent
    transcripts_dir = project_root / "data" / "downloads" / "transcripts"
    output_dir = project_root / "data" / "downloads" / "transcripts-preprocessed"

    if not transcripts_dir.exists():
        print(f"Transcripts directory not found: {transcripts_dir}")
        sys.exit(1)

    # Find all channel folders
    try:
        channel_folders = FSUtil.find_directories(transcripts_dir, exclude_hidden=True)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not channel_folders:
        print(f"No channel folders found in {transcripts_dir}")
        sys.exit(1)

    print(f"Found {len(channel_folders)} channel folder(s)")

    # Initialize preprocessor
    preprocessor = SRTPreprocessor()

    # Process all channels
    total_processed = 0
    total_failed = 0
    all_failures: list[tuple[str, str]] = []

    for channel_folder in channel_folders:
        processed, failed, failures = process_channel_srt_files(channel_folder, output_dir, preprocessor)
        total_processed += processed
        total_failed += failed
        all_failures.extend(failures)

    # Report summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {total_processed} processed, {total_failed} failed")

    if all_failures:
        print("\nFailures:")
        for filepath, reason in all_failures:
            print(f"  - {filepath}: {reason}")

    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
