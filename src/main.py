"""Agentic News Generator - Main Entry Point."""

import sys
from pathlib import Path

import tiktoken

from src.agents.topic_segmentation.orchestrator import TopicSegmentationOrchestrator
from src.config import Config
from src.util import FSUtil


def process_transcript_file(
    transcript_file: Path,
    channel_name: str,
    orchestrator: TopicSegmentationOrchestrator,
) -> tuple[bool, str | None]:
    """Process a single transcript file.

    Args:
        transcript_file: Path to the transcript file.
        channel_name: Name of the channel.
        orchestrator: Topic segmentation orchestrator.

    Returns:
        Tuple of (success, error_message).
    """
    try:
        # Read preprocessed transcript (already in simplified format)
        simplified_transcript = FSUtil.read_text_file(transcript_file)

        # Count tokens and characters
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(simplified_transcript))
        num_chars = len(simplified_transcript)

        print(f"    Processing: {transcript_file.name}, {num_tokens} tokens, {num_chars} chars...")

        # Extract video ID from filename
        video_id = transcript_file.stem

        # Segment transcript
        result = orchestrator.segment_transcript(
            video_id=video_id,
            video_title=video_id,  # TODO: Get from metadata
            channel_name=channel_name,
            simplified_transcript=simplified_transcript,
        )

        if result.success:
            if not result.best_attempt:
                raise ValueError("Successful result must have best_attempt")
            num_segments = len(result.best_attempt.response.segments)
            segment_summaries = [seg.summary for seg in result.best_attempt.response.segments]
            print(f"      ✓ Success after {len(result.attempts)} attempt(s)")
            print(f"      Detected {num_segments} topic segment(s):")
            for i, summary in enumerate(segment_summaries, 1):
                print(f"        {i}. {summary}")
            return (True, None)

        # Handle failure
        if not result.best_attempt:
            raise ValueError("Failed result must have best_attempt")
        if not result.best_attempt.critic_rating:
            raise ValueError("Best attempt must have critic_rating")
        rating = result.best_attempt.critic_rating.rating
        print(f"      ✗ Failed: {result.failure_reason}")
        print(f"      Best attempt rating: {rating}")
        return (False, result.failure_reason or "Unknown")

    except Exception as e:
        print(f"      ✗ Unexpected error: {e}")
        return (False, str(e))


def process_channel(
    channel_folder: Path,
    orchestrator: TopicSegmentationOrchestrator,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Process all transcript files in a channel.

    Args:
        channel_folder: Path to the channel folder.
        orchestrator: Topic segmentation orchestrator.

    Returns:
        Tuple of (success_count, failure_count, failures).
    """
    channel_name = channel_folder.name
    print(f"\nProcessing channel: {channel_name}")

    # Find all .txt files in this channel
    try:
        transcript_files = FSUtil.find_files_by_extension(channel_folder, ".txt", recursive=False)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"  Error finding files: {e}")
        return (0, 0, [])

    if not transcript_files:
        print(f"  No transcript files found in {channel_name}")
        return (0, 0, [])

    print(f"  Found {len(transcript_files)} transcript file(s)")

    # Process each transcript file
    success_count = 0
    failure_count = 0
    failures: list[tuple[str, str]] = []

    for transcript_file in transcript_files:
        success, error = process_transcript_file(transcript_file, channel_name, orchestrator)
        if success:
            success_count += 1
        else:
            failure_count += 1
            if error:
                failures.append((f"{channel_name}/{transcript_file.name}", error))

    return (success_count, failure_count, failures)


def main() -> None:
    """Main function to run topic segmentation."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config(config_path)

    try:
        ts_config = config.get_topic_segmentation_config()
    except KeyError as e:
        print(f"Configuration error: {e}")
        print("Please ensure topic_segmentation is configured in config.yaml")
        sys.exit(1)

    # Initialize orchestrator
    orchestrator = TopicSegmentationOrchestrator(config=ts_config)

    # Find preprocessed transcripts directory
    transcripts_dir = Path(__file__).parent.parent / "data" / "downloads" / "transcripts-preprocessed"
    if not transcripts_dir.exists():
        print(f"Preprocessed transcripts directory not found: {transcripts_dir}")
        print("Please run the preprocessor first: uv run src/processing/srt_preprocessor.py")
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

    # Process all channels
    total_success = 0
    total_failure = 0
    all_failures: list[tuple[str, str]] = []

    for channel_folder in channel_folders:
        success, failure, failures = process_channel(channel_folder, orchestrator)
        total_success += success
        total_failure += failure
        all_failures.extend(failures)

    # Report summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {total_success} succeeded, {total_failure} failed")

    if all_failures:
        print("\nFailures:")
        for filename, reason in all_failures:
            print(f"  - {filename}: {reason}")

    sys.exit(1 if total_failure > 0 else 0)


if __name__ == "__main__":
    main()
