"""Agentic News Generator - Main Entry Point."""

import sys
from pathlib import Path

import tiktoken

from src.agents.topic_segmentation.models import AgentSegmentationResponse
from src.agents.topic_segmentation.orchestrator import TopicSegmentationOrchestrator
from src.config import Config
from src.util import FSUtil


def is_transcript_already_processed(
    video_id: str,
    channel_name: str,
    output_dir: Path,
) -> bool:
    """Check if a transcript has already been processed.

    Args:
        video_id: Video identifier (filename stem).
        channel_name: Name of the channel.
        output_dir: Base output directory for topic segmentation results.

    Returns:
        True if the output JSON file already exists, False otherwise.
    """
    output_file = output_dir / channel_name / f"{video_id}.json"
    return output_file.exists()


def process_transcript_file(
    transcript_file: Path,
    channel_name: str,
    orchestrator: TopicSegmentationOrchestrator,
    output_dir: Path,
) -> tuple[bool, str | None]:
    """Process a single transcript file.

    Args:
        transcript_file: Path to the transcript file.
        channel_name: Name of the channel.
        orchestrator: Topic segmentation orchestrator.
        output_dir: Base output directory for topic segmentation results.

    Returns:
        Tuple of (success, error_message).
    """
    try:
        # Extract video ID from filename
        video_id = transcript_file.stem

        # Check if already processed
        if is_transcript_already_processed(video_id, channel_name, output_dir):
            print(f"    ⊘ Skipping {transcript_file.name} (already processed)")
            return (True, None)

        # Read preprocessed transcript (already in simplified format)
        simplified_transcript = FSUtil.read_text_file(transcript_file)

        # Count tokens and characters
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(simplified_transcript))
        num_chars = len(simplified_transcript)

        print(f"    Processing: {transcript_file.name}, {num_tokens} tokens, {num_chars} chars...")

        # Define callback to save immediately after each agent response
        def save_agent_response(attempt_num: int, agent_response: AgentSegmentationResponse) -> None:
            """Save agent response immediately to disk."""
            output_file = output_dir / channel_name / f"{video_id}.json"
            output_data = {
                "video_id": video_id,
                "video_title": video_id,
                "channel_name": channel_name,
                "attempt": attempt_num,
                "segments": [seg.model_dump() for seg in agent_response.segments],
            }
            FSUtil.write_json_file(output_file, output_data, create_parents=True)

        # Segment transcript
        result = orchestrator.segment_transcript(
            simplified_transcript=simplified_transcript,
            on_agent_response=save_agent_response,
        )

        if not result.best_attempt:
            raise ValueError("Result must have best_attempt")
        if not result.best_attempt.critic_rating:
            raise ValueError("Best attempt must have critic_rating")

        segments = result.best_attempt.response.segments
        num_segments = len(segments)
        critic_rating = result.best_attempt.critic_rating

        if result.success:
            print(f"      ✓ Success after {len(result.attempts)} attempt(s)")
        else:
            print(f"      ✗ Failed: {result.failure_reason}")
            print(f"      Best attempt rating: {critic_rating.rating}")

        print(f"      Detected {num_segments} topic segment(s):")
        for i, seg in enumerate(segments, 1):
            print(f"        {i}. [{seg.topic}] {seg.summary}")

        # Save result with critic feedback to JSON file
        output_file = output_dir / channel_name / f"{video_id}.json"
        output_data = {
            "video_id": video_id,
            "video_title": video_id,
            "channel_name": channel_name,
            "success": result.success,
            "critic_rating": critic_rating.rating,
            "critic_pass": critic_rating.pass_,
            "critic_reasoning": critic_rating.reasoning,
            "critic_suggestions": critic_rating.improvement_suggestions,
            "segments": [seg.model_dump() for seg in segments],
        }
        FSUtil.write_json_file(output_file, output_data, create_parents=True)
        print(f"      Saved to: {output_file}")

        return (result.success, result.failure_reason if not result.success else None)

    except Exception as e:
        print(f"      ✗ Unexpected error: {e}")
        return (False, str(e))


def process_transcript_files(
    transcript_files: list[Path],
    channel_name: str,
    orchestrator: TopicSegmentationOrchestrator,
    output_dir: Path,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Process a list of transcript files.

    Args:
        transcript_files: List of transcript file paths to process.
        channel_name: Name of the channel.
        orchestrator: Topic segmentation orchestrator.
        output_dir: Base output directory for topic segmentation results.

    Returns:
        Tuple of (success_count, failure_count, failures).
        Failures are tuples of (file_path_string, error_message).
    """
    success_count = 0
    failure_count = 0
    failures: list[tuple[str, str]] = []

    for transcript_file in transcript_files:
        success, error = process_transcript_file(transcript_file, channel_name, orchestrator, output_dir)
        if success:
            success_count += 1
        else:
            failure_count += 1
            if error:
                failures.append((f"{channel_name}/{transcript_file.name}", error))

    return (success_count, failure_count, failures)


def process_channel(
    channel_folder: Path,
    orchestrator: TopicSegmentationOrchestrator,
    output_dir: Path,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Process all transcript files in a channel.

    Args:
        channel_folder: Path to the channel folder.
        orchestrator: Topic segmentation orchestrator.
        output_dir: Base output directory for topic segmentation results.

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

    return process_transcript_files(transcript_files, channel_name, orchestrator, output_dir)


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
    orchestrator = TopicSegmentationOrchestrator(ts_config=ts_config, config=config)

    # Find preprocessed transcripts directory
    transcripts_dir = Path(__file__).parent.parent / "data" / "downloads" / "transcripts-preprocessed"
    if not transcripts_dir.exists():
        print(f"Preprocessed transcripts directory not found: {transcripts_dir}")
        print("Please run the preprocessor first: uv run src/processing/srt_preprocessor.py")
        sys.exit(1)

    # Create output directory for topic segmentation results
    output_dir = Path(__file__).parent.parent / "data" / "downloads" / "transcripts-topics"
    FSUtil.ensure_directory_exists(output_dir)

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
        success, failure, failures = process_channel(channel_folder, orchestrator, output_dir)
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
