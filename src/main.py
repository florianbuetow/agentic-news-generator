"""Agentic News Generator - Main Entry Point."""

import sys
from pathlib import Path

from src.agents.topic_segmentation.orchestrator import TopicSegmentationOrchestrator
from src.agents.topic_segmentation.srt_converter import srt_to_simplified_format
from src.config import Config


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

    # Find SRT files
    transcripts_dir = Path(__file__).parent.parent / "data" / "output" / "transcripts"
    if not transcripts_dir.exists():
        print(f"Transcripts directory not found: {transcripts_dir}")
        print("Please run transcription step first.")
        sys.exit(1)

    srt_files = list(transcripts_dir.glob("*.srt"))
    print(f"Found {len(srt_files)} transcript files")

    # Process each file
    success_count = 0
    failure_count = 0
    failures: list[tuple[str, str]] = []

    for srt_file in srt_files:
        print(f"\nProcessing: {srt_file.name}")

        try:
            # Read and convert SRT
            srt_content = srt_file.read_text(encoding="utf-8")
            simplified_transcript = srt_to_simplified_format(srt_content)

            # Extract metadata (TODO: get from metadata file)
            video_id = srt_file.stem

            # Segment transcript
            result = orchestrator.segment_transcript(
                video_id=video_id,
                video_title=video_id,  # TODO: Get from metadata
                channel_name="Unknown",  # TODO: Get from metadata
                simplified_transcript=simplified_transcript,
            )

            if result.success:
                success_count += 1
                print(f"  ✓ Success after {len(result.attempts)} attempt(s)")
            else:
                failure_count += 1
                failures.append((srt_file.name, result.failure_reason or "Unknown"))
                rating = result.best_attempt.critic_rating.rating if result.best_attempt and result.best_attempt.critic_rating else "N/A"
                print(f"  ✗ Failed: {result.failure_reason}")
                print(f"  Best attempt rating: {rating}")

        except Exception as e:
            failure_count += 1
            failures.append((srt_file.name, str(e)))
            print(f"  ✗ Unexpected error: {e}")

    # Report summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {success_count} succeeded, {failure_count} failed")

    if failures:
        print("\nFailures:")
        for filename, reason in failures:
            print(f"  - {filename}: {reason}")

    sys.exit(1 if failure_count > 0 else 0)


if __name__ == "__main__":
    main()
