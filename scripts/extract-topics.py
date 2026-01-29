#!/usr/bin/env python3
"""Topic extraction from segmented transcripts using LLM.

Step 3 of the three-step topic detection pipeline.
Reads _segmentation.json files and extracts topics/descriptions using an LLM agent.
Outputs _topics.json files.

Usage:
    uv run python scripts/extract-topics.py
    uv run python scripts/extract-topics.py --file path/to/transcript_segmentation.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import tiktoken
from pydantic import BaseModel, ConfigDict

from src.config import Config
from src.topic_detection.agents.topic_extraction_agent import TopicExtractionAgent
from src.topic_detection.segmentation.schemas import SegmentationOutput


def format_duration(seconds: float) -> str:
    """Format seconds as [hh:mm] duration."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"[{hours:02d}:{minutes:02d}]"


class SegmentTopics(BaseModel):
    """Topics extracted for a single segment."""

    segment_id: int
    start_timestamp: str
    end_timestamp: str
    high_level_topics: list[str]
    mid_level_topics: list[str]
    specific_topics: list[str]
    description: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class TranscriptTopics(BaseModel):
    """Complete topic detection result for a transcript."""

    source_file: str
    processed_at: str
    extraction_model: str
    total_segments: int
    segments: list[SegmentTopics]

    model_config = ConfigDict(frozen=True, extra="forbid")


def process_segments(segmentation_path: Path, config: Config) -> TranscriptTopics:
    """Process a segmentation file and extract topics.

    Args:
        segmentation_path: Path to the _segmentation.json file.
        config: Configuration object.

    Returns:
        TranscriptTopics with extracted topic information.
    """
    td_config = config.get_topic_detection_config()

    # 1. Load segmentation file
    print("  Loading segmentation file...")
    with open(segmentation_path, encoding="utf-8") as f:
        data = json.load(f)

    segmentation_data = SegmentationOutput.model_validate(data)
    print(f"  Found {segmentation_data.total_segments} segments to process")

    # 2. Create topic extraction agent and tokenizer
    print("  Initializing topic extraction agent...")
    llm_config = td_config.topic_detection_llm
    agent = TopicExtractionAgent(
        llm_config=llm_config,
        max_retries=llm_config.max_retries,
        retry_delay=llm_config.retry_delay,
    )

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # 3. Process each segment
    segment_results: list[SegmentTopics] = []
    for seg in segmentation_data.segments:
        token_count = len(tokenizer.encode(seg.text))
        print(f"    Processing segment {seg.segment_id}/{segmentation_data.total_segments} ({token_count} tokens)...")
        topics = agent.detect(seg.text)
        segment_results.append(
            SegmentTopics(
                segment_id=seg.segment_id,
                start_timestamp=seg.start_timestamp,
                end_timestamp=seg.end_timestamp,
                high_level_topics=topics.high_level_topics,
                mid_level_topics=topics.mid_level_topics,
                specific_topics=topics.specific_topics,
                description=topics.description,
            )
        )

    return TranscriptTopics(
        source_file=segmentation_data.source_file,
        processed_at=datetime.now().isoformat(),
        extraction_model=llm_config.model,
        total_segments=len(segment_results),
        segments=segment_results,
    )


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Extract topics from segmented transcripts using LLM")
    parser.add_argument(
        "--file",
        type=Path,
        help="Process a single _segmentation.json file instead of all files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if _topics.json already exists",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config(config_path)
    td_config = config.get_topic_detection_config()

    # Output directory
    output_dir = config.getDataDir() / td_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine files to process
    if args.file:
        # Single file mode
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
        if not args.file.name.endswith("_segmentation.json"):
            print(f"Error: Expected _segmentation.json file, got: {args.file}", file=sys.stderr)
            return 1
        segmentation_files = [args.file]
        base_dir = args.file.parent
    else:
        # Process all _segmentation.json files in output directory
        if not output_dir.exists():
            print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
            return 1

        segmentation_files = sorted(output_dir.rglob("*_segmentation.json"))
        segmentation_files = [f for f in segmentation_files if not f.name.startswith("._")]
        base_dir = output_dir

    if not segmentation_files:
        print("No _segmentation.json files found to process.")
        return 0

    print(f"Found {len(segmentation_files)} _segmentation.json file(s)")
    print(f"Output directory: {output_dir}")

    # Pre-scan to separate files that need processing from those to skip
    files_to_process: list[tuple[Path, Path, Path]] = []  # (input, output_subdir, output_path)
    skipped_count = 0

    for segmentation_file in segmentation_files:
        relative_path = segmentation_file.relative_to(base_dir)
        output_subdir = output_dir / relative_path.parent
        output_filename = relative_path.name.replace("_segmentation.json", "_topics.json")
        topics_output_path = output_subdir / output_filename

        if topics_output_path.exists() and not args.force:
            print(f"Skipping: {relative_path} (topics file already exists)")
            skipped_count += 1
        else:
            files_to_process.append((segmentation_file, output_subdir, topics_output_path))

    if not files_to_process:
        print()
        print("=" * 50)
        print(f"Completed: 0 succeeded, {skipped_count} skipped")
        return 0

    print()
    print(f"Processing {len(files_to_process)} file(s), {skipped_count} already processed")
    print()

    success_count = 0
    total_to_process = len(files_to_process)
    start_time = time.time()

    # Process each file with ETA tracking
    for file_idx, (segmentation_file, output_subdir, topics_output_path) in enumerate(files_to_process, start=1):
        relative_path = segmentation_file.relative_to(base_dir)
        progress_pct = (file_idx / total_to_process) * 100

        # Calculate ETA based on average time per processed file
        if file_idx > 1:
            elapsed = time.time() - start_time
            avg_time_per_file = elapsed / (file_idx - 1)
            remaining_files = total_to_process - file_idx + 1
            eta_seconds = avg_time_per_file * remaining_files
            eta_str = f" ETA {format_duration(eta_seconds)}"
        else:
            eta_str = ""

        print(f"Processing [{file_idx}/{total_to_process}] ({progress_pct:.0f}%){eta_str}: {relative_path}")

        topics_result = process_segments(segmentation_file, config)

        # Write topics JSON output immediately
        output_subdir.mkdir(parents=True, exist_ok=True)

        with open(topics_output_path, "w", encoding="utf-8") as f:
            json.dump(topics_result.model_dump(), f, indent=2, ensure_ascii=False)

        print(f"  → {topics_result.total_segments} segments extracted → {topics_output_path}")

        success_count += 1
        print()

    # Summary with total elapsed time
    total_elapsed = time.time() - start_time
    print("=" * 50)
    print(f"Completed: {success_count} succeeded, {skipped_count} skipped (elapsed: {format_duration(total_elapsed)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
