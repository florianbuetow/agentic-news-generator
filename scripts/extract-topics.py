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
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.config import Config
from src.topic_detection.agents.topic_extraction_agent import TopicExtractionAgent
from src.topic_detection.segmentation.schemas import SegmentationOutput


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

    # 2. Create topic extraction agent
    print("  Initializing topic extraction agent...")
    agent = TopicExtractionAgent(td_config.topic_detection_llm)

    # 3. Process each segment
    segment_results: list[SegmentTopics] = []
    for seg in segmentation_data.segments:
        print(f"    Processing segment {seg.segment_id}/{segmentation_data.total_segments}...")
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

    print(f"Found {len(segmentation_files)} _segmentation.json file(s) to process")
    print(f"Output directory: {output_dir}")
    print()

    success_count = 0

    # Process each segmentation file - write output immediately after each file
    for segmentation_file in segmentation_files:
        relative_path = segmentation_file.relative_to(base_dir)
        print(f"Processing: {relative_path}")

        topics_result = process_segments(segmentation_file, config)

        # Write topics JSON output immediately
        # Replace _segmentation.json with _topics.json
        output_subdir = output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_filename = relative_path.name.replace("_segmentation.json", "_topics.json")
        topics_output_path = output_subdir / output_filename

        with open(topics_output_path, "w", encoding="utf-8") as f:
            json.dump(topics_result.model_dump(), f, indent=2, ensure_ascii=False)

        print(f"  → {topics_result.total_segments} segments extracted → {topics_output_path}")

        success_count += 1
        print()

    # Summary
    print("=" * 50)
    print(f"Completed: {success_count} succeeded")

    return 0


if __name__ == "__main__":
    sys.exit(main())
