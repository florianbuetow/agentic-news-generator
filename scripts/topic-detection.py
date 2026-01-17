#!/usr/bin/env python3
"""Topic detection pipeline.

Reads cleaned SRT files, performs embedding-based segmentation,
then extracts topics and descriptions using an LLM agent.

Usage:
    uv run python scripts/topic-detection.py
    uv run python scripts/topic-detection.py --file path/to/transcript.srt
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.config import Config, TopicDetectionConfig
from src.topic_detection.agents.topic_extraction_agent import TopicExtractionAgent
from src.topic_detection.embedding.factory import EmbeddingGeneratorFactory
from src.topic_detection.segmentation.data_types import ChunkData, Segment
from src.topic_detection.segmentation.segmenter import SlidingWindowTopicSegmenter
from src.util.srt_util import SRTEntry, SRTUtil


@dataclass(frozen=True)
class SegmentWithTimestamps:
    """A segment with mapped start and end timestamps."""

    start: timedelta
    end: timedelta
    text: str
    start_token: int
    end_token: int


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


def format_timedelta(td: timedelta) -> str:
    """Format a timedelta as SRT timestamp string (HH:MM:SS,mmm)."""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def map_to_timestamps(segments: list[Segment], entries: list[SRTEntry]) -> list[SegmentWithTimestamps]:
    """Map segment token boundaries to SRT timestamps.

    Args:
        segments: Segments from the segmenter with start_token/end_token.
        entries: SRT entries with word position tracking.

    Returns:
        List of segments with mapped timestamps.
    """
    result: list[SegmentWithTimestamps] = []
    for seg in segments:
        start_time = SRTUtil.word_position_to_timestamp(seg.start_token, entries)
        # Use end_token - 1 for inclusive end position
        end_pos = seg.end_token - 1 if seg.end_token > 0 else 0
        end_time = SRTUtil.word_position_to_timestamp(end_pos, entries)
        result.append(
            SegmentWithTimestamps(
                start=start_time,
                end=end_time,
                text=seg.text,
                start_token=seg.start_token,
                end_token=seg.end_token,
            )
        )
    return result


def create_embeddings_data(
    chunk_data: ChunkData,
    entries: list[SRTEntry],
    td_config: TopicDetectionConfig,
) -> dict[str, object]:
    """Create embeddings data structure for JSON export.

    Args:
        chunk_data: ChunkData with embeddings and chunk positions.
        entries: SRT entries for timestamp mapping.
        td_config: Topic detection configuration.

    Returns:
        Dictionary with embedding model info and per-window embeddings with timestamps.
    """
    sw_config = td_config.sliding_window
    windows: list[dict[str, object]] = []

    for offset, embedding in zip(chunk_data.chunk_positions, chunk_data.embeddings, strict=True):
        # Calculate end position (offset + window_size, but capped at last word)
        end_offset = offset + sw_config.window_size
        # Use offset for start timestamp, end_offset - 1 for end timestamp (inclusive)
        start_time = SRTUtil.word_position_to_timestamp(offset, entries)
        end_time = SRTUtil.word_position_to_timestamp(end_offset - 1, entries)

        windows.append(
            {
                "offset": offset,
                "length": sw_config.window_size,
                "embed": embedding.tolist(),
                "start_timestamp": format_timedelta(start_time),
                "end_timestamp": format_timedelta(end_time),
            }
        )

    return {
        "embedding_model": td_config.embedding.model_name,
        "window_size": sw_config.window_size,
        "stride": sw_config.stride,
        "windows": windows,
    }


def process_transcript(srt_path: Path, config: Config) -> tuple[TranscriptTopics, dict[str, object] | None]:
    """Process a single transcript file.

    Args:
        srt_path: Path to the cleaned SRT file.
        config: Configuration object.

    Returns:
        Tuple of (TranscriptTopics, embeddings_data or None if no chunk_data).
    """
    td_config = config.get_topic_detection_config()

    # 1. Parse SRT
    print("  Parsing SRT file...")
    entries = SRTUtil.parse_srt_file(srt_path)
    text = SRTUtil.entries_to_text(entries)
    total_words = SRTUtil.get_total_words(entries)
    print(f"  Found {len(entries)} SRT entries, {total_words} total words")

    # 2. Create embedding generator from config (via factory)
    print("  Creating embedding generator...")
    embedding_generator = EmbeddingGeneratorFactory.create(td_config.embedding)

    # 3. Run sliding window segmenter (with injected embedding generator)
    print("  Running topic segmentation...")
    sw_config = td_config.sliding_window
    segmenter = SlidingWindowTopicSegmenter(
        embedding_generator=embedding_generator,
        window_size=sw_config.window_size,
        stride=sw_config.stride,
        threshold_method=sw_config.threshold_method,
        threshold_value=sw_config.threshold_value,
        min_segment_tokens=sw_config.min_segment_tokens,
        smoothing_passes=sw_config.smoothing_passes,
    )
    result = segmenter.segment(text)
    print(f"  Detected {len(result.segments)} segments")

    # 4. Create embeddings data if chunk_data is available
    embeddings_data: dict[str, object] | None = None
    if result.chunk_data is not None:
        embeddings_data = create_embeddings_data(result.chunk_data, entries, td_config)
        print(f"  Created embeddings for {len(result.chunk_data.chunk_positions)} windows")

    # 5. Map segments to timestamps
    segments_with_timestamps = map_to_timestamps(result.segments, entries)

    # 6. Run topic detection on each segment
    print("  Extracting topics from segments...")
    agent = TopicExtractionAgent(td_config.topic_detection_llm)
    segment_results: list[SegmentTopics] = []
    for i, seg in enumerate(segments_with_timestamps):
        print(f"    Processing segment {i + 1}/{len(segments_with_timestamps)}...")
        topics = agent.detect(seg.text)
        segment_results.append(
            SegmentTopics(
                segment_id=i + 1,
                start_timestamp=format_timedelta(seg.start),
                end_timestamp=format_timedelta(seg.end),
                high_level_topics=topics.high_level_topics,
                mid_level_topics=topics.mid_level_topics,
                specific_topics=topics.specific_topics,
                description=topics.description,
            )
        )

    transcript_topics = TranscriptTopics(
        source_file=str(srt_path),
        processed_at=datetime.now().isoformat(),
        total_segments=len(segment_results),
        segments=segment_results,
    )

    return transcript_topics, embeddings_data


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Run topic detection on cleaned SRT transcripts")
    parser.add_argument(
        "--file",
        type=Path,
        help="Process a single SRT file instead of all files",
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
        srt_files = [args.file]
        # For single file, use file name as relative path
        base_dir = args.file.parent
    else:
        # Process all cleaned SRT files
        cleaned_dir = config.getDataDownloadsTranscriptsCleanedDir()
        if not cleaned_dir.exists():
            print(f"Error: Cleaned transcripts directory not found: {cleaned_dir}", file=sys.stderr)
            return 1

        # Find all SRT files recursively
        srt_files = sorted(cleaned_dir.rglob("*.srt"))
        srt_files = [f for f in srt_files if not f.name.startswith("._")]
        base_dir = cleaned_dir

    if not srt_files:
        print("No SRT files found to process.")
        return 0

    print(f"Found {len(srt_files)} SRT file(s) to process")
    print(f"Output directory: {output_dir}")
    print()

    success_count = 0

    # Process each transcript - fail fast on first error
    for srt_file in srt_files:
        relative_path = srt_file.relative_to(base_dir)
        print(f"Processing: {relative_path}")

        topics_result, embeddings_data = process_transcript(srt_file, config)

        # Write topics JSON output
        topics_output_path = output_dir / relative_path.with_suffix(".topics.json")
        topics_output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(topics_output_path, "w", encoding="utf-8") as f:
            json.dump(topics_result.model_dump(), f, indent=2, ensure_ascii=False)

        print(f"  → {topics_result.total_segments} segments extracted → {topics_output_path}")

        # Write embeddings JSON output if available
        if embeddings_data is not None:
            embeddings_output_path = output_dir / relative_path.with_suffix(".embeddings.json")
            with open(embeddings_output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
            print(f"  → embeddings saved → {embeddings_output_path}")

        success_count += 1

        print()

    # Summary
    print("=" * 50)
    print(f"Completed: {success_count} succeeded")

    return 0


if __name__ == "__main__":
    sys.exit(main())
