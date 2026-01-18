#!/usr/bin/env python3
"""Detect topic boundaries from pre-computed embeddings.

Step 2 of the three-step topic detection pipeline.
Reads _embeddings.json files, calculates similarity between adjacent windows,
detects topic boundaries, and outputs _segmentation.json files.

Usage:
    uv run python scripts/detect-boundaries.py
    uv run python scripts/detect-boundaries.py --file path/to/transcript_embeddings.json
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.config import Config
from src.topic_detection.embedding.factory import EmbeddingGeneratorFactory
from src.topic_detection.segmentation.data_types import ChunkData
from src.topic_detection.segmentation.schemas import (
    EmbeddingsOutput,
    SegmentationConfigData,
    SegmentationOutput,
    SegmentData,
    SRTEntryData,
)
from src.topic_detection.segmentation.segmenter import SlidingWindowTopicSegmenter
from src.util.srt_util import SRTEntry


def parse_timestamp(timestamp_str: str) -> timedelta:
    """Parse SRT timestamp string (HH:MM:SS,mmm) to timedelta."""
    parts = timestamp_str.replace(",", ":").split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    milliseconds = int(parts[3])
    return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)


def format_timedelta(td: timedelta) -> str:
    """Format a timedelta as SRT timestamp string (HH:MM:SS,mmm)."""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def srt_entry_data_to_entry(entry_data: SRTEntryData) -> SRTEntry:
    """Convert SRTEntryData back to SRTEntry for timestamp mapping."""
    return SRTEntry(
        index=entry_data.index,
        start=parse_timestamp(entry_data.start_timestamp),
        end=parse_timestamp(entry_data.end_timestamp),
        content=entry_data.content,
        word_start=entry_data.word_start,
        word_end=entry_data.word_end,
    )


def word_position_to_timestamp(word_pos: int, entries: list[SRTEntry]) -> timedelta:
    """Map a word position back to the corresponding SRT timestamp."""
    for entry in entries:
        if entry.word_start <= word_pos < entry.word_end:
            return entry.start
    return entries[-1].end if entries else timedelta(0)


def process_embeddings(embeddings_path: Path, data_dir: Path, config: Config) -> SegmentationOutput:
    """Process an embeddings file and detect topic boundaries.

    Args:
        embeddings_path: Path to the _embeddings.json file.
        data_dir: Data directory for computing relative paths.
        config: Configuration object.

    Returns:
        SegmentationOutput with detected segments and timestamps.
    """
    td_config = config.get_topic_detection_config()
    sw_config = td_config.sliding_window

    # 1. Load embeddings file
    print("  Loading embeddings file...")
    with open(embeddings_path, encoding="utf-8") as f:
        data = json.load(f)

    embeddings_data = EmbeddingsOutput.model_validate(data)
    print(f"  Found {len(embeddings_data.windows)} windows, {embeddings_data.total_words} words")

    # 2. Reconstruct ChunkData from loaded embeddings
    embeddings_array = np.array(
        [w.embedding for w in embeddings_data.windows],
        dtype=np.float32,
    )
    chunk_positions = [w.offset for w in embeddings_data.windows]
    chunk_data = ChunkData(embeddings=embeddings_array, chunk_positions=chunk_positions)

    # 3. Reconstruct SRT entries for timestamp mapping
    srt_entries = [srt_entry_data_to_entry(e) for e in embeddings_data.srt_entries]

    # 4. Create embedding generator (needed for segmenter initialization, but not used)
    embedding_generator = EmbeddingGeneratorFactory.create(td_config.embedding)

    # 5. Create segmenter and perform boundary detection
    print("  Detecting boundaries...")
    segmenter = SlidingWindowTopicSegmenter(
        embedding_generator=embedding_generator,
        window_size=sw_config.window_size,
        stride=sw_config.stride,
        threshold_method=sw_config.threshold_method,
        threshold_value=sw_config.threshold_value,
        smoothing_passes=sw_config.smoothing_passes,
    )

    result = segmenter.segment_from_chunk_data(embeddings_data.words, chunk_data)
    print(f"  Detected {len(result.segments)} segments")

    # 6. Map segments to timestamps and build output
    segment_results: list[SegmentData] = []
    for i, seg in enumerate(result.segments):
        start_time = word_position_to_timestamp(seg.start_word, srt_entries)
        end_pos = seg.end_word - 1 if seg.end_word > 0 else 0
        end_time = word_position_to_timestamp(end_pos, srt_entries)

        segment_results.append(
            SegmentData(
                segment_id=i + 1,
                start_timestamp=format_timedelta(start_time),
                end_timestamp=format_timedelta(end_time),
                text=seg.text,
                start_word=seg.start_word,
                end_word=seg.end_word,
            )
        )

    config_data = SegmentationConfigData(
        embedding_model=embeddings_data.config.embedding_model,
        window_size=sw_config.window_size,
        stride=sw_config.stride,
        threshold_method=sw_config.threshold_method,
        threshold_value=sw_config.threshold_value,
        smoothing_passes=sw_config.smoothing_passes,
    )

    # Compute relative path for embeddings file
    relative_embeddings_path = embeddings_path.relative_to(data_dir)

    return SegmentationOutput(
        source_file=embeddings_data.source_file,  # Already relative from generate-embeddings
        embeddings_file=str(relative_embeddings_path),
        segmented_at=datetime.now().isoformat(),
        config=config_data,
        total_segments=len(segment_results),
        segments=segment_results,
    )


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Detect topic boundaries from embeddings")
    parser.add_argument(
        "--file",
        type=Path,
        help="Process a single _embeddings.json file instead of all files",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config(config_path)
    td_config = config.get_topic_detection_config()

    # Data and output directories
    data_dir = config.getDataDir()
    output_dir = data_dir / td_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine files to process
    if args.file:
        # Single file mode
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
        if not args.file.name.endswith("_embeddings.json"):
            print(f"Error: Expected _embeddings.json file, got: {args.file}", file=sys.stderr)
            return 1
        embeddings_files = [args.file]
        base_dir = args.file.parent
    else:
        # Process all _embeddings.json files in output directory
        if not output_dir.exists():
            print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
            return 1

        embeddings_files = sorted(output_dir.rglob("*_embeddings.json"))
        embeddings_files = [f for f in embeddings_files if not f.name.startswith("._")]
        base_dir = output_dir

    if not embeddings_files:
        print("No _embeddings.json files found to process.")
        return 0

    print(f"Found {len(embeddings_files)} _embeddings.json file(s) to process")
    print(f"Output directory: {output_dir}")
    print()

    success_count = 0
    failure_count = 0
    skipped_count = 0

    for embeddings_file in embeddings_files:
        relative_path = embeddings_file.relative_to(base_dir)

        # Check if segmentation file already exists
        output_subdir = output_dir / relative_path.parent
        output_filename = relative_path.name.replace("_embeddings.json", "_segmentation.json")
        output_path = output_subdir / output_filename

        if output_path.exists():
            print(f"Skipping (already exists): {relative_path}")
            skipped_count += 1
            continue

        print(f"Processing: {relative_path}")

        try:
            segmentation_result = process_embeddings(embeddings_file, data_dir, config)

            # Write segmentation JSON output
            # Replace _embeddings.json with _segmentation.json
            output_subdir = output_dir / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_filename = relative_path.name.replace("_embeddings.json", "_segmentation.json")
            output_path = output_subdir / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(segmentation_result.model_dump(), f, indent=2, ensure_ascii=False)

            print(f"  → {segmentation_result.total_segments} segments → {output_path}")
            success_count += 1
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            failure_count += 1
        print()

    # Summary
    print("=" * 50)
    print(f"Completed: {success_count} succeeded, {skipped_count} skipped, {failure_count} failed")

    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
