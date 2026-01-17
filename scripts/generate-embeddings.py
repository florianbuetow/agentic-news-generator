#!/usr/bin/env python3
"""Generate embeddings from SRT transcripts.

Step 1 of the three-step topic detection pipeline.
Reads SRT files, tokenizes the text, generates embeddings using the configured
embedding model, and outputs _embeddings.json files.

Usage:
    uv run python scripts/generate-embeddings.py
    uv run python scripts/generate-embeddings.py --file path/to/transcript.srt
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.config import Config
from src.topic_detection.embedding.factory import EmbeddingGeneratorFactory
from src.topic_detection.segmentation.schemas import (
    EmbeddingConfigData,
    EmbeddingsOutput,
    SRTEntryData,
    WindowData,
)
from src.topic_detection.segmentation.segmenter import SlidingWindowTopicSegmenter
from src.util.srt_util import SRTEntry, SRTUtil


def format_timedelta(td: timedelta) -> str:
    """Format a timedelta as SRT timestamp string (HH:MM:SS,mmm)."""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def srt_entry_to_data(entry: SRTEntry) -> SRTEntryData:
    """Convert SRTEntry to SRTEntryData for JSON serialization."""
    return SRTEntryData(
        index=entry.index,
        start_timestamp=format_timedelta(entry.start),
        end_timestamp=format_timedelta(entry.end),
        content=entry.content,
        word_start=entry.word_start,
        word_end=entry.word_end,
    )


def process_srt(srt_path: Path, config: Config) -> EmbeddingsOutput:
    """Process a single SRT file and generate embeddings.

    Args:
        srt_path: Path to the SRT file.
        config: Configuration object.

    Returns:
        EmbeddingsOutput with tokens, SRT entries, and embeddings.
    """
    td_config = config.get_topic_detection_config()
    sw_config = td_config.sliding_window

    # 1. Parse SRT
    print("  Parsing SRT file...")
    entries = SRTUtil.parse_srt_file(srt_path)
    text = SRTUtil.entries_to_text(entries)
    total_words = SRTUtil.get_total_words(entries)
    print(f"  Found {len(entries)} SRT entries, {total_words} total words")

    # 2. Create embedding generator
    print("  Creating embedding generator...")
    embedding_generator = EmbeddingGeneratorFactory.create(td_config.embedding)

    # 3. Create segmenter for embedding generation only
    segmenter = SlidingWindowTopicSegmenter(
        embedding_generator=embedding_generator,
        window_size=sw_config.window_size,
        stride=sw_config.stride,
        threshold_method=sw_config.threshold_method,
        threshold_value=sw_config.threshold_value,
        min_segment_tokens=sw_config.min_segment_tokens,
        smoothing_passes=sw_config.smoothing_passes,
    )

    # 4. Generate embeddings
    print("  Generating embeddings...")
    tokens, chunk_data = segmenter.generate_embeddings(text)
    print(f"  Generated {len(chunk_data.chunk_positions)} windows")

    # 5. Build output structure
    srt_entries_data = [srt_entry_to_data(e) for e in entries]

    windows_data = [
        WindowData(
            offset=offset,
            embedding=embedding.tolist(),
        )
        for offset, embedding in zip(chunk_data.chunk_positions, chunk_data.embeddings, strict=True)
    ]

    config_data = EmbeddingConfigData(
        embedding_model=td_config.embedding.model_name,
        window_size=sw_config.window_size,
        stride=sw_config.stride,
    )

    return EmbeddingsOutput(
        source_file=str(srt_path),
        generated_at=datetime.now().isoformat(),
        config=config_data,
        total_tokens=len(tokens),
        tokens=tokens,
        srt_entries=srt_entries_data,
        windows=windows_data,
    )


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Generate embeddings from SRT transcripts")
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
        if args.file.suffix.lower() != ".srt":
            print(f"Error: Expected .srt file, got: {args.file}", file=sys.stderr)
            return 1
        srt_files = [args.file]
        base_dir = args.file.parent
    else:
        # Process all cleaned SRT files
        cleaned_dir = config.getDataDownloadsTranscriptsCleanedDir()
        if not cleaned_dir.exists():
            print(f"Error: Cleaned transcripts directory not found: {cleaned_dir}", file=sys.stderr)
            return 1

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
    failure_count = 0

    for srt_file in srt_files:
        relative_path = srt_file.relative_to(base_dir)
        print(f"Processing: {relative_path}")

        try:
            embeddings_result = process_srt(srt_file, config)

            # Determine output path: channel/videofilename_embeddings.json
            output_subdir = output_dir / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_filename = relative_path.stem + "_embeddings.json"
            output_path = output_subdir / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings_result.model_dump(), f, indent=2, ensure_ascii=False)

            print(f"  → {len(embeddings_result.windows)} windows → {output_path}")
            success_count += 1
        except ValueError as e:
            print(f"  Error: {e}", file=sys.stderr)
            failure_count += 1
        print()

    # Summary
    print("=" * 50)
    print(f"Completed: {success_count} succeeded, {failure_count} failed")

    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
