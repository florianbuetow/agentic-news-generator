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


def process_srt(srt_path: Path, config: Config) -> tuple[EmbeddingsOutput, bool]:
    """Process a single SRT file and generate embeddings.

    Args:
        srt_path: Path to the SRT file.
        config: Configuration object.

    Returns:
        Tuple of (EmbeddingsOutput, is_short) where is_short indicates
        the text was shorter than window_size.
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
        smoothing_passes=sw_config.smoothing_passes,
    )

    # 4. Generate embeddings
    print("  Generating embeddings...")
    words, chunk_data, is_short = segmenter.generate_embeddings(text)
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

    output = EmbeddingsOutput(
        source_file=str(srt_path),
        generated_at=datetime.now().isoformat(),
        config=config_data,
        total_words=len(words),
        words=words,
        srt_entries=srt_entries_data,
        windows=windows_data,
    )
    return output, is_short


def get_srt_files(args: argparse.Namespace, config: Config) -> tuple[list[Path], Path] | int:
    """Determine which SRT files to process.

    Args:
        args: Parsed command line arguments.
        config: Configuration object.

    Returns:
        Tuple of (srt_files, base_dir) or an exit code if error.
    """
    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
        if args.file.suffix.lower() != ".srt":
            print(f"Error: Expected .srt file, got: {args.file}", file=sys.stderr)
            return 1
        return [args.file], args.file.parent

    cleaned_dir = config.getDataDownloadsTranscriptsCleanedDir()
    if not cleaned_dir.exists():
        print(f"Error: Cleaned transcripts directory not found: {cleaned_dir}", file=sys.stderr)
        return 1

    srt_files = sorted(cleaned_dir.rglob("*.srt"))
    srt_files = [f for f in srt_files if not f.name.startswith("._")]
    return srt_files, cleaned_dir


def process_single_file(
    srt_file: Path,
    base_dir: Path,
    output_dir: Path,
    config: Config,
) -> tuple[str, str | None]:
    """Process a single SRT file.

    Args:
        srt_file: Path to the SRT file.
        base_dir: Base directory for relative paths.
        output_dir: Output directory for embeddings.
        config: Configuration object.

    Returns:
        Tuple of (status, relative_path) where status is "success", "skipped",
        "warning", or "empty".
    """
    relative_path = srt_file.relative_to(base_dir)
    output_subdir = output_dir / relative_path.parent
    output_filename = relative_path.stem + "_embeddings.json"
    output_path = output_subdir / output_filename

    if output_path.exists():
        return "skipped", None

    print(f"Processing: {relative_path}")

    try:
        embeddings_result, is_short = process_srt(srt_file, config)
        output_subdir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embeddings_result.model_dump(), f, indent=2, ensure_ascii=False)

        if is_short:
            print("  ⚠ WARNING: Text too short, created single embedding")
            print(f"  → {len(embeddings_result.windows)} windows → {output_path}")
            print()
            return "warning", None

        print(f"  → {len(embeddings_result.windows)} windows → {output_path}")
        print()
        return "success", None
    except ValueError as e:
        print(f"  ⚠ SKIPPED: {e}")
        print()
        return "empty", str(relative_path)


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

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config(config_path)
    td_config = config.get_topic_detection_config()

    output_dir = config.getDataDir() / td_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result = get_srt_files(args, config)
    if isinstance(result, int):
        return result
    srt_files, base_dir = result

    if not srt_files:
        print("No SRT files found to process.")
        return 0

    print(f"Found {len(srt_files)} SRT file(s) to check")
    print(f"Output directory: {output_dir}")
    print()

    counts = {"success": 0, "skipped": 0, "warning": 0}
    empty_files: list[str] = []

    for srt_file in srt_files:
        status, path = process_single_file(srt_file, base_dir, output_dir, config)
        if status == "empty" and path:
            empty_files.append(path)
        else:
            counts[status] += 1

    print("=" * 50)
    summary = f"Completed: {counts['success']} new, {counts['skipped']} skipped"
    if counts["warning"] > 0:
        summary += f", {counts['warning']} warnings"
    if empty_files:
        summary += f", {len(empty_files)} empty/skipped"
    print(summary)

    if empty_files:
        print()
        print("Empty/whitespace-only files skipped:")
        for path in empty_files:
            print(f"  - {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
