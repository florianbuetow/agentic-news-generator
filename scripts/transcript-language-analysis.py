#!/usr/bin/env python3
"""Analyze language distribution across transcript files.

Iterates over all .txt transcript files, performs language detection on
~1000 word chunks, and generates per-channel markdown reports.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config import Config
from src.nlp.language_detector import LanguageDetector
from src.util.fs_util import FSUtil


@dataclass
class FileLanguageResult:
    """Language analysis result for a single file."""

    filename: str
    languages: list[str]  # One language code per chunk
    word_count: int  # Total words in file

    @property
    def language_summary(self) -> str:
        """Get comma-separated unique language codes (deduplicated).

        Shows unique languages sorted alphabetically.
        Example: "de,en" for a file with German and English chunks.
        """
        if not self.languages:
            return "no_content"
        unique_langs = sorted(set(self.languages))  # Sort for consistency
        return ",".join(unique_langs)

    @property
    def is_english_only(self) -> bool:
        """Check if file contains only English.

        Returns:
            True if all chunks are English or file is empty.
        """
        unique_langs = set(self.languages)
        return unique_langs == {"en"} or unique_langs == set()


@dataclass
class ChannelLanguageReport:
    """Language analysis report for a channel."""

    channel_name: str
    transcripts_path: Path
    file_results: list[FileLanguageResult]


def chunk_text(text: str, target_words: int = 1000) -> list[str]:
    """Split text into chunks ensuring each has ~target_words.

    The last chunk is expanded to include words from the previous chunk
    if it's smaller than target_words, unless total text < target_words.

    Args:
        text: Input text to chunk.
        target_words: Target number of words per chunk.

    Returns:
        List of text chunks, empty list if no content.
    """
    words = text.split()
    if not words:
        return []

    # If total text < target_words, return as single chunk
    if len(words) < target_words:
        return [text.strip()]

    chunks = []
    for i in range(0, len(words), target_words):
        chunk = " ".join(words[i : i + target_words])
        if chunk.strip():
            chunks.append(chunk)

    # Ensure last chunk has target_words (expand from previous if needed)
    if len(chunks) > 1:
        last_chunk_words = chunks[-1].split()
        if len(last_chunk_words) < target_words:
            # Remove last chunk and expand it
            chunks.pop()
            # Calculate how many words to take from end
            words_needed = target_words
            expanded_chunk = " ".join(words[-words_needed:])
            chunks.append(expanded_chunk)

    return chunks


def analyze_file(txt_file: Path, detector: LanguageDetector, target_words: int = 1000) -> FileLanguageResult:
    """Analyze language distribution in a single transcript file.

    Args:
        txt_file: Path to .txt transcript file.
        detector: Initialized LanguageDetector instance.
        target_words: Number of words per chunk for analysis.

    Returns:
        FileLanguageResult with detected languages per chunk and word count.
    """
    try:
        content = FSUtil.read_text_file(txt_file)
    except Exception as e:
        print(f"  ⚠️  Warning: Could not read {txt_file.name}: {e}")
        return FileLanguageResult(filename=txt_file.name, languages=[], word_count=0)

    # Count total words
    total_words = len(content.split())

    chunks = chunk_text(content, target_words=target_words)

    if not chunks:
        return FileLanguageResult(filename=txt_file.name, languages=[], word_count=0)

    # Detect language for each chunk
    languages = []
    for chunk in chunks:
        lang_code = detector.detect_language(chunk)
        if lang_code == "??":
            print(f"  ⚠️  Warning: No alphabetic words found in chunk from {txt_file.name}")
        languages.append(lang_code if lang_code else "unknown")

    return FileLanguageResult(filename=txt_file.name, languages=languages, word_count=total_words)


def generate_channel_report(
    report: ChannelLanguageReport,
    output_dir: Path,
    data_dir: Path,
) -> Path:
    """Generate markdown report for a channel.

    Args:
        report: ChannelLanguageReport with analysis results.
        output_dir: Directory to save report in.
        data_dir: Root data directory for computing relative paths.

    Returns:
        Path to generated report file.
    """
    # Create output directory if needed
    FSUtil.ensure_directory_exists(output_dir)

    # Generate report filename
    report_file = output_dir / f"{report.channel_name}_language_analysis.md"

    # Compute relative path for display
    try:
        relative_path = report.transcripts_path.relative_to(data_dir)
    except ValueError:
        relative_path = report.transcripts_path

    # Generate timestamp
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build markdown content
    lines = [
        f"# Language Analysis: {report.channel_name}",
        "",
        f"**Transcripts Path:** `{relative_path}`",
        "",
        f"**Analysis Date:** {timestamp}",
        "",
        "## Files by Language",
        "",
    ]

    # Sort files alphabetically
    sorted_results = sorted(report.file_results, key=lambda r: r.filename)

    for result in sorted_results:
        lang_summary = result.language_summary
        lines.append(f"- `[{lang_summary}]` `{result.word_count} words` {result.filename}")

    # Add Non-English transcriptions section
    non_english_files = [r for r in sorted_results if not r.is_english_only]
    if non_english_files:
        lines.extend(
            [
                "",
                "## Non-English Transcriptions",
                "",
            ]
        )
        for result in non_english_files:
            lang_summary = result.language_summary
            lines.append(f"- `[{lang_summary}]` `{result.word_count} words` {result.filename}")

    lines.extend(
        [
            "",
            "---",
            "*Generated by transcript-language-analysis.py*",
            "",
        ]
    )

    # Write report
    content = "\n".join(lines)
    FSUtil.write_text_file(report_file, content, create_parents=True)

    return report_file


def report_non_english_files(non_english_files: list[tuple[str, Path, FileLanguageResult]]) -> None:
    """Report detected non-English files to stdout.

    Args:
        non_english_files: List of (channel_name, file_path, result) tuples for non-English files.
    """
    print()
    print("⚠️  Non-English transcriptions detected:")
    print()
    for channel_name, file_path, result in non_english_files:
        unique_langs = result.language_summary
        print(f"  [{channel_name}] `[{unique_langs}]` {file_path}")
    print()
    print(f"Total non-English files: {len(non_english_files)}")


def main() -> int:
    """Main entry point for transcript language analysis.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config(config_path)

    # Get paths
    transcripts_dir = config.getDataDownloadsTranscriptsDir()
    output_dir = config.getDataOutputDir() / "language_analysis"
    data_dir = config.getDataDir()

    # Initialize language detector
    model_path = config.getDataModelsDir() / "fasttext" / "lid.176.ftz"
    try:
        detector = LanguageDetector(model_path=model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Check if transcripts directory exists
    if not transcripts_dir.exists():
        print(f"Error: Transcripts directory not found: {transcripts_dir}")
        print("Ensure transcripts have been downloaded.")
        return 1

    # Discover channels
    channel_dirs = FSUtil.find_directories(transcripts_dir, exclude_hidden=True)
    channel_dirs = [d for d in channel_dirs if not d.name.startswith(".")]
    channel_dirs = sorted(channel_dirs, key=lambda d: d.name)

    if not channel_dirs:
        print(f"No channel directories found in {transcripts_dir}")
        return 1

    print("Analyzing language distribution in transcripts...")
    print()

    total_files = 0
    total_channels = len(channel_dirs)
    non_english_files_global: list[tuple[str, Path, FileLanguageResult]] = []  # (channel, file_path, result)

    # Process each channel
    for channel_dir in channel_dirs:
        channel_name = channel_dir.name
        print(f"Processing channel: {channel_name}")

        # Find all .txt files
        txt_files = FSUtil.find_files_by_extension(channel_dir, ".txt", recursive=True)
        txt_files = [f for f in txt_files if not f.name.startswith("._")]

        if not txt_files:
            print("  No .txt files found")
            print()
            continue

        # Analyze each file
        file_results = []
        for txt_file in txt_files:
            result = analyze_file(txt_file, detector)
            file_results.append(result)

            # Track non-English files
            if not result.is_english_only:
                non_english_files_global.append((channel_name, txt_file, result))

            # Display progress
            lang_display = f"[{result.language_summary}]"
            chunk_count = len(result.languages)
            print(f"  - {txt_file.name}: {chunk_count} chunks analyzed {lang_display}")

        total_files += len(txt_files)

        # Generate report
        report = ChannelLanguageReport(
            channel_name=channel_name,
            transcripts_path=channel_dir,
            file_results=file_results,
        )

        report_file = generate_channel_report(report, output_dir, data_dir)
        try:
            display_path = report_file.relative_to(Path.cwd())
        except ValueError:
            display_path = report_file
        print(f"✅ Report saved: {display_path}")
        print()

    print(f"✅ Analysis complete: {total_files} files analyzed across {total_channels} channels")

    # Report non-English files and exit with error if any found
    if non_english_files_global:
        report_non_english_files(non_english_files_global)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
