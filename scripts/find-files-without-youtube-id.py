#!/usr/bin/env python3
"""Report configured data files that have no YouTube ID and no ID-bearing sibling."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import Config

YOUTUBE_ID_BEFORE_EXTENSION_RE = re.compile(r"\s*\[([A-Za-z0-9_-]+)\](?=\.[^/]+$)")

ID_CONVENTION_DIR_KEYS = (
    "data_downloads_videos_dir",
    "data_downloads_audio_dir",
    "data_downloads_transcripts_dir",
    "data_downloads_transcripts_hallucinations_dir",
    "data_downloads_transcripts_cleaned_dir",
    "data_downloads_transcripts_summaries_dir",
    "data_downloads_metadata_dir",
    "data_archive_videos_dir",
)

# Per-channel bookkeeping files that are ID-less by design and must never be removed,
# so they are not meaningful "missing a YouTube ID" findings (e.g. the yt-dlp archive).
IGNORED_FILENAMES = frozenset({"downloaded.txt"})

# File extensions that never represent downloadable video content (e.g. helper shell scripts).
IGNORED_SUFFIXES = frozenset({".sh"})

REPORT_FILENAME = "files-without-youtube-id.md"


@dataclass(frozen=True)
class FolderResult:
    """Orphan files found in one configured data folder."""

    config_key: str
    relative_dir: Path
    orphans: list[Path]


def extract_youtube_id(filename: str) -> str | None:
    """Return the bracketed YouTube ID immediately before the extension."""
    match = YOUTUBE_ID_BEFORE_EXTENSION_RE.search(filename)
    return match.group(1) if match is not None else None


def filename_without_youtube_id(filename: str) -> str | None:
    """Return the sibling filename after removing the bracketed ID token."""
    if YOUTUBE_ID_BEFORE_EXTENSION_RE.search(filename) is None:
        return None
    return YOUTUBE_ID_BEFORE_EXTENSION_RE.sub("", filename, count=1)


def iter_id_convention_data_dirs(config: Config) -> Iterable[tuple[str, Path]]:
    """Yield (config_key, path) for each configured ID-convention data directory."""
    raw_paths: dict[str, Any] = config.get_paths_config().model_dump()
    for config_key in ID_CONVENTION_DIR_KEYS:
        raw_value = raw_paths.get(config_key)
        if raw_value is None:
            continue
        yield config_key, Path(str(raw_value)).expanduser()


def _is_scannable_file(file_path: Path, base_dir: Path) -> bool:
    """Return whether a file should be considered when scanning for orphans."""
    if not file_path.is_file():
        return False
    if file_path.name in IGNORED_FILENAMES:
        return False
    if file_path.suffix in IGNORED_SUFFIXES:
        return False
    return not any(part.startswith(".") for part in file_path.relative_to(base_dir).parts)


def iter_orphan_files(base_dir: Path) -> Iterable[Path]:
    """Yield files in base_dir that lack a YouTube ID and have no same-folder ID-bearing sibling."""
    if not base_dir.is_dir():
        return

    files_by_parent: dict[Path, list[Path]] = defaultdict(list)
    for file_path in base_dir.rglob("*"):
        if _is_scannable_file(file_path, base_dir):
            files_by_parent[file_path.parent].append(file_path)

    for parent_dir in sorted(files_by_parent):
        files = files_by_parent[parent_dir]
        covered_plain_names = {
            filename_without_youtube_id(file_path.name) for file_path in files if extract_youtube_id(file_path.name) is not None
        }
        for file_path in sorted(files):
            if extract_youtube_id(file_path.name) is not None:
                continue
            if file_path.name in covered_plain_names:
                continue
            yield file_path


def collect_results(config: Config) -> list[FolderResult]:
    """Scan each ID-convention folder and collect its orphan files, relative to that folder."""
    data_dir = config.get_data_dir()
    results: list[FolderResult] = []
    for config_key, base_dir in iter_id_convention_data_dirs(config):
        relative_dir = base_dir.resolve().relative_to(data_dir.resolve())
        orphans = sorted(orphan.relative_to(base_dir) for orphan in iter_orphan_files(base_dir))
        results.append(FolderResult(config_key=config_key, relative_dir=relative_dir, orphans=orphans))
    return results


def _box_rule(left: str, middle: str, right: str, category_width: int, orphans_width: int) -> str:
    """Return a horizontal box-drawing rule spanning both padded columns."""
    return f"{left}{'─' * (category_width + 2)}{middle}{'─' * (orphans_width + 2)}{right}"


def _box_row(category: str, orphans: str, category_width: int, orphans_width: int) -> str:
    """Return a box-drawing table row with a left-aligned and a right-aligned cell."""
    return f"│ {category.ljust(category_width)} │ {orphans.rjust(orphans_width)} │"


def render_box_table(rows: list[tuple[str, int]], total: int) -> str:
    """Render category/orphan-count rows as an aligned box-drawing table with a TOTAL row."""
    header_category = "Category"
    header_orphans = "Orphans"
    total_label = "TOTAL"
    category_width = max([len(header_category), len(total_label)] + [len(category) for category, _ in rows])
    count_strings = [str(count) for _, count in rows] + [str(total)]
    orphans_width = max([len(header_orphans)] + [len(value) for value in count_strings])

    lines = [
        _box_rule("┌", "┬", "┐", category_width, orphans_width),
        _box_row(header_category, header_orphans, category_width, orphans_width),
        _box_rule("├", "┼", "┤", category_width, orphans_width),
    ]
    for category, count in rows:
        lines.append(_box_row(category, str(count), category_width, orphans_width))
    lines.append(_box_rule("├", "┼", "┤", category_width, orphans_width))
    lines.append(_box_row(total_label, str(total), category_width, orphans_width))
    lines.append(_box_rule("└", "┴", "┘", category_width, orphans_width))
    return "\n".join(lines)


def summary_table(results: list[FolderResult]) -> str:
    """Render the per-folder orphan-count summary as a box-drawing table."""
    rows = [(str(result.relative_dir), len(result.orphans)) for result in results]
    total = sum(len(result.orphans) for result in results)
    return render_box_table(rows, total)


def format_channel_video_line(channel: str, title: str) -> str:
    """Format a deduplicated channel/video pair as a single report line."""
    if not channel:
        return title
    return f"{channel} — {title}"


def deduplicated_channel_video_pairs(results: list[FolderResult]) -> list[tuple[str, str]]:
    """Collapse every orphan path to a (channel, video title) pair, deduplicated across all folders.

    The channel is the first path component under the data folder; the video title is the file
    name without its final extension. The same video appearing as several files (e.g. .srt, .vtt,
    .md across folders) therefore collapses to a single entry.
    """
    pairs: set[tuple[str, str]] = set()
    for result in results:
        for orphan in result.orphans:
            channel = orphan.parts[0] if len(orphan.parts) > 1 else ""
            pairs.add((channel, orphan.stem))
    return sorted(pairs)


def render_report(results: list[FolderResult]) -> str:
    """Render the report: summary table, per-folder file listings, and a deduplicated channel/video list."""
    parts = ["```", summary_table(results), "```", ""]
    for result in results:
        if not result.orphans:
            continue
        parts.append(f"# {result.relative_dir}")
        parts.append("")
        parts.extend(str(orphan) for orphan in result.orphans)
        parts.append("")
    parts.append("# Channel/Video list")
    parts.append("")
    parts.extend(format_channel_video_line(channel, title) for channel, title in deduplicated_channel_video_pairs(results))
    return "\n".join(parts).rstrip() + "\n"


def write_report(config: Config, content: str) -> Path:
    """Write the report under the configured reports directory and return its path."""
    report_path = config.get_reports_dir() / REPORT_FILENAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return report_path


def main() -> int:
    """Load config, scan ID-convention folders, write a report, and print a summary."""
    config_path = Config.repo_config_path()
    if len(sys.argv) != 1:
        print(f"Usage: uv run python {Path(__file__)}", file=sys.stderr)
        return 2

    try:
        config = Config(config_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    results = collect_results(config)
    report_path = write_report(config, render_report(results))

    print(summary_table(results))
    print()
    total = sum(len(result.orphans) for result in results)
    unique_pairs = len(deduplicated_channel_video_pairs(results))
    print(f"Found {total} file(s) without a YouTube ID across all scanned folders.")
    print(f"Deduplicated to {unique_pairs} unique channel/video pair(s).")
    print(f"Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
