#!/usr/bin/env python3
"""Print configured data files that have no YouTube ID and no ID-bearing sibling."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config  # noqa: E402

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


def iter_orphan_files(base_dir: Path) -> Iterable[Path]:
    """Yield files in base_dir that lack a YouTube ID and have no same-folder ID-bearing sibling."""
    if not base_dir.is_dir():
        return

    files_by_parent: dict[Path, list[Path]] = defaultdict(list)
    for file_path in base_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name in IGNORED_FILENAMES:
            continue
        if any(part.startswith("._") for part in file_path.relative_to(base_dir).parts):
            continue
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


def path_relative_to_data_dir(path: Path, data_dir: Path) -> Path:
    """Return path relative to the configured data directory."""
    return path.resolve().relative_to(data_dir.resolve())


def print_scan(config: Config) -> int:
    """Print one section per ID-convention data directory and return the total orphan count."""
    total_count = 0
    data_dir = config.get_data_dir()
    for config_key, base_dir in iter_id_convention_data_dirs(config):
        print(f"=== Scanning {config_key}: {base_dir} ===")
        folder_count = 0
        for orphan in iter_orphan_files(base_dir):
            print(path_relative_to_data_dir(orphan, data_dir))
            folder_count += 1
        if folder_count == 0:
            print("No files without a YouTube ID found in this folder.")
        else:
            print(f"{folder_count} file(s) without a YouTube ID.")
        total_count += folder_count
        print()
    return total_count


def main() -> int:
    """Load config and print data files that have no YouTube ID and no ID-bearing sibling."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    if len(sys.argv) != 1:
        print(f"Usage: uv run python {Path(__file__)}", file=sys.stderr)
        return 2

    try:
        config = Config(config_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    total = print_scan(config)
    if total == 0:
        print("No files without a YouTube ID found.")
    else:
        print(f"Found {total} file(s) without a YouTube ID across all scanned folders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
