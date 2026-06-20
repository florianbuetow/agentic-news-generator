#!/usr/bin/env python3
"""Print configured data files that have bracketed-ID and non-ID siblings."""

from __future__ import annotations

import re
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import Config

YOUTUBE_ID_BEFORE_EXTENSION_RE = re.compile(r"\s*\[([A-Za-z0-9_-]+)\](?=\.[^/]+$)")


@dataclass(frozen=True)
class MatchedPair:
    """A file with a bracketed ID and its same-folder non-ID sibling."""

    config_key: str
    data_dir: Path
    base_dir: Path
    channel_name: str
    video_id: str
    file_with_id: Path
    file_without_id: Path


def extract_youtube_id(filename: str) -> str | None:
    """Return the bracketed YouTube ID immediately before the extension."""
    match = YOUTUBE_ID_BEFORE_EXTENSION_RE.search(filename)
    return match.group(1) if match is not None else None


def filename_without_youtube_id(filename: str) -> str | None:
    """Return the sibling filename after removing the bracketed ID token."""
    if YOUTUBE_ID_BEFORE_EXTENSION_RE.search(filename) is None:
        return None
    return YOUTUBE_ID_BEFORE_EXTENSION_RE.sub("", filename, count=1)


def iter_leaf_data_dirs(config: Config) -> Iterable[tuple[str, Path]]:
    """Yield configured data directories that are not parents of other configured data directories."""
    raw_paths: dict[str, Any] = config.get_paths_config().model_dump()
    data_dirs = {key: Path(str(value)).expanduser() for key, value in raw_paths.items() if key.startswith("data_") and key.endswith("_dir")}
    resolved_dirs = {key: path.resolve() for key, path in data_dirs.items()}

    for key, path in sorted(data_dirs.items()):
        resolved_path = resolved_dirs[key]
        is_parent_of_another_configured_dir = any(
            other_key != key and other_path.is_relative_to(resolved_path) for other_key, other_path in resolved_dirs.items()
        )
        if not is_parent_of_another_configured_dir:
            yield key, path


def iter_channel_dirs(base_dir: Path) -> Iterable[Path]:
    """Yield channel directories directly inside one configured data directory."""
    if not base_dir.is_dir():
        return

    for channel_dir in sorted(base_dir.iterdir(), key=lambda path: path.name):
        if channel_dir.is_dir() and not channel_dir.name.startswith("._"):
            yield channel_dir


def iter_youtube_id_file_pairs(config_key: str, data_dir: Path, base_dir: Path, channel_dir: Path) -> Iterable[MatchedPair]:
    """Yield files whose same folder contains the matching non-ID filename."""
    for file_path in sorted(channel_dir.rglob("*"), key=lambda path: str(path)):
        if not file_path.is_file() or file_path.name.startswith("._"):
            continue

        video_id = extract_youtube_id(file_path.name)
        if video_id is None:
            continue

        sibling_name = filename_without_youtube_id(file_path.name)
        if sibling_name is None:
            continue

        sibling_path = file_path.with_name(sibling_name)
        if not sibling_path.is_file():
            continue

        yield MatchedPair(
            config_key=config_key,
            data_dir=data_dir,
            base_dir=base_dir,
            channel_name=channel_dir.name,
            video_id=video_id,
            file_with_id=file_path,
            file_without_id=sibling_path,
        )


def path_relative_to_data_dir(path: Path, data_dir: Path) -> Path:
    """Return path relative to the configured data directory."""
    return path.resolve().relative_to(data_dir.resolve())


def format_file_lines(match: MatchedPair) -> tuple[str, str]:
    """Format a matched pair as two descriptive single-file records."""
    file_with_id = path_relative_to_data_dir(match.file_with_id, match.data_dir)
    file_without_id = path_relative_to_data_dir(match.file_without_id, match.data_dir)
    return (
        f"[{match.video_id}] id_filename     {file_with_id}",
        f"[{match.video_id}] plain_filename  {file_without_id}",
    )


def print_matches(matches: Iterable[MatchedPair]) -> int:
    """Print one descriptive line per file and return the number of pairs printed."""
    count = 0
    for match in matches:
        for line in format_file_lines(match):
            print(line)
        count += 1
    return count


def iter_all_matches(config: Config) -> Iterable[MatchedPair]:
    """Scan configured data directories, then channel directories, then matching files."""
    data_dir = config.get_data_dir()
    for config_key, base_dir in iter_leaf_data_dirs(config):
        for channel_dir in iter_channel_dirs(base_dir):
            yield from iter_youtube_id_file_pairs(config_key, data_dir, base_dir, channel_dir)


def copy_plain_files(config: Config, backup_data_dir: Path) -> int:
    """Copy all plain filenames to backup_data_dir, preserving paths relative to data_dir."""
    copied_count = 0
    failed_count = 0
    backup_data_dir = backup_data_dir.expanduser()
    for match in iter_all_matches(config):
        backup_path = backup_data_dir / path_relative_to_data_dir(match.file_without_id, match.data_dir)
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(match.file_without_id, backup_path)
            if not backup_path.is_file():
                print(f"[{match.video_id}] copy verification failed  {backup_path}", file=sys.stderr)
                failed_count += 1
                continue
            match.file_without_id.unlink()
        except OSError as exc:
            print(f"[{match.video_id}] move failed  {match.file_without_id}: {exc}", file=sys.stderr)
            failed_count += 1
            continue
        source_label = path_relative_to_data_dir(match.file_without_id, match.data_dir)
        print(f"[{match.video_id}] moved_plain_filename  {source_label} -> {backup_path}")
        copied_count += 1

    if copied_count == 0:
        print("No plain filenames found to move.")
    else:
        print(f"Moved {copied_count} plain filename file(s).")
    if failed_count > 0:
        print(f"Failed to move {failed_count} plain filename file(s).", file=sys.stderr)
    return 1 if failed_count > 0 else 0


def get_backup_data_dir(config: Config) -> Path:
    """Return the backup data directory next to the configured data directory."""
    return config.get_data_dir().parent / "backup" / "data"


def print_scan(config: Config) -> int:
    """Print scan headers per configured data directory and one line per matched pair."""
    total_count = 0
    data_dir = config.get_data_dir()
    for config_key, base_dir in iter_leaf_data_dirs(config):
        print(f"=== Scanning {config_key}: {base_dir} ===")
        folder_count = 0
        for channel_dir in iter_channel_dirs(base_dir):
            folder_count += print_matches(iter_youtube_id_file_pairs(config_key, data_dir, base_dir, channel_dir))
        if folder_count == 0:
            print("No matching file pairs found in this folder.")
        total_count += folder_count
        print()
    return total_count


def main() -> int:
    """Load config and move plain filename files to the configured backup directory."""
    config_path = Config.repo_config_path()
    if len(sys.argv) != 1:
        print(f"Usage: uv run python {Path(__file__)}", file=sys.stderr)
        return 2

    try:
        config = Config(config_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    return copy_plain_files(config, get_backup_data_dir(config))


if __name__ == "__main__":
    sys.exit(main())
