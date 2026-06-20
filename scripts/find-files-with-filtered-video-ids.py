#!/usr/bin/env python3
"""Flag data files whose YouTube ID is listed in the filter file but were never removed.

Every entry in the filter file marks a video that should have been filtered out of the
pipeline. This read-only check loads the filter file via ``config.py``, builds a set of
filtered YouTube IDs, then walks each ID-convention data directory channel-by-channel.
Any file whose bracketed YouTube ID is in that set is reported as an offender, so the
leftover artifacts of a video that should have been filtered can be found and removed.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from src.config import Config

YOUTUBE_ID_BEFORE_EXTENSION_RE = re.compile(r"\[([A-Za-z0-9_-]+)\](?=\.[^/]+$)")


def load_filtered_video_ids(filter_path: Path) -> set[str]:
    """Return the set of video IDs listed across every key of the filter file.

    Entries have the form ``Channel/video_id``; the video ID portion is collected
    regardless of which directory key it sits under.
    """
    with filter_path.open(encoding="utf-8") as handle:
        filter_data = json.load(handle)

    video_ids: set[str] = set()
    for entries in filter_data.values():
        for entry in entries:
            if "/" not in entry:
                print(f"WARN: malformed filter entry '{entry}', skipping", file=sys.stderr)
                continue
            video_id = entry.split("/", 1)[1]
            if video_id:
                video_ids.add(video_id)
    return video_ids


def extract_video_id(filename: str) -> str | None:
    """Return the bracketed YouTube ID immediately before the extension, if present."""
    match = YOUTUBE_ID_BEFORE_EXTENSION_RE.search(filename)
    return match.group(1) if match is not None else None


def scan_data_dirs(config: Config) -> list[Path]:
    """Return every ID-convention data directory to scan, in pipeline order."""
    return [
        config.get_data_downloads_videos_dir(),
        config.get_data_downloads_audio_dir(),
        config.get_data_downloads_transcripts_dir(),
        config.get_data_downloads_transcripts_hallucinations_dir(),
        config.get_data_downloads_transcripts_cleaned_dir(),
        config.get_data_downloads_transcripts_summaries_dir(),
        config.get_data_downloads_metadata_dir(),
        config.get_data_archive_videos_dir(),
    ]


def iter_channel_files(base_dir: Path) -> Iterable[Path]:
    """Yield non-hidden files within each channel subdirectory of ``base_dir``."""
    for channel_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        for file_path in sorted(channel_dir.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith("."):
                yield file_path


def find_offenders(base_dir: Path, filtered_ids: set[str]) -> list[Path]:
    """Return files under ``base_dir`` whose video ID is in the filtered set."""
    offenders: list[Path] = []
    for file_path in iter_channel_files(base_dir):
        video_id = extract_video_id(file_path.name)
        if video_id is not None and video_id in filtered_ids:
            offenders.append(file_path)
    return offenders


def relative_to_data_dir(path: Path, data_dir: Path) -> Path:
    """Return ``path`` relative to ``data_dir`` for display, or ``path`` if unrelated."""
    try:
        return path.resolve().relative_to(data_dir)
    except ValueError:
        return path


def main() -> int:
    """Load config + filter set, scan data folders, and flag files that escaped filtering."""
    try:
        config = Config.load_default()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    filter_path = config.get_filefilter_path()
    if not filter_path.is_file():
        print(f"Error: filter file not found: {filter_path}", file=sys.stderr)
        return 1

    try:
        filtered_ids = load_filtered_video_ids(filter_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error reading filter file {filter_path}: {exc}", file=sys.stderr)
        return 1

    data_dir = config.get_data_dir().resolve()
    print(f"Checking {len(filtered_ids)} filtered video ID(s) against data folders...")

    offenders: list[Path] = []
    for base_dir in scan_data_dirs(config):
        if not base_dir.is_dir():
            continue
        print(f"Scanning: {relative_to_data_dir(base_dir, data_dir)}")
        for offender in find_offenders(base_dir, filtered_ids):
            print(f"ERROR: the following file should have been filtered out according to {filter_path.name}: {offender}")
            offenders.append(offender)

    print()
    if offenders:
        print(f"Found {len(offenders)} file(s) that should have been filtered out according to {filter_path.name}.")
        return 1
    print("No leftover files found for any filtered video ID.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
