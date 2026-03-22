#!/usr/bin/env python3
"""Find video IDs in downloaded.txt that have no file anywhere on disk.

Checks filenames containing [VIDEO_ID] across downloads/ and archive/.
For transcripts without IDs in their names, builds a title-to-ID lookup
from metadata .info.json files to match them.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config

ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


def collect_ids_from_find(directory: str) -> set[str]:
    result = subprocess.run(
        ["find", directory, "-type", "f"], capture_output=True, text=True
    )
    ids: set[str] = set()
    for line in result.stdout.splitlines():
        match = ID_RE.search(line)
        if match:
            ids.add(match.group(1))
    return ids


def build_title_to_id_map(metadata_dir: Path) -> dict[str, str]:
    """Build a map from normalized title -> video ID using .info.json files."""
    title_map: dict[str, str] = {}
    result = subprocess.run(
        ["find", str(metadata_dir), "-path", "*/video/*.info.json", "-type", "f"],
        capture_output=True, text=True,
    )
    for filepath in result.stdout.splitlines():
        if not filepath:
            continue
        try:
            with open(filepath) as f:
                data = json.load(f)
            vid = data.get("id")
            title = data.get("title")
            if vid and title:
                title_map[normalize_title(title)] = vid
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
    return title_map


def normalize_title(title: str) -> str:
    """Normalize a title for fuzzy matching."""
    # Remove special chars, lowercase, collapse whitespace
    title = re.sub(r"[^\w\s]", "", title.lower())
    return re.sub(r"\s+", " ", title).strip()


def collect_transcript_ids_by_title(
    transcripts_dir: Path, title_map: dict[str, str]
) -> set[str]:
    """Match transcript files without [ID] in name to IDs via title lookup."""
    ids: set[str] = set()
    result = subprocess.run(
        ["find", str(transcripts_dir), "-type", "f"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if not line:
            continue
        filename = Path(line).stem
        # Skip files that already have an ID (handled by regex scan)
        if ID_RE.search(filename):
            continue
        normalized = normalize_title(filename)
        if normalized in title_map:
            ids.add(title_map[normalized])
    return ids


def main() -> None:
    project_root = Path(__file__).parent.parent
    config = Config(project_root / "config" / "config.yaml")
    videos_dir = config.getDataDownloadsVideosDir()
    metadata_dir = config.getDataDownloadsMetadataDir()
    transcripts_dir = config.getDataDownloadsTranscriptsDir()

    # 1. Collect IDs from filenames with [VIDEO_ID] anywhere on disk
    on_disk_ids: set[str] = set()
    on_disk_ids |= collect_ids_from_find(str(config.getDataDownloadsDir()))
    on_disk_ids |= collect_ids_from_find(str(config.getDataArchiveDir()))

    # 2. Match old transcripts (no ID in filename) via title lookup
    title_map = build_title_to_id_map(metadata_dir)
    on_disk_ids |= collect_transcript_ids_by_title(transcripts_dir, title_map)

    total_orphaned = 0
    total_archived = 0

    for channel_dir in sorted(videos_dir.iterdir()):
        if not channel_dir.is_dir():
            continue
        archive_file = channel_dir / "downloaded.txt"
        if not archive_file.exists():
            continue

        channel_name = channel_dir.name
        archived_ids: set[str] = set()
        for line in archive_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                archived_ids.add(parts[1])

        orphaned = archived_ids - on_disk_ids
        total_archived += len(archived_ids)
        total_orphaned += len(orphaned)

        if orphaned:
            print(f"{channel_name}: {len(orphaned)} orphaned / {len(archived_ids)} archived")
            for vid in sorted(orphaned):
                print(f"  https://www.youtube.com/watch?v={vid}")
            print()

    if total_orphaned == 0:
        print("No discrepancies found.")

    print(f"\n{'=' * 70}")
    print(f"Total: {total_orphaned} orphaned IDs out of {total_archived} archived")

    if total_orphaned > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
