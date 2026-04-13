#!/usr/bin/env python3
"""Remove files listed in ``config/filefilter.json`` and their upstream copies.

The filter file is keyed by ``config.yaml`` path keys (currently only
``data_downloads_audio_dir``) with values of the form ``Channel/video_id``.
For each entry this script finds:

- every file matching ``*[<video_id>]*`` inside the listed dir's
  ``<channel>/`` subdir
- every file matching that pattern in all **upstream** dirs

Upstream flow: ``data_downloads_videos_dir`` → ``data_downloads_audio_dir`` →
``data_downloads_transcripts_dir``. An entry listed under ``audio_dir`` therefore
also triggers removal in ``videos_dir``; an entry under ``transcripts_dir``
triggers removal in both ``audio_dir`` and ``videos_dir``.

Matching is lexical on the yt-dlp ``[<id>]`` substring, so sibling metadata
(``.info.json``, ``.silence_map.json``) and macOS AppleDouble sidecars
(``._*``) are swept up automatically.

Default mode is **dry-run**. Pass ``--execute`` to actually unlink.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config

# Ordered upstream → downstream. A filter entry found under key K also
# removes files under every key that appears BEFORE K in this list.
_PIPELINE_ORDER: list[str] = [
    "data_downloads_videos_dir",
    "data_downloads_audio_dir",
    "data_downloads_transcripts_dir",
]

_DIR_ACCESSOR = {
    "data_downloads_videos_dir": "getDataDownloadsVideosDir",
    "data_downloads_audio_dir": "getDataDownloadsAudioDir",
    "data_downloads_transcripts_dir": "getDataDownloadsTranscriptsDir",
}


def resolve_dir(config: Config, key: str) -> Path:
    """Resolve a pipeline dir key (e.g. ``data_downloads_audio_dir``) to a Path."""
    accessor = _DIR_ACCESSOR.get(key)
    if accessor is None:
        raise KeyError(f"Unknown filter dir key: {key}")
    return getattr(config, accessor)()


def dirs_at_or_upstream_of(key: str) -> list[str]:
    """Return ``key`` plus every pipeline key upstream of it."""
    if key not in _PIPELINE_ORDER:
        raise KeyError(f"Unknown filter dir key: {key}")
    idx = _PIPELINE_ORDER.index(key)
    return _PIPELINE_ORDER[: idx + 1]


def find_matching_files(channel_dir: Path, video_id: str) -> list[Path]:
    """Return every file in ``channel_dir`` whose name contains ``[video_id]``."""
    if not channel_dir.is_dir():
        return []
    token = f"[{video_id}]"
    return sorted(p for p in channel_dir.iterdir() if p.is_file() and token in p.name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    ap.add_argument("--filter", default="config/filefilter.json", help="Path to filefilter.json")
    ap.add_argument("--execute", action="store_true", help="Actually delete files (default: dry-run)")
    return ap.parse_args()


def _resolve_entry_targets(config: Config, filter_key: str, entry: str) -> list[tuple[str, str, str, Path]]:
    """Expand one filter entry to ``(dir_key, channel, video_id, path)`` tuples."""
    if "/" not in entry:
        print(f"WARN: malformed entry '{entry}' under '{filter_key}', skipping", file=sys.stderr)
        return []
    channel, video_id = entry.split("/", 1)
    if not channel or not video_id:
        print(f"WARN: empty channel/id in entry '{entry}', skipping", file=sys.stderr)
        return []

    targets: list[tuple[str, str, str, Path]] = []
    for dir_key in dirs_at_or_upstream_of(filter_key):
        base_dir = resolve_dir(config, dir_key)
        targets.extend((dir_key, channel, video_id, path) for path in find_matching_files(base_dir / channel, video_id))
    return targets


def _resolve_all_targets(config: Config, filter_data: dict[str, list[str]]) -> list[tuple[str, str, str, Path]]:
    """Expand every filter entry to concrete on-disk file targets."""
    targets: list[tuple[str, str, str, Path]] = []
    for filter_key, entries in filter_data.items():
        if filter_key not in _PIPELINE_ORDER:
            print(f"WARN: unknown filter key '{filter_key}', skipping", file=sys.stderr)
            continue
        for entry in entries:
            targets.extend(_resolve_entry_targets(config, filter_key, entry))
    return targets


def _print_targets(targets: list[tuple[str, str, str, Path]], mode_label: str) -> int:
    """Print grouped target listing and return the total byte count."""
    print("=" * 80)
    print(f"{mode_label}: {len(targets)} file(s) resolved from filter entries")
    print("=" * 80)

    grouped: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    for dir_key, channel, video_id, path in targets:
        grouped.setdefault((channel, video_id), []).append((dir_key, path))

    total_bytes = 0
    for (channel, video_id), items in sorted(grouped.items()):
        print(f"\n{channel}/{video_id}")
        for dir_key, path in items:
            try:
                size = path.stat().st_size
            except OSError as exc:
                print(f"  [   ?    ] ({dir_key})  {path}  -- stat failed: {exc}")
                continue
            total_bytes += size
            print(f"  [{size:>14,} B] ({dir_key})  {path}")

    print(f"\nTotal: {len(targets)} file(s), {total_bytes:,} bytes ({total_bytes / 1e9:.2f} GB)")
    return total_bytes


def _delete_targets(targets: list[tuple[str, str, str, Path]]) -> tuple[int, int]:
    """Unlink every target path, returning ``(deleted, failed)`` counts."""
    deleted = 0
    failed = 0
    for _, _, _, path in targets:
        try:
            path.unlink()
            deleted += 1
        except OSError as exc:
            print(f"FAIL: {path}: {exc}", file=sys.stderr)
            failed += 1
    return deleted, failed


def main() -> int:
    """Resolve filter entries to on-disk files and optionally delete them."""
    args = parse_args()

    config = Config(args.config)

    filter_path = Path(args.filter)
    if not filter_path.is_file():
        print(f"ERROR: filter file not found: {filter_path}", file=sys.stderr)
        return 2
    with filter_path.open() as f:
        filter_data = json.load(f)

    targets = _resolve_all_targets(config, filter_data)

    if not targets:
        print("No files on disk match any filtered entry. Nothing to do.")
        return 0

    mode_label = "EXECUTE" if args.execute else "DRY RUN"
    _print_targets(targets, mode_label)

    if not args.execute:
        print("\nDry-run only. Re-run with --execute to actually delete.")
        return 0

    deleted, failed = _delete_targets(targets)
    print(f"\nDeleted: {deleted}  Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
