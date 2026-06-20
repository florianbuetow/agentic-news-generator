#!/usr/bin/env python3
"""Fetch thumbnail images for YouTube videos.

For each given video ID, looks up the existing .info.json file in the channel
metadata directory and writes the fetched thumbnail next to it with the same
stem, in YouTube's original format (jpg or webp).

Usage (explicit):
    uv run python scripts/fetch-video-thumbnails.py <channel_name> <video_id> [<video_id> ...]

Usage (scan all channels):
    uv run python scripts/fetch-video-thumbnails.py

Usage (scan one channel):
    uv run python scripts/fetch-video-thumbnails.py <channel_name>

In scan mode, walks existing .info.json files and fetches any missing thumbnails.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from src.config import Config

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_VIDEO_ID_IN_STEM_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")

# Thumbnail file extensions yt-dlp may produce from YouTube responses.
# Observed in corpus: .jpg, .webp. .png included as conservative safety net
# per yt-dlp's --convert-thumbnails documented format set.
_THUMBNAIL_EXTENSIONS = ("jpg", "webp", "png")


def validate_video_id(video_id: str) -> str:
    """Validate a YouTube video ID for use as an argparse type."""
    if not _VIDEO_ID_RE.match(video_id):
        raise argparse.ArgumentTypeError(f"Invalid YouTube video ID: {video_id!r}. Expected 11 characters of [A-Za-z0-9_-].")
    return video_id


def find_info_json_stem(metadata_video_dir: Path, video_id: str) -> str | None:
    """Find the .info.json file stem for the given video ID in the channel metadata dir."""
    matches = [p for p in metadata_video_dir.glob(f"*[[]{video_id}[]]*.info.json") if not p.name.startswith("._")]
    if not matches:
        return None
    return matches[0].name.removesuffix(".info.json")


def thumbnail_exists(metadata_video_dir: Path, stem: str) -> bool:
    """Return True if any thumbnail extension exists for this stem."""
    return any((metadata_video_dir / f"{stem}.{ext}").exists() for ext in _THUMBNAIL_EXTENSIONS)


def _scan_missing_by_channel(
    metadata_dir: Path,
    metadata_video_subdir: str,
    channel_filter: str | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Return {channel_name: [(video_id, stem)]} for .info.json files missing a sibling thumbnail.

    If channel_filter is given, only that channel's metadata directory is scanned.
    """
    missing_by_channel: dict[str, list[tuple[str, str]]] = {}
    for channel_dir in sorted(ch for ch in metadata_dir.iterdir() if ch.is_dir()):
        if channel_filter is not None and channel_dir.name != channel_filter:
            continue
        metadata_video_dir = channel_dir / metadata_video_subdir
        if not metadata_video_dir.is_dir():
            continue
        missing: list[tuple[str, str]] = []
        for info_file in sorted(metadata_video_dir.glob("*.info.json")):
            if info_file.name.startswith("._"):
                continue
            stem = info_file.name.removesuffix(".info.json")
            match = _VIDEO_ID_IN_STEM_RE.search(stem)
            if not match:
                continue
            if not thumbnail_exists(metadata_video_dir, stem):
                missing.append((match.group(1), stem))
        if missing:
            missing_by_channel[channel_dir.name] = missing
    return missing_by_channel


def _fetch_items(
    channel_name: str,
    items: list[tuple[str, str]],
    metadata_dir: Path,
    metadata_video_subdir: str,
    shell_script: Path,
) -> list[tuple[str, str, str]]:
    """Fetch thumbnails for (video_id, stem) pairs.

    Returns a list of (channel_name, video_id, reason) tuples for any failures.
    """
    failures: list[tuple[str, str, str]] = []
    for video_id, stem in items:
        output_file_base = metadata_dir / channel_name / metadata_video_subdir / stem
        result = subprocess.run(["bash", str(shell_script), str(output_file_base), video_id])
        if result.returncode != 0:
            reason = f"yt-dlp exited {result.returncode}"
            print(f"  ✗ yt-dlp failed for {video_id} ({reason})", file=sys.stderr)
            failures.append((channel_name, video_id, reason))
    return failures


def _scan_and_fetch(config: Config, shell_script: Path, channel_filter: str | None = None) -> int:
    """Scan .info.json files and fetch any missing thumbnails. Optionally limit to one channel."""
    metadata_dir = config.get_data_downloads_metadata_dir()
    metadata_video_subdir = config.get_transcription_metadata_video_subdir()

    if not metadata_dir.exists():
        print(f"Error: metadata directory not found: {metadata_dir}", file=sys.stderr)
        return 1

    if channel_filter is not None:
        channel_metadata_dir = metadata_dir / channel_filter
        if not channel_metadata_dir.is_dir():
            print(f"Error: channel metadata directory not found: {channel_metadata_dir}", file=sys.stderr)
            return 1

    missing_by_channel = _scan_missing_by_channel(metadata_dir, metadata_video_subdir, channel_filter)
    total_missing = sum(len(v) for v in missing_by_channel.values())

    scope = f"channel {channel_filter!r}" if channel_filter is not None else f"{len(missing_by_channel)} channel(s)"

    if total_missing == 0:
        print(f"✓ All .info.json files have a sibling thumbnail ({scope})")
        return 0

    print(f"Found {total_missing} .info.json file(s) missing thumbnails across {scope}. Fetching...")
    print()

    all_failures: list[tuple[str, str, str]] = []
    for channel_name, items in missing_by_channel.items():
        print(f"  {channel_name} ({len(items)} file(s))...", end=" ", flush=True)
        channel_failures = _fetch_items(channel_name, items, metadata_dir, metadata_video_subdir, shell_script)
        if channel_failures:
            print("✗")
            all_failures.extend(channel_failures)
        else:
            print("✓")

    print()
    fetched = total_missing - len(all_failures)
    if all_failures:
        affected_channels = sorted({ch for ch, _, _ in all_failures})
        print("--- Failure Summary ---")
        for channel, video_id, reason in all_failures:
            print(f"❌ {channel}/{video_id}: {reason}")
        print(
            f"\n✗ {len(all_failures)} fetch failure(s) across {len(affected_channels)} channel(s); "
            f"{fetched} thumbnail(s) fetched successfully",
            file=sys.stderr,
        )
        return 1

    print(f"✓ Fetched {total_missing} thumbnail(s)")
    return 0


def _fetch_explicit(
    channel_name: str,
    video_ids: list[str],
    config: Config,
    shell_script: Path,
) -> int:
    """Fetch thumbnails for explicitly given video IDs in a single channel."""
    metadata_dir = config.get_data_downloads_metadata_dir()
    metadata_video_subdir = config.get_transcription_metadata_video_subdir()
    metadata_video_dir = metadata_dir / channel_name / metadata_video_subdir

    if not metadata_video_dir.is_dir():
        print(f"Error: metadata video directory not found: {metadata_video_dir}", file=sys.stderr)
        return 1

    failures: list[tuple[str, str]] = []
    fetched = 0
    skipped = 0
    for video_id in video_ids:
        stem = find_info_json_stem(metadata_video_dir, video_id)
        if stem is None:
            reason = f"no .info.json found in {metadata_video_dir}"
            print(f"✗ {video_id}: {reason}", file=sys.stderr)
            failures.append((video_id, reason))
            continue

        if thumbnail_exists(metadata_video_dir, stem):
            print(f"✓ {video_id}: thumbnail already present (skipped)")
            skipped += 1
            continue

        output_file_base = metadata_video_dir / stem
        result = subprocess.run(["bash", str(shell_script), str(output_file_base), video_id])
        if result.returncode != 0:
            reason = "yt-dlp failed"
            print(f"✗ {video_id}: {reason}", file=sys.stderr)
            failures.append((video_id, reason))
        else:
            fetched += 1

    print()
    print(f"Summary: {fetched} fetched, {skipped} already present, {len(failures)} failed")
    if failures:
        print("--- Failure Summary ---")
        for vid, reason in failures:
            print(f"❌ {vid}: {reason}")
        return 1
    return 0


def main() -> int:
    """Fetch thumbnails for given video IDs, or scan all videos with metadata."""
    parser = argparse.ArgumentParser(
        description="Fetch thumbnail images for YouTube videos, or scan all videos with metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "channel_name",
        nargs="?",
        help=("Channel directory name (e.g. Y_Combinator). Omit to scan all channels; provide alone to scan one channel."),
    )
    parser.add_argument(
        "video_ids",
        nargs="*",
        type=validate_video_id,
        help="One or more YouTube video IDs to fetch thumbnails for in the given channel",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    try:
        config = Config.load_default()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    shell_script = project_root / "scripts" / "fetch-video-thumbnail.sh"

    if args.channel_name is None:
        return _scan_and_fetch(config, shell_script)

    if not args.video_ids:
        return _scan_and_fetch(config, shell_script, channel_filter=args.channel_name)

    return _fetch_explicit(args.channel_name, args.video_ids, config, shell_script)


if __name__ == "__main__":
    sys.exit(main())
