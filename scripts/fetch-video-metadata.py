#!/usr/bin/env python3
"""Fetch .info.json metadata for specific YouTube video IDs.

For each given video ID, resolves the video's canonical file stem from whatever
artifact still exists — audio WAV, active or archived video, or transcript — and
writes the fetched metadata under that exact stem, so the rest of the pipeline
(which looks up metadata by that stem) can find it. Because archived videos are
searched too, this restores metadata for videos whose WAV was already removed by
``archive-videos`` (e.g. a stale .info.json deleted because its ``timestamp``
field was null — see TROUBLESHOOTING-GUIDE.md).

Usage (explicit):
    uv run python scripts/fetch-video-metadata.py <channel_name> <video_id> [<video_id> ...]

Usage (scan):
    uv run python scripts/fetch-video-metadata.py

In scan mode, discovers all active and archived video files and fetches any missing metadata.

Example:
    uv run python scripts/fetch-video-metadata.py Y_Combinator DOez-RwJ7mg lJausFj_Dto
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from src.config import Config
from src.util.media_stem import resolve_media_stem

# YouTube video IDs: exactly 11 chars of [A-Za-z0-9_-]
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_VIDEO_ID_IN_STEM_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")

_VIDEO_EXTENSIONS = {"mp4", "mkv", "webm", "m4a", "mov", "m4v", "avi", "flv"}


def validate_video_id(video_id: str) -> str:
    """Validate a YouTube video ID for use as an argparse type."""
    if not _VIDEO_ID_RE.match(video_id):
        raise argparse.ArgumentTypeError(f"Invalid YouTube video ID: {video_id!r}. Expected 11 characters of [A-Za-z0-9_-].")
    return video_id


def _channel_search_dirs(config: Config, channel_name: str) -> list[Path]:
    """Artifact directories to search for a channel's media stem, most authoritative first.

    Audio and active-video directories cover in-flight videos; the archive video
    directory and transcripts cover videos whose WAV/MP4 were removed by
    archival, so metadata can still be restored for them.
    """
    return [
        config.get_data_downloads_audio_dir() / channel_name,
        config.get_data_downloads_videos_dir() / channel_name,
        config.get_data_archive_videos_dir() / channel_name,
        config.get_data_downloads_transcripts_dir() / channel_name,
    ]


def _scan_missing_by_channel(
    video_roots: list[Path],
    metadata_dir: Path,
    metadata_video_subdir: str,
) -> dict[str, list[tuple[str, str]]]:
    """Return {channel_name: [(video_id, stem)]} for video files missing .info.json.

    Scans each root in order (active videos first, then the archive) so a video
    present in both is judged once, by its active copy. Archived videos are
    included so their metadata can be restored after the WAV is gone.
    """
    missing_by_channel: dict[str, list[tuple[str, str]]] = {}
    seen_by_channel: dict[str, set[str]] = {}
    for videos_dir in video_roots:
        if not videos_dir.is_dir():
            continue
        for channel_dir in sorted(ch for ch in videos_dir.iterdir() if ch.is_dir()):
            metadata_channel_dir = metadata_dir / channel_dir.name / metadata_video_subdir
            seen = seen_by_channel.setdefault(channel_dir.name, set())
            for video_file in sorted(channel_dir.iterdir()):
                if video_file.name.startswith("._"):
                    continue
                if video_file.suffix.lstrip(".").lower() not in _VIDEO_EXTENSIONS:
                    continue
                match = _VIDEO_ID_IN_STEM_RE.search(video_file.stem)
                if not match:
                    continue
                video_id = match.group(1)
                if video_id in seen:
                    continue
                seen.add(video_id)
                if not (metadata_channel_dir / f"{video_file.stem}.info.json").exists():
                    missing_by_channel.setdefault(channel_dir.name, []).append((video_id, video_file.stem))
    return missing_by_channel


def _fetch_items(
    channel_name: str,
    items: list[tuple[str, str]],
    metadata_dir: Path,
    metadata_video_subdir: str,
    shell_script: Path,
) -> int:
    """Fetch metadata for (video_id, stem) pairs. Returns failure count."""
    failures = 0
    for video_id, stem in items:
        output_file_base = metadata_dir / channel_name / metadata_video_subdir / stem
        result = subprocess.run(["bash", str(shell_script), str(output_file_base), video_id])
        if result.returncode != 0:
            print(f"  ✗ yt-dlp failed for {video_id}", file=sys.stderr)
            failures += 1
    return failures


def _scan_and_fetch(config: Config, shell_script: Path) -> int:
    """Scan active and archived video files and fetch any missing metadata."""
    videos_dir = config.get_data_downloads_videos_dir()
    archive_videos_dir = config.get_data_archive_videos_dir()
    metadata_dir = config.get_data_downloads_metadata_dir()
    metadata_video_subdir = config.get_transcription_metadata_video_subdir()

    video_roots = [videos_dir, archive_videos_dir]
    if not any(root.exists() for root in video_roots):
        print(f"Error: no video directory found (looked in {videos_dir} and {archive_videos_dir})", file=sys.stderr)
        return 1

    missing_by_channel = _scan_missing_by_channel(video_roots, metadata_dir, metadata_video_subdir)
    total_missing = sum(len(v) for v in missing_by_channel.values())

    if total_missing == 0:
        print("✓ All video files have metadata")
        return 0

    print(f"Found {total_missing} video file(s) missing metadata across {len(missing_by_channel)} channel(s). Fetching...")
    print()

    failures = 0
    for channel_name, items in missing_by_channel.items():
        print(f"  {channel_name} ({len(items)} file(s))...", end=" ", flush=True)
        n = _fetch_items(channel_name, items, metadata_dir, metadata_video_subdir, shell_script)
        if n > 0:
            print("✗")
            failures += 1
        else:
            print("✓")

    print()
    if failures:
        print(f"✗ {failures} channel(s) had fetch failures", file=sys.stderr)
        return 1

    print(f"✓ Fetched {total_missing} metadata file(s)")
    return 0


def main() -> int:
    """Fetch .info.json metadata files for given video IDs, or scan all videos."""
    parser = argparse.ArgumentParser(
        description="Fetch .info.json metadata for YouTube video IDs, or scan all active and archived videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "channel_name",
        nargs="?",
        help="Channel directory name (e.g. Y_Combinator). Omit to scan all active and archived videos.",
    )
    parser.add_argument(
        "video_ids",
        nargs="*",
        type=validate_video_id,
        help="One or more YouTube video IDs to fetch metadata for",
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

    shell_script = project_root / "scripts" / "fetch-video-metadata.sh"

    if args.channel_name is None:
        return _scan_and_fetch(config, shell_script)

    if not args.video_ids:
        parser.error("video_ids required when channel_name is given")

    metadata_dir = config.get_data_downloads_metadata_dir()
    metadata_video_subdir = config.get_transcription_metadata_video_subdir()

    search_dirs = _channel_search_dirs(config, args.channel_name)
    metadata_channel_video_dir = metadata_dir / args.channel_name / metadata_video_subdir

    if not any(directory.is_dir() for directory in search_dirs):
        print(f"Error: no audio/video/transcript directory found for channel {args.channel_name!r}", file=sys.stderr)
        return 1

    failures = 0
    for video_id in args.video_ids:
        stem = resolve_media_stem(search_dirs, video_id)
        if stem is None:
            print(f"✗ No audio/video/transcript artifact found for video ID {video_id} in channel {args.channel_name}", file=sys.stderr)
            failures += 1
            continue

        output_file_base = metadata_channel_video_dir / stem
        result = subprocess.run(
            ["bash", str(shell_script), str(output_file_base), video_id],
        )
        if result.returncode != 0:
            print(f"✗ yt-dlp failed for {video_id}", file=sys.stderr)
            failures += 1

    if failures > 0:
        print(f"\n{failures} of {len(args.video_ids)} fetches failed", file=sys.stderr)
        return 1

    print(f"\n✓ Fetched metadata for {len(args.video_ids)} video(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
