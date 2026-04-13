#!/usr/bin/env python3
"""Fetch .info.json metadata for specific YouTube video IDs.

For each given video ID, looks up the existing WAV file stem in the channel
audio directory and writes the fetched metadata under that exact stem, so the
rest of the pipeline (which looks up metadata by WAV stem) can find it.

Usage:
    uv run python scripts/fetch-video-metadata.py <channel_name> <video_id> [<video_id> ...]

Example:
    uv run python scripts/fetch-video-metadata.py Y_Combinator DOez-RwJ7mg lJausFj_Dto
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config  # noqa: E402

# YouTube video IDs: exactly 11 chars of [A-Za-z0-9_-]
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def validate_video_id(video_id: str) -> str:
    """Validate a YouTube video ID for use as an argparse type."""
    if not _VIDEO_ID_RE.match(video_id):
        raise argparse.ArgumentTypeError(f"Invalid YouTube video ID: {video_id!r}. Expected 11 characters of [A-Za-z0-9_-].")
    return video_id


def find_wav_stem(audio_channel_dir: Path, video_id: str) -> str | None:
    """Find the WAV file stem in the channel audio directory matching the given video ID."""
    matches = list(audio_channel_dir.glob(f"*[[]{video_id}[]]*.wav"))
    matches = [m for m in matches if not m.name.startswith("._")]
    if not matches:
        return None
    return matches[0].stem


def main() -> int:
    """Fetch .info.json metadata files for given video IDs."""
    parser = argparse.ArgumentParser(
        description="Fetch .info.json metadata for specific YouTube video IDs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "channel_name",
        help="Channel directory name (as stored under the audio/metadata dirs, e.g. Y_Combinator)",
    )
    parser.add_argument(
        "video_ids",
        nargs="+",
        type=validate_video_id,
        help="One or more YouTube video IDs to fetch metadata for",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"

    try:
        config = Config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    audio_dir = Path(config.getDataDownloadsAudioDir())
    metadata_dir = Path(config.getDataDownloadsMetadataDir())
    metadata_video_subdir = config.getTranscriptionMetadataVideoSubdir()

    audio_channel_dir = audio_dir / args.channel_name
    metadata_channel_video_dir = metadata_dir / args.channel_name / metadata_video_subdir

    if not audio_channel_dir.is_dir():
        print(f"Error: audio channel directory not found: {audio_channel_dir}", file=sys.stderr)
        return 1

    shell_script = project_root / "scripts" / "fetch-video-metadata.sh"

    failures = 0
    for video_id in args.video_ids:
        stem = find_wav_stem(audio_channel_dir, video_id)
        if stem is None:
            print(f"✗ No WAV file found for video ID {video_id} in {audio_channel_dir}", file=sys.stderr)
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
