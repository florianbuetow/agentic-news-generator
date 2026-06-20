#!/usr/bin/env python3
"""Check for WAV files missing their .info.json metadata and fetch them automatically."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from src.config import Config

_VIDEO_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


def scan_channel(
    channel_dir: Path,
    metadata_dir: Path,
    metadata_video_subdir: str,
) -> list[tuple[str, str]]:
    """Return (video_id, wav_stem) pairs missing .info.json in this channel."""
    metadata_channel_dir = metadata_dir / channel_dir.name / metadata_video_subdir
    missing: list[tuple[str, str]] = []
    for wav_file in sorted(channel_dir.glob("*.wav")):
        if wav_file.name.startswith("._"):
            continue
        match = _VIDEO_ID_RE.search(wav_file.stem)
        if not match:
            continue
        if not (metadata_channel_dir / f"{wav_file.stem}.info.json").exists():
            missing.append((match.group(1), wav_file.stem))
    return missing


def main() -> int:
    """Scan all channels for WAV files missing .info.json and fetch them."""
    project_root = Path(__file__).parent.parent

    try:
        config = Config.load_default()
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    audio_dir = Path(config.get_data_downloads_audio_dir())
    metadata_dir = Path(config.get_data_downloads_metadata_dir())
    metadata_video_subdir = config.get_transcription_metadata_video_subdir()

    if not audio_dir.exists():
        print(f"Error: audio directory not found: {audio_dir}", file=sys.stderr)
        return 1

    channel_dirs = sorted(ch for ch in audio_dir.iterdir() if ch.is_dir())
    n = len(channel_dirs)

    print(f"Scanning {n} channel(s)...")
    missing_by_channel: dict[str, list[tuple[str, str]]] = {}
    total_wav = 0

    for i, channel_dir in enumerate(channel_dirs, 1):
        wav_count = sum(1 for f in channel_dir.glob("*.wav") if not f.name.startswith("._"))
        total_wav += wav_count
        items = scan_channel(channel_dir, metadata_dir, metadata_video_subdir)
        label = f"  [{i:>{len(str(n))}}/{n}] {channel_dir.name}"
        if items:
            print(f"{label} — {len(items)} missing")
            missing_by_channel[channel_dir.name] = items
        else:
            print(f"{label} — ok")

    print()

    if not missing_by_channel:
        print(f"✓ All {total_wav} WAV file(s) have metadata")
        return 0

    total_missing = sum(len(v) for v in missing_by_channel.values())
    print(f"Found {total_missing} missing file(s) across {len(missing_by_channel)} channel(s). Fetching...\n")

    fetch_script = project_root / "scripts" / "fetch-video-metadata.py"
    failures = 0

    for channel_name, items in missing_by_channel.items():
        video_ids = [video_id for video_id, _ in items]
        print(f"  {channel_name} ({len(items)} file(s))...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, str(fetch_script), channel_name, "--", *video_ids],
            capture_output=True,
        )
        if result.returncode != 0:
            print("✗")
            print(result.stderr.decode(errors="replace"), file=sys.stderr)
            failures += 1
        else:
            print("✓")

    print()
    if failures:
        print(f"✗ {failures} channel(s) had fetch failures", file=sys.stderr)
        return 1

    print(f"✓ Fetched {total_missing} metadata file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
