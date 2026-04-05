#!/usr/bin/env python3
"""Count total hours of audio by summing the last timestamp in each transcript file."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config

TIMESTAMP_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),\d{3}\s*-->\s*(\d{2}):(\d{2}):(\d{2}),\d{3}")


def last_end_seconds(srt_path: Path) -> float:
    """Extract the end timestamp (in seconds) from the last subtitle entry."""
    last_h, last_m, last_s = 0, 0, 0
    for line in srt_path.read_text(errors="replace").splitlines():
        m = TIMESTAMP_RE.match(line)
        if m:
            last_h, last_m, last_s = int(m.group(4)), int(m.group(5)), int(m.group(6))
    return last_h * 3600 + last_m * 60 + last_s


def main() -> int:
    """Sum transcript durations and print total audio hours."""
    project_root = Path(__file__).parent.parent
    config = Config(project_root / "config" / "config.yaml")

    transcripts_dir = config.getDataDownloadsTranscriptsDir()
    if not transcripts_dir.exists():
        print(f"Error: Transcripts directory not found: {transcripts_dir}")
        return 1

    total_seconds = 0.0
    file_count = 0
    channel_totals: dict[str, tuple[int, float]] = {}

    for channel_dir in sorted(transcripts_dir.iterdir()):
        if not channel_dir.is_dir():
            continue
        ch_seconds = 0.0
        ch_count = 0
        for srt_file in sorted(channel_dir.iterdir()):
            if srt_file.suffix != ".srt" or srt_file.name.startswith("._"):
                continue
            ch_seconds += last_end_seconds(srt_file)
            ch_count += 1
        if ch_count > 0:
            channel_totals[channel_dir.name] = (ch_count, ch_seconds)
            total_seconds += ch_seconds
            file_count += ch_count

    if file_count == 0:
        print("No transcript files found.")
        return 0

    total_hours = total_seconds / 3600
    channel_width = min(max(len(n) for n in channel_totals) + 2, 45)

    print()
    print("=" * (channel_width + 30))
    print("TOTAL AUDIO HOURS (from transcripts)")
    print("=" * (channel_width + 30))
    print()
    print(f"{'Channel':<{channel_width}} {'Files':>6} {'Hours':>8}")
    print("-" * (channel_width + 30))

    for name in sorted(channel_totals):
        count, secs = channel_totals[name]
        display = name[:channel_width] if len(name) > channel_width else name
        print(f"{display:<{channel_width}} {count:>6} {secs / 3600:>8.1f}")

    print("-" * (channel_width + 30))
    print(f"{'TOTAL':<{channel_width}} {file_count:>6} {total_hours:>8.1f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
