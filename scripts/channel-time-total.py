#!/usr/bin/env python3
"""Compute summed transcript time for a cleaned-transcript channel directory."""

from __future__ import annotations

import re
import sys
from collections import deque
from pathlib import Path

TIMESTAMP_RE = re.compile(r"(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)[,.](?P<millis>\d{3})")


def timestamp_to_seconds(timestamp: str) -> int | None:
    """Convert `HH:MM:SS,mmm` or `HH:MM:SS.mmm` to integer seconds."""
    match = TIMESTAMP_RE.fullmatch(timestamp.strip())
    if match is None:
        return None
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    millis = int(match.group("millis"))
    return (hours * 3600) + (minutes * 60) + seconds + (millis // 1000)


def extract_last_timestamp_seconds(path: Path, tail_line_count: int = 30) -> int | None:
    """Read the last lines of an SRT and return the final timestamp in seconds."""
    tail_lines: deque[str] = deque(maxlen=tail_line_count)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                tail_lines.append(line)
    except OSError:
        return None

    last_seconds: int | None = None
    for line in tail_lines:
        for match in TIMESTAMP_RE.finditer(line):
            parsed_seconds = timestamp_to_seconds(match.group(0))
            if parsed_seconds is not None:
                last_seconds = parsed_seconds

    return last_seconds


def format_seconds_as_dhm(total_seconds: int) -> str:
    """Format seconds as `<days>d<hours>h<minutes>m`."""
    if total_seconds < 0:
        total_seconds = 0
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    return f"{days}d{hours}h{minutes}m"


def main() -> int:
    """CLI entrypoint for per-channel duration calculation."""
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/channel-time-total.py <channel-cleaned-srt-dir>")
        return 1

    channel_dir = Path(sys.argv[1])
    if not channel_dir.exists() or not channel_dir.is_dir():
        print(f"Error: channel directory not found: {channel_dir}")
        return 1

    total_seconds = 0
    counted_files = 0
    missing_timestamp_files = 0

    for srt_file in channel_dir.rglob("*.srt"):
        if not srt_file.is_file() or srt_file.name.startswith("._"):
            continue
        last_seconds = extract_last_timestamp_seconds(srt_file)
        if last_seconds is None or last_seconds < 0:
            missing_timestamp_files += 1
            continue
        total_seconds += last_seconds
        counted_files += 1

    print(format_seconds_as_dhm(total_seconds))
    print(f"seconds={total_seconds} files={counted_files} missing_timestamps={missing_timestamp_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
