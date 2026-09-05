#!/usr/bin/env python3
"""Run staged, read-only checks for inconsistent video pipeline state."""

from __future__ import annotations

import json
import sys

from src.config import Config
from src.pipeline_integrity import run_pipeline_integrity


def main() -> int:
    """Load project configuration and run the pipeline integrity stages."""
    try:
        config = Config.load_default()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    data_dir = config.get_data_dir()
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    videos_dir = config.get_data_downloads_videos_dir()
    if not videos_dir.is_dir():
        print(f"Error: videos directory not found: {videos_dir}", file=sys.stderr)
        return 1

    filter_path = config.get_filefilter_path()
    if not filter_path.is_file():
        print(f"Error: filter file not found: {filter_path}", file=sys.stderr)
        return 1

    try:
        return run_pipeline_integrity(config, filter_path)
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        print(f"Error checking pipeline integrity: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
