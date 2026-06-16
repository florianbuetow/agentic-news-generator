#!/usr/bin/env python3
"""Show free disk space for each drive referenced by config.yaml paths."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config


def find_mount_point(path: Path) -> Path:
    """Walk up the directory tree until we reach a mount point."""
    resolved = path.resolve()
    while not os.path.ismount(resolved):
        resolved = resolved.parent
    return resolved


def human_readable(n: int) -> tuple[int, str]:
    """Return (value, unit) for n bytes, using integer SI units."""
    if n < 1_000:
        return n, "bytes"
    if n < 1_000_000:
        return n // 1_000, "KB"
    if n < 1_000_000_000:
        return n // 1_000_000, "MB"
    return n // 1_000_000_000, "GB"


def collect_config_paths(config: Config) -> list[Path]:
    """Return all directory paths declared in the paths section of config.yaml."""
    return [
        config.get_data_dir(),
        config.get_data_models_dir(),
        config.get_data_downloads_dir(),
        config.get_data_downloads_videos_dir(),
        config.get_data_downloads_transcripts_dir(),
        config.get_data_downloads_transcripts_hallucinations_dir(),
        config.get_data_downloads_transcripts_cleaned_dir(),
        config.get_data_downloads_transcripts_summaries_dir(),
        config.get_data_downloads_audio_dir(),
        config.get_data_downloads_metadata_dir(),
        config.get_data_output_dir(),
        config.get_data_input_dir(),
        config.get_data_temp_dir(),
        config.get_data_archive_dir(),
        config.get_data_archive_videos_dir(),
        config.get_data_logs_dir(),
        config.get_reports_dir(),
        config.get_data_output_analytics_dir(),
    ]


def main() -> int:
    """Print free disk space for each mount point referenced by config.yaml."""
    project_root = Path(__file__).parent.parent
    config = Config(project_root / "config" / "config.yaml")

    mount_points: dict[Path, int] = {}
    for path in collect_config_paths(config):
        mp = find_mount_point(path)
        if mp not in mount_points:
            mount_points[mp] = shutil.disk_usage(mp).free

    entries = [(mp, *human_readable(free)) for mp, free in sorted(mount_points.items(), key=lambda item: str(item[0]))]

    num_width = max(len(str(value)) for _, value, _ in entries)
    unit_width = max(len(unit) for _, _, unit in entries)
    field_width = num_width + 2  # 2-space minimum left indent

    print("Free disk space:")
    for mp, value, unit in entries:
        print(f"{value:>{field_width}} {unit:<{unit_width}} free on {mp}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
