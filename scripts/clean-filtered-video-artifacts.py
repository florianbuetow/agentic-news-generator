#!/usr/bin/env python3
"""Preview or remove every pipeline artifact belonging to a filtered video ID."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import Config
from src.pipeline_integrity import (
    find_filtered_file_offenders,
    load_filtered_video_ids,
    relative_to_data_dir,
    remove_filtered_artifacts,
    video_ids_from_artifacts,
    write_report,
)


def apply_requested(argv: list[str]) -> bool:
    """Return whether the exact optional ``--apply`` argument was provided."""
    if len(argv) == 1:
        return False
    if len(argv) == 2 and argv[1] == "--apply":
        return True
    print("Usage: clean-filtered-video-artifacts.py [--apply]", file=sys.stderr)
    raise ValueError("invalid command-line arguments")


def main() -> int:
    """Recompute filtered artifacts, write the report, and optionally delete them."""
    try:
        should_apply = apply_requested(sys.argv)
    except ValueError:
        return 2

    try:
        config = Config.load_default()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    data_dir = config.get_data_dir()
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    filter_path = config.get_filefilter_path()
    if not filter_path.is_file():
        print(f"Error: filter file not found: {filter_path}", file=sys.stderr)
        return 1

    try:
        filtered_ids = load_filtered_video_ids(filter_path)
        print(f"Scanning pipeline data for {len(filtered_ids)} filtered video ID(s)...", flush=True)

        def announce_scan(base_dir: Path) -> None:
            print(f"Scanning: {relative_to_data_dir(base_dir, data_dir)}", flush=True)

        artifacts = find_filtered_file_offenders(config, filtered_ids, announce_scan)
        artifact_video_ids = video_ids_from_artifacts(artifacts)
        report_path = config.get_reports_dir().resolve() / "files-whose-video-id-is-listed-in-filefilter.txt"
        write_report(report_path, [str(artifact.resolve()) for artifact in artifacts])
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        print(f"Error finding filtered video artifacts: {exc}", file=sys.stderr)
        return 1

    print(f"Found {len(artifact_video_ids)} videos ({len(artifacts)} artifact files).", flush=True)
    print(f"Report: {report_path}", flush=True)

    if not should_apply:
        print("Dry run: no files were deleted.")
        print("Apply with: just clean-filtered-video-artifacts --apply")
        return 0

    deleted, failures = remove_filtered_artifacts(config, filtered_ids, artifacts)
    if failures:
        for failed_path, reason in failures:
            print(f"ERROR: {failed_path}: {reason}", file=sys.stderr)
        print(f"Deleted {deleted} file(s); {len(failures)} deletion(s) failed.", file=sys.stderr)
        return 1

    print(f"Deleted artifacts for {len(artifact_video_ids)} videos ({deleted} artifact files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
