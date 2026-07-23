#!/usr/bin/env python3
"""Build the per-channel publication timeseries CSV and frequency report.

Scans the video metadata written by the download pipeline, writes one CSV row
per published video, writes the channel IDs ranked ascending by publication
frequency, and prints the per-channel frequency table.

The scan is incremental per channel: a channel whose metadata file count is
unchanged since the last build keeps its rows from the existing CSV instead of
being reread, so only channels that actually gained or lost files are rescanned.
"""

import argparse
import json
from pathlib import Path

from src.analytics.errors import AnalyticsError
from src.analytics.publication_timeseries import (
    ChannelFrequency,
    PublicationRecord,
    build_channel_id_to_name,
    build_incremental_records,
    compute_channel_frequencies,
    count_channel_metadata_files,
    group_records_by_channel,
    load_records_from_csv,
    read_recorded_file_counts,
    render_channel_id_ranking,
    render_frequency_table,
    render_timeseries_csv,
    resolve_channel_dirs,
)
from src.analytics.report_writer import write_text_file
from src.config import Config
from src.util.channel_name import sanitize_channel_name


def parse_args() -> argparse.Namespace:
    """Parse the optional channel filter."""
    parser = argparse.ArgumentParser(description="Build the per-channel video publication timeseries and frequency report.")
    parser.add_argument(
        "--channel",
        dest="channel",
        default=None,
        help="Restrict to one channel, given as its config.yaml name or its sanitized directory name.",
    )
    return parser.parse_args()


def output_suffix(channel_filter: str | None) -> str:
    """Return the filename suffix scoping artifacts to the processed channel."""
    if channel_filter is None:
        return ""
    return f"_{sanitize_channel_name(channel_filter)}"


def print_frequency_report(
    records: list[PublicationRecord],
    frequencies: list[ChannelFrequency],
    csv_path: Path,
    ranking_path: Path,
) -> None:
    """Print the per-channel frequency table and the artifact locations."""
    channel_width = max(len("Channel"), max(len(frequency.channel) for frequency in frequencies))
    print()
    print(render_frequency_table(frequencies, channel_width), end="")
    print()
    print(f"Videos: {len(records)} across {len(frequencies)} channel(s)")
    print(f"CSV:     {csv_path}")
    print(f"Ranking: {ranking_path}")


def main() -> int:
    """Rescan only the channels whose file count changed; print the frequency table."""
    args = parse_args()
    channel_filter: str | None = args.channel

    config = Config.load_default()
    metadata_dir = config.get_data_downloads_metadata_dir()
    video_subdir = config.get_transcription_metadata_video_subdir()
    output_dir = config.get_data_output_analytics_dir() / "publication_timeseries"
    suffix = output_suffix(channel_filter)
    csv_path = output_dir / f"publication_timeseries{suffix}.csv"
    ranking_path = output_dir / f"channel_ids_by_frequency{suffix}.txt"
    counts_path = output_dir / f"publication_timeseries{suffix}.filecounts.json"

    print(f"Metadata directory: {metadata_dir}")
    try:
        channel_dirs = resolve_channel_dirs(metadata_dir, video_subdir, channel_filter)
    except AnalyticsError as e:
        print(f"ERROR: {e}")
        return 1

    current_counts = {channel: count_channel_metadata_files(video_dir) for channel, video_dir in channel_dirs}
    recorded_counts = read_recorded_file_counts(counts_path)

    existing_by_channel: dict[str, list[PublicationRecord]] = {}
    if csv_path.is_file():
        try:
            id_to_name = build_channel_id_to_name(metadata_dir, video_subdir, channel_filter)
            existing_by_channel = group_records_by_channel(load_records_from_csv(csv_path, id_to_name))
        except AnalyticsError as e:
            print(f"ERROR: {e}")
            return 1

    records, failures, rescanned_channels = build_incremental_records(
        channel_dirs, current_counts, recorded_counts, existing_by_channel, print
    )

    if not records:
        print("No publication records found - nothing written.")
        return 1

    frequencies = compute_channel_frequencies(records)

    if rescanned_channels:
        print(f"Rescanned {len(rescanned_channels)} changed channel(s); writing artifacts...")
        write_text_file(render_timeseries_csv(records), csv_path)
        write_text_file(render_channel_id_ranking(frequencies), ranking_path)
        write_text_file(json.dumps(current_counts, indent=2, sort_keys=True) + "\n", counts_path)
    else:
        print("All channels unchanged - reused existing CSV, no rescan.")

    print_frequency_report(records, frequencies, csv_path, ranking_path)

    if failures:
        print()
        print("--- Failure Summary ---")
        for failure in failures:
            print(f"❌ {failure}")
        print(f"Encountered {len(failures)} unreadable metadata file(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
