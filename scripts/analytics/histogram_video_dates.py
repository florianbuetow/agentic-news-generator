#!/usr/bin/env python3
"""Render terminal histograms of video release dates.

Scans the video metadata written by the download pipeline and prints three
rolling release-date histograms, top to bottom: bi-weekly buckets, one bar per
day over the last 30 days, and one bar per day over the last 7 days. Every
window ends on today's UTC date.
"""

from datetime import UTC, datetime

from src.analytics.date_histogram import build_date_histogram, render_date_histogram
from src.analytics.errors import AnalyticsError
from src.analytics.publication_timeseries import collect_publication_records
from src.config import Config

BIWEEKLY_BUCKET_DAYS: int = 14
BIWEEKLY_BUCKET_COUNT: int = 30
BIWEEKLY_LABEL_STRIDE: int = 5
MONTH_WINDOW_DAYS: int = 30
MONTH_LABEL_STRIDE: int = 7
WEEK_WINDOW_DAYS: int = 7
WEEK_LABEL_STRIDE: int = 1
DAILY_BUCKET_DAYS: int = 1
CHART_HEIGHT: int = 12
NARROW_CELL_WIDTH: int = 2
WIDE_CELL_WIDTH: int = 6


def main() -> int:
    """Scan the video metadata and print the three release-date histograms."""
    config = Config.load_default()
    metadata_dir = config.get_data_downloads_metadata_dir()
    video_subdir = config.get_transcription_metadata_video_subdir()

    print(f"Metadata directory: {metadata_dir}")
    try:
        records, failures = collect_publication_records(metadata_dir, video_subdir, None, print)
    except AnalyticsError as e:
        print(f"ERROR: {e}")
        return 1

    if not records:
        print("No publication records found - nothing to plot.")
        return 1

    today = datetime.now(UTC).date()
    release_dates = [record.published_at.date() for record in records]
    print()
    print(f"Videos with metadata: {len(release_dates):,} | Today (UTC): {today.isoformat()}")

    biweekly_window_days = BIWEEKLY_BUCKET_DAYS * BIWEEKLY_BUCKET_COUNT
    charts: list[tuple[str, int, int, int, int]] = [
        (
            f"Releases per two weeks (last {biweekly_window_days} days)",
            BIWEEKLY_BUCKET_DAYS,
            BIWEEKLY_BUCKET_COUNT,
            NARROW_CELL_WIDTH,
            BIWEEKLY_LABEL_STRIDE,
        ),
        (f"Releases per day (last {MONTH_WINDOW_DAYS} days)", DAILY_BUCKET_DAYS, MONTH_WINDOW_DAYS, NARROW_CELL_WIDTH, MONTH_LABEL_STRIDE),
        (f"Releases per day (last {WEEK_WINDOW_DAYS} days)", DAILY_BUCKET_DAYS, WEEK_WINDOW_DAYS, WIDE_CELL_WIDTH, WEEK_LABEL_STRIDE),
    ]
    for title, bucket_days, bucket_count, cell_width, label_stride in charts:
        histogram = build_date_histogram(release_dates, today, bucket_days, bucket_count, title)
        print()
        print(render_date_histogram(histogram, CHART_HEIGHT, cell_width, label_stride))

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
