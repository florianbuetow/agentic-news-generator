"""Unit tests for the release-date histogram bucketing and rendering."""

from datetime import date, timedelta

import pytest

from src.analytics.date_histogram import (
    FULL_BLOCK,
    DateHistogram,
    build_date_histogram,
    format_bucket_label,
    render_bucket_cell,
    render_date_histogram,
    render_x_axis,
    y_axis_ticks,
)

TODAY = date(2026, 8, 24)


def build_daily(release_dates: list[date], bucket_count: int) -> DateHistogram:
    """Build a daily-bucket histogram ending on the fixed test date."""
    return build_date_histogram(release_dates, TODAY, 1, bucket_count, "Daily")


class TestBuildDateHistogram:
    """Bucketing release dates into a rolling window."""

    def test_window_spans_bucket_days_times_bucket_count(self) -> None:
        """A 30-bucket daily window starts 29 days before its end."""
        histogram = build_daily([], 30)
        assert histogram.window_start == TODAY - timedelta(days=29)
        assert histogram.window_end == TODAY

    def test_buckets_are_contiguous_and_cover_the_window(self) -> None:
        """Bucket spans tile the window with no gap and no overlap."""
        histogram = build_date_histogram([], TODAY, 14, 30, "Biweekly")
        assert histogram.buckets[0].start == histogram.window_start
        assert histogram.buckets[-1].end == histogram.window_end
        for previous, current in zip(histogram.buckets, histogram.buckets[1:], strict=False):
            assert current.start == previous.end + timedelta(days=1)

    def test_each_biweekly_bucket_covers_fourteen_days(self) -> None:
        """Every bucket span is bucket_days wide, inclusive."""
        histogram = build_date_histogram([], TODAY, 14, 30, "Biweekly")
        for bucket in histogram.buckets:
            assert (bucket.end - bucket.start).days + 1 == 14

    def test_newest_bucket_ends_today(self) -> None:
        """The rolling window always ends on the reference date."""
        histogram = build_date_histogram([TODAY], TODAY, 14, 30, "Biweekly")
        assert histogram.buckets[-1].end == TODAY
        assert histogram.buckets[-1].count == 1

    def test_release_on_window_start_lands_in_first_bucket(self) -> None:
        """The window start date is inside the window."""
        histogram = build_daily([TODAY - timedelta(days=29)], 30)
        assert histogram.buckets[0].count == 1
        assert histogram.total_in_window == 1

    def test_release_before_window_counts_as_older(self) -> None:
        """A date one day before the window is not dropped silently."""
        histogram = build_daily([TODAY - timedelta(days=30)], 30)
        assert histogram.older_count == 1
        assert histogram.total_in_window == 0

    def test_release_after_window_counts_as_newer(self) -> None:
        """A future-dated release is not dropped silently."""
        histogram = build_daily([TODAY + timedelta(days=1)], 30)
        assert histogram.newer_count == 1
        assert histogram.total_in_window == 0

    def test_counts_land_in_the_right_daily_buckets(self) -> None:
        """Each date increments exactly its own day bucket."""
        histogram = build_daily([TODAY, TODAY, TODAY - timedelta(days=6)], 7)
        assert [bucket.count for bucket in histogram.buckets] == [1, 0, 0, 0, 0, 0, 2]

    def test_biweekly_boundary_dates_split_between_buckets(self) -> None:
        """Dates on both sides of a bucket boundary land apart."""
        histogram = build_date_histogram([TODAY - timedelta(days=13), TODAY - timedelta(days=14)], TODAY, 14, 2, "Biweekly")
        assert [bucket.count for bucket in histogram.buckets] == [1, 1]

    @pytest.mark.parametrize(("bucket_days", "bucket_count"), [(0, 30), (-1, 30), (14, 0), (14, -5)])
    def test_non_positive_dimensions_raise(self, bucket_days: int, bucket_count: int) -> None:
        """Bucket geometry must be positive."""
        with pytest.raises(ValueError, match="greater than zero"):
            build_date_histogram([], TODAY, bucket_days, bucket_count, "Bad")


class TestFormatBucketLabel:
    """X-axis label formatting per bucket width."""

    def test_daily_bucket_uses_compact_month_day(self) -> None:
        """One-day buckets sit in short windows and stay compact."""
        assert format_bucket_label(date(2026, 8, 3), 1) == "08-03"

    def test_wider_bucket_uses_full_iso_date(self) -> None:
        """Multi-day buckets can span years and keep the year."""
        assert format_bucket_label(date(2026, 8, 3), 14) == "2026-08-03"


class TestRenderBucketCell:
    """Cell fill selection for one bucket at one row."""

    def test_count_at_or_above_ceiling_is_full(self) -> None:
        """A bucket that fills the row shows a full block."""
        assert render_bucket_cell(10, 8.0, 10.0, 2.0) == FULL_BLOCK

    def test_count_just_above_floor_is_half(self) -> None:
        """A bucket under half the row shows a half block."""
        assert render_bucket_cell(9, 8.0, 10.0, 2.0) != " "

    def test_count_at_or_below_floor_is_blank(self) -> None:
        """A bucket that never reaches the row stays empty."""
        assert render_bucket_cell(8, 8.0, 10.0, 2.0) == " "


class TestYAxisTicks:
    """Row-to-label mapping for the count axis."""

    def test_top_row_shows_the_maximum(self) -> None:
        """The 100% tick labels the tallest bucket count."""
        ticks = y_axis_ticks(12, 40)
        assert ticks[12] == "40"
        assert len(ticks) == 12
        assert ticks[11] == ""

    def test_colliding_fractions_keep_the_larger_one(self) -> None:
        """With a one-row chart every fraction rounds to row one."""
        ticks = y_axis_ticks(1, 3)
        assert ticks == {1: "3"}


class TestRenderDateHistogram:
    """Rendering a bucketed histogram as terminal text."""

    def test_empty_window_prints_message_instead_of_chart(self) -> None:
        """An empty window says so instead of drawing empty axes."""
        rendered = render_date_histogram(build_daily([], 7), 12, 6, 1)
        assert "No videos released in this window." in rendered
        assert FULL_BLOCK not in rendered

    def test_header_names_window_and_bar_width(self) -> None:
        """The header states the window dates and the bucket width."""
        rendered = render_date_histogram(build_daily([TODAY], 7), 12, 6, 1)
        assert "Videos: 1 | Window: 2026-08-18 to 2026-08-24 (7 days, 1 bar = 1 day(s))" in rendered

    def test_chart_shrinks_to_max_count_rows(self) -> None:
        """With a tallest bucket of 2, the bar area is 2 rows tall."""
        rendered = render_date_histogram(build_daily([TODAY, TODAY, TODAY - timedelta(days=1)], 7), 12, 6, 1)
        bar_rows = [line for line in rendered.splitlines() if line.rstrip().endswith("|") and FULL_BLOCK in line]
        assert len(bar_rows) == 2

    def test_tallest_bucket_reaches_the_top_row(self) -> None:
        """The maximum bucket fills the highest bar row."""
        rendered = render_date_histogram(build_daily([TODAY] * 5, 7), 12, 6, 1)
        top_bar_row = next(line for line in rendered.splitlines() if FULL_BLOCK in line)
        assert top_bar_row.startswith(f"{'5':>6} |")

    def test_daily_x_axis_labels_every_bucket_start(self) -> None:
        """With stride one every bucket start date appears."""
        rendered = render_date_histogram(build_daily([TODAY], 7), 12, 6, 1)
        for offset in range(7):
            assert (TODAY - timedelta(days=offset)).strftime("%m-%d") in rendered

    def test_biweekly_x_axis_labels_use_iso_dates(self) -> None:
        """Bi-weekly labels keep the year."""
        histogram = build_date_histogram([TODAY], TODAY, 14, 30, "Biweekly")
        rendered = render_date_histogram(histogram, 12, 2, 5)
        assert histogram.buckets[0].start.isoformat() in rendered

    def test_future_dated_releases_are_reported(self) -> None:
        """A future-dated release shows up under the chart."""
        rendered = render_date_histogram(build_daily([TODAY, TODAY + timedelta(days=2)], 7), 12, 6, 1)
        assert "Future-dated releases after 2026-08-24: 1 video(s)" in rendered

    @pytest.mark.parametrize(("chart_height", "cell_width", "label_stride"), [(0, 6, 1), (12, 0, 1), (12, 6, 0)])
    def test_non_positive_render_parameters_raise(self, chart_height: int, cell_width: int, label_stride: int) -> None:
        """Render geometry must be positive."""
        with pytest.raises(ValueError, match="greater than zero"):
            render_date_histogram(build_daily([TODAY], 7), chart_height, cell_width, label_stride)


class TestRenderXAxis:
    """Tick and label row layout."""

    def test_tick_marks_sit_at_labeled_bucket_edges(self) -> None:
        """Every labeled bucket gets a tick at its left edge."""
        histogram = build_daily([TODAY], 7)
        tick_row = render_x_axis(histogram, 6, 2)[0]
        plot = tick_row[8:]
        assert [index for index, char in enumerate(plot) if char == "|"] == [0, 12, 24, 36]

    def test_labels_never_exceed_plot_width(self) -> None:
        """Labels near the right edge are clamped inside the plot."""
        histogram = build_date_histogram([TODAY], TODAY, 14, 30, "Biweekly")
        for row in render_x_axis(histogram, 2, 5):
            assert len(row) <= 8 + 30 * 2
