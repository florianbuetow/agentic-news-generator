"""Unit tests for the date-window video listing."""

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from src.analytics.errors import AnalyticsError
from src.analytics.publication_timeseries import PublicationRecord
from src.analytics.video_listing import (
    ChannelListing,
    DateWindow,
    WindowArgumentError,
    group_by_channel_chronological,
    parse_window_args,
    records_from_rows,
    render_video_listing,
    select_records_in_window,
)

TODAY = date(2026, 9, 4)


def make_record(channel: str, published_at: datetime, **overrides: Any) -> PublicationRecord:
    """Build a synthetic publication record."""
    fields: dict[str, Any] = {
        "channel": channel,
        "channel_id": f"UC{channel}",
        "published_at": published_at,
        "title": "A Video",
        "duration_minutes": 10,
        "video_id": "abc123XYZ09",
    }
    fields.update(overrides)
    return PublicationRecord(**fields)


def at(day: date, hour: int, minute: int) -> datetime:
    """Build a UTC instant on the given calendar date."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


class TestDateWindow:
    """The inclusive calendar-date window model."""

    def test_single_day_window_has_one_day(self) -> None:
        """Start and end on the same date is a one-day window."""
        assert DateWindow(start=TODAY, end=TODAY).day_count == 1

    def test_day_count_is_inclusive_of_both_ends(self) -> None:
        """A window from the 1st to the 31st covers 31 days."""
        assert DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 31)).day_count == 31

    def test_start_after_end_is_rejected(self) -> None:
        """A window cannot end before it starts."""
        with pytest.raises(ValueError, match="after"):
            DateWindow(start=TODAY, end=TODAY - timedelta(days=1))


class TestParseWindowArgs:
    """Turning the just-target arguments into a window."""

    def test_since_runs_from_the_date_to_today(self) -> None:
        """'since <date>' ends on the reference date."""
        window = parse_window_args(["since", "2026-08-01"], TODAY)
        assert window == DateWindow(start=date(2026, 8, 1), end=TODAY)

    def test_two_dates_are_the_inclusive_bounds(self) -> None:
        """'<from> <to>' uses both dates as given."""
        window = parse_window_args(["2026-08-01", "2026-08-31"], TODAY)
        assert window == DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 31))

    def test_same_date_twice_is_a_one_day_window(self) -> None:
        """'<date> <date>' selects exactly that day."""
        window = parse_window_args(["2026-08-15", "2026-08-15"], TODAY)
        assert window.day_count == 1

    def test_last_n_covers_n_calendar_days_ending_today(self) -> None:
        """'last 7' is today plus the six days before it."""
        window = parse_window_args(["last", "7"], TODAY)
        assert window == DateWindow(start=TODAY - timedelta(days=6), end=TODAY)
        assert window.day_count == 7

    def test_last_one_is_today_only(self) -> None:
        """'last 1' selects only the reference date."""
        assert parse_window_args(["last", "1"], TODAY) == DateWindow(start=TODAY, end=TODAY)

    def test_window_argument_error_is_an_analytics_error(self) -> None:
        """Callers can catch the argument error with the package base error."""
        assert issubclass(WindowArgumentError, AnalyticsError)

    def test_no_arguments_is_an_error(self) -> None:
        """A window is required; nothing is not a window."""
        with pytest.raises(WindowArgumentError, match="No date window"):
            parse_window_args([], TODAY)

    @pytest.mark.parametrize("args", [["since"], ["last"], ["2026-08-01"], ["since", "2026-08-01", "extra"], ["a", "b", "c"]])
    def test_wrong_argument_count_is_an_error(self, args: list[str]) -> None:
        """Every form takes exactly two words."""
        with pytest.raises(WindowArgumentError, match="two"):
            parse_window_args(args, TODAY)

    @pytest.mark.parametrize("bad_date", ["notadate", "2026-13-01", "01.08.2026"])
    def test_invalid_since_date_is_an_error(self, bad_date: str) -> None:
        """Dates must be YYYY-MM-DD."""
        with pytest.raises(WindowArgumentError, match="YYYY-MM-DD"):
            parse_window_args(["since", bad_date], TODAY)

    def test_invalid_range_date_names_the_bad_value(self) -> None:
        """A bad date in the two-date form is reported with its value."""
        with pytest.raises(WindowArgumentError, match="2026-08-99"):
            parse_window_args(["2026-08-01", "2026-08-99"], TODAY)

    def test_unknown_keyword_is_an_error(self) -> None:
        """A first word that is neither a keyword nor a date is rejected."""
        with pytest.raises(WindowArgumentError, match="YYYY-MM-DD"):
            parse_window_args(["until", "2026-08-01"], TODAY)

    @pytest.mark.parametrize("bad_count", ["0", "-3", "seven", "1.5"])
    def test_last_requires_a_positive_integer(self, bad_count: str) -> None:
        """'last' takes a whole number of days greater than zero."""
        with pytest.raises(WindowArgumentError, match="positive"):
            parse_window_args(["last", bad_count], TODAY)

    def test_range_start_after_end_is_an_error(self) -> None:
        """The first date must not be after the second."""
        with pytest.raises(WindowArgumentError, match="after"):
            parse_window_args(["2026-08-31", "2026-08-01"], TODAY)

    def test_since_a_future_date_is_an_error(self) -> None:
        """'since' a date after today has no days in it."""
        with pytest.raises(WindowArgumentError, match="after"):
            parse_window_args(["since", "2026-09-05"], TODAY)


class TestSelectRecordsInWindow:
    """Choosing the records whose publish date falls inside the window."""

    def test_records_on_both_bounds_are_included(self) -> None:
        """The first and last day of the window are inside it."""
        window = DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 31))
        first = make_record("A", at(date(2026, 8, 1), 0, 0), video_id="first")
        last = make_record("A", at(date(2026, 8, 31), 23, 59), video_id="last")
        assert select_records_in_window([first, last], window) == [first, last]

    def test_records_just_outside_both_bounds_are_excluded(self) -> None:
        """The day before the start and the day after the end are outside."""
        window = DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 31))
        before = make_record("A", at(date(2026, 7, 31), 23, 59), video_id="before")
        after = make_record("A", at(date(2026, 9, 1), 0, 0), video_id="after")
        assert select_records_in_window([before, after], window) == []

    def test_selection_keeps_input_order(self) -> None:
        """Selection filters; it does not sort."""
        window = DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 31))
        newer = make_record("A", at(date(2026, 8, 20), 0, 0), video_id="newer")
        older = make_record("A", at(date(2026, 8, 10), 0, 0), video_id="older")
        assert select_records_in_window([newer, older], window) == [newer, older]


class TestGroupByChannelChronological:
    """Grouping selected records by channel with chronological order inside."""

    def test_channels_are_ordered_by_name(self) -> None:
        """Channel groups appear alphabetically."""
        records = [
            make_record("Zeta", at(date(2026, 8, 1), 0, 0)),
            make_record("Alpha", at(date(2026, 8, 2), 0, 0)),
            make_record("Mid", at(date(2026, 8, 3), 0, 0)),
        ]
        assert [group.channel for group in group_by_channel_chronological(records)] == ["Alpha", "Mid", "Zeta"]

    def test_videos_inside_a_channel_are_oldest_first(self) -> None:
        """Records within a channel are sorted by publish instant."""
        records = [
            make_record("A", at(date(2026, 8, 3), 9, 0), video_id="third"),
            make_record("A", at(date(2026, 8, 1), 9, 0), video_id="first"),
            make_record("A", at(date(2026, 8, 2), 9, 0), video_id="second"),
        ]
        groups = group_by_channel_chronological(records)
        assert [record.video_id for record in groups[0].records] == ["first", "second", "third"]

    def test_same_instant_is_ordered_by_video_id(self) -> None:
        """Ties on the publish instant are broken deterministically."""
        records = [
            make_record("A", at(date(2026, 8, 1), 9, 0), video_id="bbb"),
            make_record("A", at(date(2026, 8, 1), 9, 0), video_id="aaa"),
        ]
        groups = group_by_channel_chronological(records)
        assert [record.video_id for record in groups[0].records] == ["aaa", "bbb"]

    def test_records_are_split_by_channel(self) -> None:
        """Each channel group holds only its own records."""
        records = [
            make_record("A", at(date(2026, 8, 1), 0, 0), video_id="a1"),
            make_record("B", at(date(2026, 8, 1), 0, 0), video_id="b1"),
            make_record("A", at(date(2026, 8, 2), 0, 0), video_id="a2"),
        ]
        groups = group_by_channel_chronological(records)
        assert groups == [
            ChannelListing(channel="A", records=[records[0], records[2]]),
            ChannelListing(channel="B", records=[records[1]]),
        ]

    def test_no_records_gives_no_groups(self) -> None:
        """An empty selection has no channel groups."""
        assert group_by_channel_chronological([]) == []


class TestRenderVideoListing:
    """Rendering the grouped listing as terminal text."""

    WINDOW = DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 31))

    def test_header_states_the_inclusive_window(self) -> None:
        """The first line names both bounds and the day count."""
        rendered = render_video_listing([], self.WINDOW)
        assert rendered.splitlines()[0] == "Window: 2026-08-01 to 2026-08-31 (31 days, inclusive)"

    def test_empty_listing_says_so(self) -> None:
        """No matching videos prints a message instead of an empty table."""
        rendered = render_video_listing([], self.WINDOW)
        assert "No videos published in this window." in rendered
        assert "Videos:" not in rendered

    def test_channel_heading_shows_name_and_count(self) -> None:
        """Each group starts with the channel name and its video count."""
        group = ChannelListing(
            channel="Anthropic",
            records=[
                make_record("Anthropic", at(date(2026, 8, 3), 14, 5), video_id="v1"),
                make_record("Anthropic", at(date(2026, 8, 20), 9, 0), video_id="v2"),
            ],
        )
        assert "Anthropic (2 videos)" in render_video_listing([group], self.WINDOW)

    def test_single_video_channel_uses_singular(self) -> None:
        """A channel with one video reads '1 video'."""
        group = ChannelListing(channel="Solo", records=[make_record("Solo", at(date(2026, 8, 3), 14, 5))])
        assert "Solo (1 video)" in render_video_listing([group], self.WINDOW)

    def test_video_line_shows_date_time_title_and_id(self) -> None:
        """Each video line carries the UTC publish instant, the title, and the ID."""
        group = ChannelListing(
            channel="Anthropic",
            records=[make_record("Anthropic", at(date(2026, 8, 3), 14, 5), title="Building RAG Systems", video_id="abc123XYZ09")],
        )
        assert "  2026-08-03 14:05  Building RAG Systems  [abc123XYZ09]" in render_video_listing([group], self.WINDOW).splitlines()

    def test_groups_render_in_given_order_with_blank_line_between(self) -> None:
        """Channel groups keep their order and are separated by one empty line."""
        groups = [
            ChannelListing(channel="Alpha", records=[make_record("Alpha", at(date(2026, 8, 3), 0, 0))]),
            ChannelListing(channel="Beta", records=[make_record("Beta", at(date(2026, 8, 4), 0, 0))]),
        ]
        lines = render_video_listing(groups, self.WINDOW).splitlines()
        alpha_index = lines.index("Alpha (1 video)")
        beta_index = lines.index("Beta (1 video)")
        assert alpha_index < beta_index
        assert lines[beta_index - 1] == ""

    def test_footer_totals_videos_and_channels(self) -> None:
        """The last line sums videos across all channel groups."""
        groups = [
            ChannelListing(
                channel="Alpha",
                records=[
                    make_record("Alpha", at(date(2026, 8, 3), 0, 0), video_id="a1"),
                    make_record("Alpha", at(date(2026, 8, 4), 0, 0), video_id="a2"),
                ],
            ),
            ChannelListing(channel="Beta", records=[make_record("Beta", at(date(2026, 8, 4), 0, 0))]),
        ]
        assert render_video_listing(groups, self.WINDOW).splitlines()[-1] == "Videos: 3 across 2 channel(s)"


class TestRecordsFromRows:
    """Turning database rows into publication records."""

    @staticmethod
    def row(**overrides: Any) -> dict[str, Any]:
        """A complete database row for one video."""
        fields: dict[str, Any] = {
            "file_channel": "Anthropic",
            "file_path": "Anthropic/video/v.info.json",
            "id": "abc123XYZ09",
            "channel_id": "UCtestchannelid0000000",
            "title": "Building RAG Systems",
            "published_at": "2026-08-03T14:05:00+00:00",
            "duration": 601,
        }
        fields.update(overrides)
        return fields

    def test_complete_row_becomes_a_record(self) -> None:
        """Every record field is taken from the row."""
        records, failures = records_from_rows([self.row()])
        assert failures == []
        assert records == [
            PublicationRecord(
                channel="Anthropic",
                channel_id="UCtestchannelid0000000",
                published_at=datetime(2026, 8, 3, 14, 5, tzinfo=UTC),
                title="Building RAG Systems",
                duration_minutes=11,
                video_id="abc123XYZ09",
            )
        ]

    @pytest.mark.parametrize("field", ["id", "channel_id", "title", "duration"])
    def test_row_missing_a_required_field_is_a_failure(self, field: str) -> None:
        """A NULL required field is reported with the file path and field name."""
        records, failures = records_from_rows([self.row(**{field: None})])
        assert records == []
        assert len(failures) == 1
        assert field in failures[0]
        assert "Anthropic/video/v.info.json" in failures[0]

    def test_failures_do_not_stop_other_rows(self) -> None:
        """One bad row is reported and the good rows are still returned."""
        records, failures = records_from_rows([self.row(title=None), self.row(id="second")])
        assert [record.video_id for record in records] == ["second"]
        assert len(failures) == 1
