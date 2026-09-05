"""List videos published inside an inclusive calendar-date window, grouped by channel.

Builds on the publication records derived from yt-dlp metadata: the publish
instant is the metadata ``timestamp`` field in UTC, and a video belongs to the
window when its UTC calendar date lies between the window bounds, both ends
included. Channels are listed alphabetically; inside a channel the videos are
listed oldest first. The records come from the SQLite metadata database rows
selected for the window, so no metadata file is read at listing time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.analytics.errors import MetadataError, WindowArgumentError
from src.analytics.publication_timeseries import SECONDS_PER_MINUTE, PublicationRecord

DATE_FORMAT_HINT: str = "YYYY-MM-DD"
SINCE_KEYWORD: str = "since"
LAST_KEYWORD: str = "last"
WINDOW_WORD_COUNT: int = 2
USAGE: str = "usage: since <YYYY-MM-DD> | <YYYY-MM-DD> <YYYY-MM-DD> | last <days>"
VIDEO_TIME_FORMAT: str = "%Y-%m-%d %H:%M"


class DateWindow(BaseModel):
    """An inclusive span of calendar dates."""

    start: date = Field(..., description="First calendar date inside the window (inclusive)")
    end: date = Field(..., description="Last calendar date inside the window (inclusive)")

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _end_not_before_start(self) -> DateWindow:
        if self.start > self.end:
            raise ValueError(f"Window start {self.start.isoformat()} is after its end {self.end.isoformat()}")
        return self

    @property
    def day_count(self) -> int:
        """Number of calendar dates inside the window, both ends counted."""
        return (self.end - self.start).days + 1


class ChannelListing(BaseModel):
    """One channel's videos inside the window, oldest first."""

    channel: str = Field(..., min_length=1, description="Sanitized channel directory name")
    records: list[PublicationRecord] = Field(..., description="Videos published inside the window, oldest first")

    model_config = ConfigDict(frozen=True, extra="forbid")


def parse_date(value: str) -> date:
    """Parse a YYYY-MM-DD word, or raise WindowArgumentError naming the value."""
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise WindowArgumentError(f"Not a {DATE_FORMAT_HINT} date: '{value}'") from e


def parse_day_count(value: str) -> int:
    """Parse the day count after 'last', or raise WindowArgumentError naming the value."""
    try:
        count = int(value)
    except ValueError as e:
        raise WindowArgumentError(f"'{LAST_KEYWORD}' needs a positive whole number of days, got '{value}'") from e
    if count <= 0:
        raise WindowArgumentError(f"'{LAST_KEYWORD}' needs a positive whole number of days, got '{value}'")
    return count


def parse_window_args(args: list[str], today: date) -> DateWindow:
    """Turn the just-target words into a date window.

    Accepted forms, each exactly two words:
        since <date>     from that date up to today
        <date> <date>    from the first date up to the second date
        last <days>      the given number of calendar days ending today

    Args:
        args: The words as given on the command line.
        today: Reference date that closes the 'since' and 'last' forms.

    Returns:
        The inclusive window.

    Raises:
        WindowArgumentError: If the words are missing, malformed, or describe a
            window that ends before it starts.
    """
    if not args:
        raise WindowArgumentError("No date window given.")
    if len(args) != WINDOW_WORD_COUNT:
        raise WindowArgumentError(f"A date window takes exactly two words, got {len(args)}.")

    first, second = args
    if first == SINCE_KEYWORD:
        start = parse_date(second)
        end = today
    elif first == LAST_KEYWORD:
        start = today - timedelta(days=parse_day_count(second) - 1)
        end = today
    else:
        start = parse_date(first)
        end = parse_date(second)

    if start > end:
        raise WindowArgumentError(f"Window start {start.isoformat()} is after its end {end.isoformat()}")
    return DateWindow(start=start, end=end)


def required_string(row: Mapping[str, object], key: str, file_path: str) -> str:
    """Return a non-empty string column, or raise MetadataError naming the file."""
    value = row[key]
    if not isinstance(value, str) or not value:
        raise MetadataError(f"Metadata '{key}' is missing or not a non-empty string: {file_path}")
    return value


def required_number(row: Mapping[str, object], key: str, file_path: str) -> float:
    """Return a numeric column as a float, or raise MetadataError naming the file."""
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MetadataError(f"Metadata '{key}' is missing or not a number: {file_path}")
    return float(value)


def record_from_row(row: Mapping[str, object]) -> PublicationRecord:
    """Build a publication record from one metadata database row.

    Raises:
        MetadataError: If a field the record needs is NULL or has the wrong type.
    """
    file_path = str(row["file_path"])
    duration_seconds = required_number(row, "duration", file_path)
    if duration_seconds < 0:
        raise MetadataError(f"Metadata 'duration' is negative: {duration_seconds} in {file_path}")
    return PublicationRecord(
        channel=required_string(row, "file_channel", file_path),
        channel_id=required_string(row, "channel_id", file_path),
        published_at=datetime.fromisoformat(required_string(row, "published_at", file_path)),
        title=required_string(row, "title", file_path),
        duration_minutes=math.ceil(duration_seconds / SECONDS_PER_MINUTE),
        video_id=required_string(row, "id", file_path),
    )


def records_from_rows(rows: Iterable[Mapping[str, object]]) -> tuple[list[PublicationRecord], list[str]]:
    """Convert database rows into records; rows that cannot become one are reported, not dropped."""
    records: list[PublicationRecord] = []
    failures: list[str] = []
    for row in rows:
        try:
            records.append(record_from_row(row))
        except MetadataError as e:
            failures.append(str(e))
    return records, failures


def select_records_in_window(records: list[PublicationRecord], window: DateWindow) -> list[PublicationRecord]:
    """Keep the records whose UTC publish date lies inside the window, in input order."""
    return [record for record in records if window.start <= record.published_at.date() <= window.end]


def group_by_channel_chronological(records: list[PublicationRecord]) -> list[ChannelListing]:
    """Group records by channel, channels alphabetically and videos oldest first."""
    grouped: dict[str, list[PublicationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.channel].append(record)
    return [
        ChannelListing(channel=channel, records=sorted(grouped[channel], key=lambda record: (record.published_at, record.video_id)))
        for channel in sorted(grouped)
    ]


def video_count_label(count: int) -> str:
    """Render a video count with the right plural."""
    if count == 1:
        return "1 video"
    return f"{count} videos"


def render_video_line(record: PublicationRecord) -> str:
    """Render one video as its UTC publish instant, title, and ID."""
    return f"  {record.published_at.strftime(VIDEO_TIME_FORMAT)}  {record.title}  [{record.video_id}]"


def render_video_listing(groups: list[ChannelListing], window: DateWindow) -> str:
    """Render the grouped listing as terminal text.

    Args:
        groups: Channel groups in the order they should appear.
        window: The window the groups were selected from.

    Returns:
        The header, one block per channel, and a totals footer; or a message
        when no group has any video.
    """
    lines: list[str] = [f"Window: {window.start.isoformat()} to {window.end.isoformat()} ({window.day_count} days, inclusive)", ""]
    if not groups:
        lines.append("No videos published in this window.")
        return "\n".join(lines)

    for group in groups:
        lines.append(f"{group.channel} ({video_count_label(len(group.records))})")
        lines.extend(render_video_line(record) for record in group.records)
        lines.append("")

    total = sum(len(group.records) for group in groups)
    lines.append(f"Videos: {total} across {len(groups)} channel(s)")
    return "\n".join(lines)
