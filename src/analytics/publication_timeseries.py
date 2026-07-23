"""Publication cadence timeseries derived from yt-dlp video metadata.

Reads the per-channel ``.info.json`` files the download pipeline files under the
metadata directory and turns each one into a single publication event. This is
independent of the transcript/summary corpus built by ``index_builder``: a video
counts here as soon as its metadata exists, whether or not it was ever
transcribed or summarized.

The publication instant comes from the metadata ``timestamp`` field (epoch
seconds), which carries a time of day; ``upload_date`` is deliberately not used
as a substitute because it is date-only and would flatten every video of a day
onto the same instant.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.analytics.errors import AnalyticsError, MetadataError
from src.util.channel_name import sanitize_channel_name

DAYS_PER_WEEK: float = 7.0
# Mean Gregorian calendar month (365.2425 / 12), so monthly rates stay comparable
# across channels whose windows cover different months.
DAYS_PER_MONTH: float = 30.436875
SECONDS_PER_MINUTE: int = 60
CSV_HEADER: tuple[str, ...] = ("channelid", "datetime", "videotitle", "videoduration_minutes", "videoid")


class PublicationRecord(BaseModel):
    """One published video, derived from its yt-dlp ``.info.json`` metadata."""

    channel: str = Field(..., min_length=1, description="Sanitized channel directory name the metadata was found under")
    channel_id: str = Field(..., min_length=1, description="YouTube channel ID from the video metadata")
    published_at: datetime = Field(..., description="Publication instant in UTC from the metadata 'timestamp' field")
    title: str = Field(..., min_length=1, description="Video title")
    duration_minutes: int = Field(..., ge=0, description="Video duration in whole minutes, rounded up")
    video_id: str = Field(..., min_length=1, description="YouTube video ID")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ChannelFrequency(BaseModel):
    """Mean publication frequency for one channel over its observed window."""

    channel: str = Field(..., min_length=1, description="Sanitized channel directory name")
    channel_id: str = Field(..., min_length=1, description="YouTube channel ID of the channel's latest video")
    video_count: int = Field(..., gt=0, description="Published videos observed for this channel")
    first_published: datetime = Field(..., description="Earliest observed publication instant")
    last_published: datetime = Field(..., description="Latest observed publication instant")
    span_days: int = Field(..., gt=0, description="Inclusive day span from first to last publication")
    per_day: float = Field(..., ge=0.0, description="Mean videos published per day across the span")
    per_week: float = Field(..., ge=0.0, description="Mean videos published per week across the span")
    per_month: float = Field(..., ge=0.0, description="Mean videos published per month across the span")

    model_config = ConfigDict(frozen=True, extra="forbid")


def _require_str(fields: dict[str, object], key: str, path: Path) -> str:
    """Return a non-empty string field, or raise MetadataError naming the file."""
    value = fields.get(key)
    if not isinstance(value, str) or not value:
        raise MetadataError(f"Metadata '{key}' is missing or not a non-empty string: {path}")
    return value


def _require_number(fields: dict[str, object], key: str, path: Path) -> float:
    """Return a numeric field as a float, or raise MetadataError naming the file."""
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MetadataError(f"Metadata '{key}' is missing or not a number: {path}")
    return float(value)


def read_publication_record(path: Path, channel: str) -> PublicationRecord:
    """Parse one ``.info.json`` file into a publication record.

    Args:
        path: Path to the yt-dlp video metadata file.
        channel: Sanitized channel directory name the file was found under.

    Returns:
        The publication record for that video.

    Raises:
        MetadataError: If the file is unreadable, is not a JSON object, or lacks
            any of the fields the timeseries needs.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise MetadataError(f"Metadata file unreadable or invalid JSON: {path}: {e}") from e

    if not isinstance(payload, dict):
        raise MetadataError(f"Metadata file is not a JSON object: {path}")
    fields = cast(dict[str, object], payload)

    timestamp = _require_number(fields, "timestamp", path)
    try:
        published_at = datetime.fromtimestamp(timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError) as e:
        raise MetadataError(f"Metadata 'timestamp' is not a representable instant: {timestamp} in {path}") from e

    duration_seconds = _require_number(fields, "duration", path)
    if duration_seconds < 0:
        raise MetadataError(f"Metadata 'duration' is negative: {duration_seconds} in {path}")

    return PublicationRecord(
        channel=channel,
        channel_id=_require_str(fields, "channel_id", path),
        published_at=published_at,
        title=_require_str(fields, "title", path),
        duration_minutes=math.ceil(duration_seconds / SECONDS_PER_MINUTE),
        video_id=_require_str(fields, "id", path),
    )


def resolve_channel_dirs(metadata_dir: Path, video_subdir: str, channel_filter: str | None) -> list[tuple[str, Path]]:
    """List the channel metadata directories to scan, in stable order.

    Args:
        metadata_dir: Root metadata directory holding one subdirectory per channel.
        video_subdir: Name of the per-channel subdirectory holding video metadata.
        channel_filter: Channel to restrict to, matched against the raw config
            name or its sanitized directory form; None scans every channel.

    Returns:
        Sorted ``(channel_name, video_metadata_dir)`` pairs for channels that
        actually have a video metadata subdirectory.

    Raises:
        MetadataError: If the metadata root is absent, or a filter matches nothing.
    """
    if not metadata_dir.is_dir():
        raise MetadataError(f"Metadata directory not found: {metadata_dir}")

    wanted: str | None = None
    if channel_filter is not None:
        wanted = sanitize_channel_name(channel_filter)

    channel_dirs: list[tuple[str, Path]] = []
    for channel_dir in sorted(metadata_dir.iterdir()):
        if not channel_dir.is_dir() or channel_dir.name.startswith("."):
            continue
        if wanted is not None and channel_dir.name != wanted:
            continue
        video_dir = channel_dir / video_subdir
        if video_dir.is_dir():
            channel_dirs.append((channel_dir.name, video_dir))

    if wanted is not None and not channel_dirs:
        raise MetadataError(f"No video metadata directory for channel '{channel_filter}' under {metadata_dir}")
    return channel_dirs


def count_channel_metadata_files(video_dir: Path) -> int:
    """Count one channel's video metadata files, ignoring AppleDouble sidecars.

    This is the per-channel change signal: it counts files on disk without
    reading them, so it stays cheap and stable even when some files never index.
    """
    return sum(1 for path in video_dir.glob("*.info.json") if not path.name.startswith("._"))


def read_channel_records(channel: str, video_dir: Path, progress: Callable[[str], None]) -> tuple[list[PublicationRecord], list[str]]:
    """Read one channel's video metadata into records plus failure messages.

    Individual unreadable files never abort the scan: each failure is collected
    and returned. The channel is announced before its files are read so progress
    stays visible while the slow work runs.
    """
    metadata_files = sorted(path for path in video_dir.glob("*.info.json") if not path.name.startswith("._"))
    if not metadata_files:
        progress(f"Skipping: {channel} (no video metadata)")
        return [], []
    progress(f"Processing: {channel} ({len(metadata_files)} metadata files)")
    records: list[PublicationRecord] = []
    failures: list[str] = []
    for path in metadata_files:
        try:
            records.append(read_publication_record(path, channel))
        except MetadataError as e:
            failures.append(str(e))
    return records, failures


def collect_publication_records(
    metadata_dir: Path,
    video_subdir: str,
    channel_filter: str | None,
    progress: Callable[[str], None],
) -> tuple[list[PublicationRecord], list[str]]:
    """Read every video metadata file into publication records.

    Individual unreadable files never abort the scan: each failure is collected
    and returned so the caller can report them all and exit non-zero.

    Args:
        metadata_dir: Root metadata directory holding one subdirectory per channel.
        video_subdir: Name of the per-channel subdirectory holding video metadata.
        channel_filter: Channel to restrict to; None scans every channel.
        progress: Sink for per-channel progress lines.

    Returns:
        ``(records, failures)`` with records ordered by channel then publication
        instant, and failures as human-readable messages.

    Raises:
        MetadataError: If the metadata root is absent, or a filter matches nothing.
    """
    channel_dirs = resolve_channel_dirs(metadata_dir, video_subdir, channel_filter)
    records: list[PublicationRecord] = []
    failures: list[str] = []

    for channel, video_dir in channel_dirs:
        channel_records, channel_failures = read_channel_records(channel, video_dir, progress)
        records.extend(channel_records)
        failures.extend(channel_failures)

    records.sort(key=lambda record: (record.channel, record.published_at, record.video_id))
    return records, failures


def group_records_by_channel(records: list[PublicationRecord]) -> dict[str, list[PublicationRecord]]:
    """Group records by their channel directory name, preserving input order."""
    grouped: dict[str, list[PublicationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.channel].append(record)
    return dict(grouped)


def read_recorded_file_counts(path: Path) -> dict[str, int]:
    """Return the per-channel metadata file counts recorded by the last build.

    Returns an empty mapping when no usable record exists — the sidecar is
    absent, unreadable, or not a string-to-int object — so every channel is
    treated as changed and rescanned rather than trusting an unknown state.
    """
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    fields = cast(dict[str, object], payload)
    return {channel: count for channel, count in fields.items() if isinstance(count, int) and not isinstance(count, bool)}


def build_channel_id_to_name(metadata_dir: Path, video_subdir: str, channel_filter: str | None) -> dict[str, str]:
    """Map each channel's YouTube ID to its sanitized directory name.

    The timeseries CSV stores channel IDs but not names, so reusing it for the
    stats table needs this lookup. One readable metadata file per channel
    directory is enough to learn that channel's ID, so the whole tree is not
    reread.

    Raises:
        MetadataError: If the metadata root is absent, or a filter matches nothing.
    """
    mapping: dict[str, str] = {}
    for channel, video_dir in resolve_channel_dirs(metadata_dir, video_subdir, channel_filter):
        for path in sorted(video_dir.glob("*.info.json")):
            if path.name.startswith("._"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict):
                channel_id = cast(dict[str, object], payload).get("channel_id")
                if isinstance(channel_id, str) and channel_id:
                    mapping[channel_id] = channel
                    break
    return mapping


def build_incremental_records(
    channel_dirs: list[tuple[str, Path]],
    current_counts: dict[str, int],
    recorded_counts: dict[str, int],
    existing_by_channel: dict[str, list[PublicationRecord]],
    progress: Callable[[str], None],
) -> tuple[list[PublicationRecord], list[str], list[str]]:
    """Reuse unchanged channels' records and rescan only the changed ones.

    A channel is reused when its current metadata file count equals the recorded
    count and it has rows in the existing CSV; otherwise its metadata is reread.

    Returns:
        ``(records, failures, rescanned_channels)`` with records sorted by
        channel then publication instant, failures only from rescanned channels,
        and rescanned_channels naming the channels that were reread.
    """
    records: list[PublicationRecord] = []
    failures: list[str] = []
    rescanned: list[str] = []
    for channel, video_dir in channel_dirs:
        if recorded_counts.get(channel) == current_counts.get(channel) and channel in existing_by_channel:
            progress(f"Skipping: {channel} (unchanged, {current_counts.get(channel)} metadata files)")
            records.extend(existing_by_channel[channel])
            continue
        channel_records, channel_failures = read_channel_records(channel, video_dir, progress)
        records.extend(channel_records)
        failures.extend(channel_failures)
        rescanned.append(channel)
    records.sort(key=lambda record: (record.channel, record.published_at, record.video_id))
    return records, failures, rescanned


def compute_channel_frequencies(records: list[PublicationRecord]) -> list[ChannelFrequency]:
    """Aggregate records into per-channel mean publication frequencies.

    The observation window is the inclusive day span between a channel's first
    and last publication, so a channel that published once has a span of one day
    and a rate of one video per day. Weekly and monthly rates are the daily rate
    scaled by the mean week and Gregorian month length, which keeps channels
    comparable regardless of which months their windows happen to cover.

    Args:
        records: Publication records across any number of channels.

    Returns:
        One entry per channel, ordered by descending daily rate then channel name.
    """
    grouped: dict[str, list[PublicationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.channel].append(record)

    frequencies: list[ChannelFrequency] = []
    for channel, channel_records in grouped.items():
        ordered = sorted(channel_records, key=lambda record: record.published_at)
        first_published = ordered[0].published_at
        last_published = ordered[-1].published_at
        span_days = (last_published.date() - first_published.date()).days + 1
        per_day = len(ordered) / span_days
        frequencies.append(
            ChannelFrequency(
                channel=channel,
                # The channel dir is the grouping key; the newest video carries the
                # most current channel ID should a channel ever have been renamed.
                channel_id=ordered[-1].channel_id,
                video_count=len(ordered),
                first_published=first_published,
                last_published=last_published,
                span_days=span_days,
                per_day=per_day,
                per_week=per_day * DAYS_PER_WEEK,
                per_month=per_day * DAYS_PER_MONTH,
            )
        )

    frequencies.sort(key=lambda frequency: (-frequency.per_day, frequency.channel))
    return frequencies


def render_timeseries_csv(records: list[PublicationRecord]) -> str:
    """Render publication records as CSV text with a header row.

    Uses the csv module so titles containing commas, quotes, or newlines are
    quoted correctly rather than corrupting the column layout.

    Args:
        records: Publication records in the order they should appear.

    Returns:
        The full CSV document, newline-terminated.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    for record in records:
        writer.writerow(
            [
                record.channel_id,
                record.published_at.isoformat(),
                record.title,
                record.duration_minutes,
                record.video_id,
            ]
        )
    return buffer.getvalue()


def load_records_from_csv(csv_path: Path, id_to_name: dict[str, str]) -> list[PublicationRecord]:
    """Reconstruct publication records from a previously written timeseries CSV.

    Args:
        csv_path: The timeseries CSV to read back.
        id_to_name: Channel ID to directory name for labelling; an ID missing
            from the map is labelled by the ID itself.

    Returns:
        Records in the CSV's row order.

    Raises:
        AnalyticsError: If the CSV is unreadable, its header is unexpected, or a
            row cannot be parsed into a record.
    """
    try:
        text = csv_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise AnalyticsError(f"Existing timeseries CSV cannot be read: {csv_path}: {e}") from e

    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header != list(CSV_HEADER):
        raise AnalyticsError(f"Existing timeseries CSV has an unexpected header: {csv_path}: {header}")

    records: list[PublicationRecord] = []
    for row in reader:
        if len(row) != len(CSV_HEADER):
            raise AnalyticsError(f"Malformed row in timeseries CSV {csv_path}: {row}")
        channel_id, published, title, duration_minutes, video_id = row
        channel_name = id_to_name.get(channel_id)
        if channel_name is None:
            channel_name = channel_id
        try:
            record = PublicationRecord(
                channel=channel_name,
                channel_id=channel_id,
                published_at=datetime.fromisoformat(published),
                title=title,
                duration_minutes=int(duration_minutes),
                video_id=video_id,
            )
        except (ValueError, ValidationError) as e:
            raise AnalyticsError(f"Row in timeseries CSV cannot be parsed into a record: {csv_path}: {row}: {e}") from e
        records.append(record)
    return records


def render_channel_id_ranking(frequencies: list[ChannelFrequency]) -> str:
    """Render channel IDs one per line, ascending by publication frequency.

    Args:
        frequencies: Per-channel frequencies in any order.

    Returns:
        Newline-terminated text of channel IDs, least frequent publisher first.
    """
    ascending = sorted(frequencies, key=lambda frequency: (frequency.per_day, frequency.channel))
    return "".join(f"{frequency.channel_id}\n" for frequency in ascending)


def _two_row_header(columns: list[tuple[str, str, int]]) -> tuple[str, str]:
    """Build the two right-aligned header rows from (row1, row2, width) columns."""
    row1 = " ".join(f"{text:>{width}}" for text, _, width in columns)
    row2 = " ".join(f"{text:>{width}}" for _, text, width in columns)
    return row1, row2


def render_frequency_table(frequencies: list[ChannelFrequency], channel_width: int) -> str:
    """Render the per-channel frequency table, highest publication rate first.

    Args:
        frequencies: Per-channel frequencies, already ordered for display.
        channel_width: Column width for the channel label.

    Returns:
        The rendered table including header, rows, and a TOTAL line.
    """
    columns: list[tuple[str, str, int]] = [
        ("", "Videos", 7),
        ("First", "Published", 11),
        ("Last", "Published", 11),
        ("Span", "Days", 6),
        ("Per", "Day", 8),
        ("Per", "Week", 8),
        ("Per", "Month", 8),
    ]
    line_width = channel_width + 1 + sum(width + 1 for _, _, width in columns)
    header_row1, header_row2 = _two_row_header(columns)

    lines: list[str] = [
        "=" * line_width,
        "PUBLICATION FREQUENCY BY CHANNEL",
        "=" * line_width,
        "",
        f"{'':>{channel_width}} {header_row1}",
        f"{'Channel':<{channel_width}} {header_row2}",
        "-" * line_width,
    ]

    for frequency in frequencies:
        label = frequency.channel[:channel_width]
        lines.append(
            " ".join(
                [
                    f"{label:<{channel_width}}",
                    f"{frequency.video_count:>7}",
                    f"{frequency.first_published.date().isoformat():>11}",
                    f"{frequency.last_published.date().isoformat():>11}",
                    f"{frequency.span_days:>6}",
                    f"{frequency.per_day:>8.2f}",
                    f"{frequency.per_week:>8.2f}",
                    f"{frequency.per_month:>8.2f}",
                ]
            )
        )

    lines.append("-" * line_width)
    lines.append(_render_total_row(frequencies, channel_width))
    return "\n".join(lines) + "\n"


def _render_total_row(frequencies: list[ChannelFrequency], channel_width: int) -> str:
    """Render the TOTAL row: all videos over the union of every channel's window."""
    if not frequencies:
        return f"{'TOTAL':<{channel_width}} {0:>7}"

    total_videos = sum(frequency.video_count for frequency in frequencies)
    first_published = min(frequency.first_published for frequency in frequencies)
    last_published = max(frequency.last_published for frequency in frequencies)
    span_days = (last_published.date() - first_published.date()).days + 1
    per_day = total_videos / span_days
    return " ".join(
        [
            f"{'TOTAL':<{channel_width}}",
            f"{total_videos:>7}",
            f"{first_published.date().isoformat():>11}",
            f"{last_published.date().isoformat():>11}",
            f"{span_days:>6}",
            f"{per_day:>8.2f}",
            f"{per_day * DAYS_PER_WEEK:>8.2f}",
            f"{per_day * DAYS_PER_MONTH:>8.2f}",
        ]
    )
