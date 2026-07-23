"""Unit tests for the video publication timeseries and frequency report."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.analytics.errors import AnalyticsError, MetadataError
from src.analytics.publication_timeseries import (
    PublicationRecord,
    build_channel_id_to_name,
    build_incremental_records,
    collect_publication_records,
    compute_channel_frequencies,
    count_channel_metadata_files,
    group_records_by_channel,
    load_records_from_csv,
    read_channel_records,
    read_publication_record,
    read_recorded_file_counts,
    render_channel_id_ranking,
    render_frequency_table,
    render_timeseries_csv,
    resolve_channel_dirs,
)

# 2025-06-09T12:00:00+00:00, used as the base instant for synthetic records.
BASE_TIMESTAMP = 1749470400


def info_payload(**overrides: Any) -> dict[str, Any]:
    """Build a complete synthetic .info.json payload, with optional overrides."""
    payload: dict[str, Any] = {
        "id": "abc123XYZ09",
        "title": "Building RAG Systems",
        "channel_id": "UCtestchannelid0000000",
        "timestamp": BASE_TIMESTAMP,
        "duration": 600,
    }
    payload.update(overrides)
    return payload


def write_info_json(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write an .info.json payload into directory and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.info.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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


class TestReadPublicationRecord:
    """Parsing a single .info.json into a publication record."""

    def test_full_payload_parses(self, tmp_path: Path) -> None:
        """Every field is taken from the metadata, not inferred."""
        path = write_info_json(tmp_path, "v", info_payload())
        record = read_publication_record(path, "Anthropic")
        assert record.channel == "Anthropic"
        assert record.channel_id == "UCtestchannelid0000000"
        assert record.title == "Building RAG Systems"
        assert record.video_id == "abc123XYZ09"
        assert record.published_at == datetime(2025, 6, 9, 12, 0, tzinfo=UTC)

    def test_timestamp_preserves_time_of_day(self, tmp_path: Path) -> None:
        """The epoch timestamp carries a time, unlike the date-only upload_date."""
        path = write_info_json(tmp_path, "v", info_payload(timestamp=BASE_TIMESTAMP + 3661))
        record = read_publication_record(path, "C")
        assert record.published_at == datetime(2025, 6, 9, 13, 1, 1, tzinfo=UTC)

    def test_duration_rounds_up_to_whole_minutes(self, tmp_path: Path) -> None:
        """A partial minute counts as a full minute."""
        path = write_info_json(tmp_path, "v", info_payload(duration=601))
        assert read_publication_record(path, "C").duration_minutes == 11

    def test_exact_minute_does_not_round_up(self, tmp_path: Path) -> None:
        """An exact multiple of 60 seconds stays put."""
        path = write_info_json(tmp_path, "v", info_payload(duration=600))
        assert read_publication_record(path, "C").duration_minutes == 10

    def test_sub_minute_duration_becomes_one_minute(self, tmp_path: Path) -> None:
        """Any non-zero duration under a minute rounds up to one."""
        path = write_info_json(tmp_path, "v", info_payload(duration=1))
        assert read_publication_record(path, "C").duration_minutes == 1

    def test_float_duration_rounds_up(self, tmp_path: Path) -> None:
        """yt-dlp may emit a float duration."""
        path = write_info_json(tmp_path, "v", info_payload(duration=125.4))
        assert read_publication_record(path, "C").duration_minutes == 3

    @pytest.mark.parametrize("field", ["id", "title", "channel_id"])
    def test_missing_string_field_raises(self, tmp_path: Path, field: str) -> None:
        """A missing required string field aborts that file, naming the field."""
        payload = info_payload()
        del payload[field]
        path = write_info_json(tmp_path, "v", payload)
        with pytest.raises(MetadataError, match=field):
            read_publication_record(path, "C")

    @pytest.mark.parametrize("field", ["timestamp", "duration"])
    def test_missing_numeric_field_raises(self, tmp_path: Path, field: str) -> None:
        """A missing required numeric field aborts that file, naming the field."""
        payload = info_payload()
        del payload[field]
        path = write_info_json(tmp_path, "v", payload)
        with pytest.raises(MetadataError, match=field):
            read_publication_record(path, "C")

    def test_null_timestamp_raises(self, tmp_path: Path) -> None:
        """An explicit null timestamp is not silently replaced by upload_date."""
        path = write_info_json(tmp_path, "v", info_payload(timestamp=None, upload_date="20250609"))
        with pytest.raises(MetadataError, match="timestamp"):
            read_publication_record(path, "C")

    def test_empty_title_raises(self, tmp_path: Path) -> None:
        """An empty title is treated as missing."""
        path = write_info_json(tmp_path, "v", info_payload(title=""))
        with pytest.raises(MetadataError, match="title"):
            read_publication_record(path, "C")

    def test_negative_duration_raises(self, tmp_path: Path) -> None:
        """A negative duration is corrupt metadata, not a zero-length video."""
        path = write_info_json(tmp_path, "v", info_payload(duration=-5))
        with pytest.raises(MetadataError, match="negative"):
            read_publication_record(path, "C")

    def test_boolean_timestamp_raises(self, tmp_path: Path) -> None:
        """A bool must not slip through the numeric check as 0/1."""
        path = write_info_json(tmp_path, "v", info_payload(timestamp=True))
        with pytest.raises(MetadataError, match="timestamp"):
            read_publication_record(path, "C")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        """A truncated file names itself in the error."""
        path = tmp_path / "broken.info.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(MetadataError, match="broken.info.json"):
            read_publication_record(path, "C")

    def test_json_array_raises(self, tmp_path: Path) -> None:
        """Valid JSON that is not an object is rejected."""
        path = tmp_path / "list.info.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(MetadataError, match="not a JSON object"):
            read_publication_record(path, "C")


class TestResolveChannelDirs:
    """Selecting which channel metadata directories to scan."""

    def test_lists_all_channels_with_video_subdir(self, tmp_path: Path) -> None:
        """Every channel holding a video subdirectory is returned, sorted."""
        write_info_json(tmp_path / "Zeta" / "video", "v", info_payload())
        write_info_json(tmp_path / "Alpha" / "video", "v", info_payload())
        assert [name for name, _ in resolve_channel_dirs(tmp_path, "video", None)] == ["Alpha", "Zeta"]

    def test_channel_without_video_subdir_skipped(self, tmp_path: Path) -> None:
        """A channel with only audio metadata contributes nothing."""
        (tmp_path / "Alpha" / "audio").mkdir(parents=True)
        write_info_json(tmp_path / "Beta" / "video", "v", info_payload())
        assert [name for name, _ in resolve_channel_dirs(tmp_path, "video", None)] == ["Beta"]

    def test_filter_matches_sanitized_directory_name(self, tmp_path: Path) -> None:
        """The directory form of a channel name is accepted."""
        write_info_json(tmp_path / "AI_Engineer" / "video", "v", info_payload())
        write_info_json(tmp_path / "Anthropic" / "video", "v", info_payload())
        assert [name for name, _ in resolve_channel_dirs(tmp_path, "video", "AI_Engineer")] == ["AI_Engineer"]

    def test_filter_matches_raw_config_name(self, tmp_path: Path) -> None:
        """The raw config.yaml name is sanitized before matching."""
        write_info_json(tmp_path / "AI_Engineer" / "video", "v", info_payload())
        assert [name for name, _ in resolve_channel_dirs(tmp_path, "video", "AI Engineer")] == ["AI_Engineer"]

    def test_unknown_filter_raises(self, tmp_path: Path) -> None:
        """A filter matching nothing is an error, not an empty run."""
        write_info_json(tmp_path / "Alpha" / "video", "v", info_payload())
        with pytest.raises(MetadataError, match="Nope"):
            resolve_channel_dirs(tmp_path, "video", "Nope")

    def test_missing_metadata_root_raises(self, tmp_path: Path) -> None:
        """An absent metadata directory is an error."""
        with pytest.raises(MetadataError, match="Metadata directory not found"):
            resolve_channel_dirs(tmp_path / "absent", "video", None)


class TestCollectPublicationRecords:
    """Scanning the metadata tree into records plus failures."""

    def test_collects_across_channels(self, tmp_path: Path) -> None:
        """Records from every channel are returned together."""
        write_info_json(tmp_path / "Alpha" / "video", "a", info_payload(id="aaaaaaaaaaa"))
        write_info_json(tmp_path / "Beta" / "video", "b", info_payload(id="bbbbbbbbbbb"))
        records, failures = collect_publication_records(tmp_path, "video", None, lambda _: None)
        assert failures == []
        assert {record.channel for record in records} == {"Alpha", "Beta"}

    def test_bad_file_is_collected_not_raised(self, tmp_path: Path) -> None:
        """One corrupt file must not abort the scan of the rest."""
        video_dir = tmp_path / "Alpha" / "video"
        write_info_json(video_dir, "good", info_payload())
        (video_dir / "bad.info.json").write_text("{oops", encoding="utf-8")
        records, failures = collect_publication_records(tmp_path, "video", None, lambda _: None)
        assert len(records) == 1
        assert len(failures) == 1
        assert "bad.info.json" in failures[0]

    def test_macos_resource_forks_ignored(self, tmp_path: Path) -> None:
        """AppleDouble sidecars on the exFAT volume are not metadata."""
        video_dir = tmp_path / "Alpha" / "video"
        write_info_json(video_dir, "good", info_payload())
        (video_dir / "._good.info.json").write_text("\x00\x05", encoding="utf-8")
        records, failures = collect_publication_records(tmp_path, "video", None, lambda _: None)
        assert len(records) == 1
        assert failures == []

    def test_records_sorted_by_channel_then_time(self, tmp_path: Path) -> None:
        """Output order is a stable timeseries per channel."""
        video_dir = tmp_path / "Alpha" / "video"
        write_info_json(video_dir, "late", info_payload(id="lllllllllll", timestamp=BASE_TIMESTAMP + 100))
        write_info_json(video_dir, "early", info_payload(id="eeeeeeeeeee", timestamp=BASE_TIMESTAMP))
        records, _ = collect_publication_records(tmp_path, "video", None, lambda _: None)
        assert [record.video_id for record in records] == ["eeeeeeeeeee", "lllllllllll"]

    def test_progress_reports_each_channel(self, tmp_path: Path) -> None:
        """The operator sees which channel is being scanned."""
        write_info_json(tmp_path / "Alpha" / "video", "a", info_payload())
        lines: list[str] = []
        collect_publication_records(tmp_path, "video", None, lines.append)
        assert any("Alpha" in line for line in lines)

    def test_empty_channel_reports_single_skip_line(self, tmp_path: Path) -> None:
        """A channel with no metadata gets exactly one quiet skip line."""
        (tmp_path / "Alpha" / "video").mkdir(parents=True)
        lines: list[str] = []
        records, _ = collect_publication_records(tmp_path, "video", None, lines.append)
        assert records == []
        assert lines == ["Skipping: Alpha (no video metadata)"]


class TestComputeChannelFrequencies:
    """Turning records into per-channel mean publication rates."""

    def test_rate_over_inclusive_day_span(self) -> None:
        """Ten videos across an inclusive ten-day window is one per day.

        The expected rates are exactly representable floats (count/span is a
        clean integer division here), so exact equality is precise, not fragile.
        """
        records = [make_record("Alpha", datetime(2025, 1, day, tzinfo=UTC)) for day in range(1, 11)]
        frequency = compute_channel_frequencies(records)[0]
        assert frequency.video_count == 10
        assert frequency.span_days == 10
        assert frequency.per_day == 1.0

    def test_weekly_and_monthly_scale_from_daily(self) -> None:
        """Weekly and monthly rates are the daily rate scaled by mean lengths."""
        records = [make_record("Alpha", datetime(2025, 1, day, tzinfo=UTC)) for day in range(1, 11)]
        frequency = compute_channel_frequencies(records)[0]
        assert frequency.per_week == 7.0
        assert frequency.per_month == 30.436875

    def test_single_video_spans_one_day(self) -> None:
        """A lone video yields a one-day window rather than a divide by zero."""
        frequency = compute_channel_frequencies([make_record("Alpha", datetime(2025, 1, 1, tzinfo=UTC))])[0]
        assert frequency.span_days == 1
        assert frequency.per_day == 1.0

    def test_same_day_videos_span_one_day(self) -> None:
        """Several videos published on one day share a one-day window."""
        records = [
            make_record("Alpha", datetime(2025, 1, 1, 8, tzinfo=UTC)),
            make_record("Alpha", datetime(2025, 1, 1, 20, tzinfo=UTC)),
        ]
        frequency = compute_channel_frequencies(records)[0]
        assert frequency.span_days == 1
        assert frequency.per_day == 2.0

    def test_sorted_by_descending_daily_rate(self) -> None:
        """The busiest publisher comes first."""
        records = [
            make_record("Slow", datetime(2025, 1, 1, tzinfo=UTC)),
            make_record("Slow", datetime(2025, 1, 31, tzinfo=UTC)),
            *[make_record("Fast", datetime(2025, 1, day, tzinfo=UTC)) for day in range(1, 31)],
        ]
        assert [f.channel for f in compute_channel_frequencies(records)] == ["Fast", "Slow"]

    def test_first_and_last_published_bound_the_window(self) -> None:
        """The window is bounded by the actual extremes, not input order."""
        records = [
            make_record("Alpha", datetime(2025, 3, 5, tzinfo=UTC)),
            make_record("Alpha", datetime(2025, 1, 1, tzinfo=UTC)),
            make_record("Alpha", datetime(2025, 2, 2, tzinfo=UTC)),
        ]
        frequency = compute_channel_frequencies(records)[0]
        assert frequency.first_published == datetime(2025, 1, 1, tzinfo=UTC)
        assert frequency.last_published == datetime(2025, 3, 5, tzinfo=UTC)

    def test_channel_id_taken_from_newest_video(self) -> None:
        """A renamed channel reports its most current ID."""
        records = [
            make_record("Alpha", datetime(2025, 1, 1, tzinfo=UTC), channel_id="UCold"),
            make_record("Alpha", datetime(2025, 6, 1, tzinfo=UTC), channel_id="UCnew"),
        ]
        assert compute_channel_frequencies(records)[0].channel_id == "UCnew"

    def test_channels_are_grouped_independently(self) -> None:
        """Each channel gets its own window and count."""
        records = [
            make_record("Alpha", datetime(2025, 1, 1, tzinfo=UTC)),
            make_record("Beta", datetime(2025, 5, 1, tzinfo=UTC)),
        ]
        assert {f.channel: f.video_count for f in compute_channel_frequencies(records)} == {"Alpha": 1, "Beta": 1}

    def test_no_records_yields_no_frequencies(self) -> None:
        """An empty corpus produces no rows."""
        assert compute_channel_frequencies([]) == []


class TestRenderTimeseriesCsv:
    """The CSV artifact."""

    def test_header_matches_requested_columns(self) -> None:
        """The header names the five requested fields in order."""
        header = render_timeseries_csv([]).splitlines()[0]
        assert header == "channelid,datetime,videotitle,videoduration_minutes,videoid"

    def test_row_carries_every_field(self) -> None:
        """A record renders as its five values."""
        record = make_record("Alpha", datetime(2025, 6, 9, 12, 0, tzinfo=UTC), channel_id="UCabc", video_id="vid00000001")
        row = render_timeseries_csv([record]).splitlines()[1]
        assert row == "UCabc,2025-06-09T12:00:00+00:00,A Video,10,vid00000001"

    def test_title_with_comma_is_quoted(self) -> None:
        """A comma in a title must not shift the columns."""
        record = make_record("Alpha", datetime(2025, 6, 9, tzinfo=UTC), title="Rag, Explained")
        row = render_timeseries_csv([record]).splitlines()[1]
        assert '"Rag, Explained"' in row
        assert len(row.split(",")) == 6

    def test_title_with_quotes_is_escaped(self) -> None:
        """Embedded quotes are doubled per RFC 4180."""
        record = make_record("Alpha", datetime(2025, 6, 9, tzinfo=UTC), title='The "Best" Model')
        assert '"The ""Best"" Model"' in render_timeseries_csv([record]).splitlines()[1]

    def test_empty_records_still_write_header(self) -> None:
        """An empty result set is a valid CSV with just a header."""
        assert render_timeseries_csv([]) == "channelid,datetime,videotitle,videoduration_minutes,videoid\n"


class TestRenderChannelIdRanking:
    """The channel ID ranking text artifact."""

    def test_ascending_by_publication_frequency(self) -> None:
        """The least frequent publisher is listed first."""
        records = [
            make_record("Slow", datetime(2025, 1, 1, tzinfo=UTC), channel_id="UCslow"),
            make_record("Slow", datetime(2025, 1, 31, tzinfo=UTC), channel_id="UCslow"),
            *[make_record("Fast", datetime(2025, 1, day, tzinfo=UTC), channel_id="UCfast") for day in range(1, 31)],
        ]
        ranking = render_channel_id_ranking(compute_channel_frequencies(records))
        assert ranking == "UCslow\nUCfast\n"

    def test_contains_only_channel_ids(self) -> None:
        """No names, counts, or rates leak into the file."""
        records = [make_record("Alpha", datetime(2025, 1, 1, tzinfo=UTC), channel_id="UCabc")]
        assert render_channel_id_ranking(compute_channel_frequencies(records)) == "UCabc\n"

    def test_empty_input_renders_empty_file(self) -> None:
        """No channels means an empty file, not a stray newline."""
        assert render_channel_id_ranking([]) == ""


class TestRenderFrequencyTable:
    """The terminal frequency report."""

    def test_lists_channels_and_totals(self) -> None:
        """Each channel appears with its counts, plus a TOTAL row."""
        records = [make_record("Alpha", datetime(2025, 1, day, tzinfo=UTC)) for day in range(1, 11)]
        table = render_frequency_table(compute_channel_frequencies(records), 20)
        assert "PUBLICATION FREQUENCY BY CHANNEL" in table
        assert "Alpha" in table
        assert "TOTAL" in table

    def test_total_spans_the_union_of_all_windows(self) -> None:
        """The TOTAL row counts every video across the combined window."""
        records = [
            make_record("Alpha", datetime(2025, 1, 1, tzinfo=UTC)),
            make_record("Beta", datetime(2025, 1, 10, tzinfo=UTC)),
        ]
        table = render_frequency_table(compute_channel_frequencies(records), 20)
        total_line = next(line for line in table.splitlines() if line.startswith("TOTAL"))
        assert "2025-01-01" in total_line
        assert "2025-01-10" in total_line

    def test_rows_appear_in_descending_rate_order(self) -> None:
        """The table preserves the busiest-first ordering."""
        records = [
            make_record("Slow", datetime(2025, 1, 1, tzinfo=UTC)),
            make_record("Slow", datetime(2025, 1, 31, tzinfo=UTC)),
            *[make_record("Fast", datetime(2025, 1, day, tzinfo=UTC)) for day in range(1, 31)],
        ]
        table = render_frequency_table(compute_channel_frequencies(records), 20)
        assert table.index("\nFast") < table.index("\nSlow")

    def test_long_channel_name_truncated_to_column(self) -> None:
        """An over-wide name is clipped so columns stay aligned."""
        records = [make_record("A" * 40, datetime(2025, 1, 1, tzinfo=UTC))]
        table = render_frequency_table(compute_channel_frequencies(records), 10)
        assert "A" * 11 not in table

    def test_empty_frequencies_render_zero_total(self) -> None:
        """With no channels the table still renders a TOTAL row."""
        assert "TOTAL" in render_frequency_table([], 20)


class TestCountChannelMetadataFiles:
    """Counting one channel's metadata files as a cheap change signal."""

    def test_counts_info_json_files(self, tmp_path: Path) -> None:
        """Every video metadata file in the channel is counted."""
        video_dir = tmp_path / "Alpha" / "video"
        write_info_json(video_dir, "a", info_payload())
        write_info_json(video_dir, "b", info_payload())
        assert count_channel_metadata_files(video_dir) == 2

    def test_ignores_appledouble_sidecars(self, tmp_path: Path) -> None:
        """AppleDouble sidecars are not metadata and are not counted."""
        video_dir = tmp_path / "Alpha" / "video"
        write_info_json(video_dir, "good", info_payload())
        (video_dir / "._good.info.json").write_text("\x00", encoding="utf-8")
        assert count_channel_metadata_files(video_dir) == 1

    def test_counts_unreadable_files_too(self, tmp_path: Path) -> None:
        """The count is a change signal, independent of whether a file parses."""
        video_dir = tmp_path / "Alpha" / "video"
        write_info_json(video_dir, "good", info_payload())
        (video_dir / "bad.info.json").write_text("{oops", encoding="utf-8")
        assert count_channel_metadata_files(video_dir) == 2


class TestReadChannelRecords:
    """Reading a single channel's metadata into records plus failures."""

    def test_reads_records_and_announces(self, tmp_path: Path) -> None:
        """The channel is announced and its files parsed into records."""
        video_dir = tmp_path / "Alpha" / "video"
        write_info_json(video_dir, "a", info_payload())
        lines: list[str] = []
        records, failures = read_channel_records("Alpha", video_dir, lines.append)
        assert len(records) == 1
        assert failures == []
        assert any("Processing: Alpha" in line for line in lines)

    def test_empty_dir_reports_single_skip(self, tmp_path: Path) -> None:
        """A channel with no metadata gets exactly one quiet skip line."""
        video_dir = tmp_path / "Alpha" / "video"
        video_dir.mkdir(parents=True)
        lines: list[str] = []
        records, failures = read_channel_records("Alpha", video_dir, lines.append)
        assert records == []
        assert failures == []
        assert lines == ["Skipping: Alpha (no video metadata)"]

    def test_bad_file_collected_not_raised(self, tmp_path: Path) -> None:
        """One corrupt file is collected, not raised, so the rest still read."""
        video_dir = tmp_path / "Alpha" / "video"
        write_info_json(video_dir, "good", info_payload())
        (video_dir / "bad.info.json").write_text("{oops", encoding="utf-8")
        records, failures = read_channel_records("Alpha", video_dir, lambda _: None)
        assert len(records) == 1
        assert len(failures) == 1


class TestGroupRecordsByChannel:
    """Grouping records by their channel directory name."""

    def test_groups_by_channel_name(self) -> None:
        """Records land under their channel key."""
        records = [
            make_record("Alpha", datetime(2025, 1, 1, tzinfo=UTC)),
            make_record("Beta", datetime(2025, 1, 2, tzinfo=UTC)),
            make_record("Alpha", datetime(2025, 1, 3, tzinfo=UTC)),
        ]
        grouped = group_records_by_channel(records)
        assert set(grouped) == {"Alpha", "Beta"}
        assert len(grouped["Alpha"]) == 2

    def test_empty_input(self) -> None:
        """No records means no groups."""
        assert group_records_by_channel([]) == {}


class TestReadRecordedFileCounts:
    """Reading the per-channel file-count sidecar."""

    def test_absent_file_is_empty(self, tmp_path: Path) -> None:
        """A missing sidecar forces every channel to rescan."""
        assert read_recorded_file_counts(tmp_path / "missing.json") == {}

    def test_reads_mapping(self, tmp_path: Path) -> None:
        """A well-formed sidecar returns its channel counts."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"Alpha": 10, "Beta": 3}), encoding="utf-8")
        assert read_recorded_file_counts(path) == {"Alpha": 10, "Beta": 3}

    def test_invalid_json_is_empty(self, tmp_path: Path) -> None:
        """A corrupt sidecar is not trusted."""
        path = tmp_path / "c.json"
        path.write_text("{oops", encoding="utf-8")
        assert read_recorded_file_counts(path) == {}

    def test_non_object_is_empty(self, tmp_path: Path) -> None:
        """Valid JSON that is not an object yields nothing."""
        path = tmp_path / "c.json"
        path.write_text("[1, 2]", encoding="utf-8")
        assert read_recorded_file_counts(path) == {}

    def test_non_integer_values_dropped(self, tmp_path: Path) -> None:
        """Only integer counts survive; strings and bools are ignored."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"Alpha": 10, "Beta": "x", "Gamma": True}), encoding="utf-8")
        assert read_recorded_file_counts(path) == {"Alpha": 10}


class TestBuildChannelIdToName:
    """Mapping channel IDs back to directory names for CSV reuse."""

    def test_maps_id_to_directory_name(self, tmp_path: Path) -> None:
        """A channel's ID resolves to its sanitized directory name."""
        write_info_json(tmp_path / "Anthropic" / "video", "v", info_payload(channel_id="UCabc"))
        assert build_channel_id_to_name(tmp_path, "video", None) == {"UCabc": "Anthropic"}

    def test_one_file_per_channel(self, tmp_path: Path) -> None:
        """Each channel contributes its own ID."""
        write_info_json(tmp_path / "Alpha" / "video", "a", info_payload(channel_id="UCa"))
        write_info_json(tmp_path / "Beta" / "video", "b", info_payload(channel_id="UCb"))
        assert build_channel_id_to_name(tmp_path, "video", None) == {"UCa": "Alpha", "UCb": "Beta"}

    def test_skips_unreadable_until_valid(self, tmp_path: Path) -> None:
        """An unreadable first file is skipped in favour of a valid one."""
        video_dir = tmp_path / "Alpha" / "video"
        video_dir.mkdir(parents=True)
        (video_dir / "aaa.info.json").write_text("{oops", encoding="utf-8")
        write_info_json(video_dir, "bbb", info_payload(channel_id="UCa"))
        assert build_channel_id_to_name(tmp_path, "video", None) == {"UCa": "Alpha"}


class TestLoadRecordsFromCsv:
    """Reconstructing records from a previously written CSV."""

    def test_roundtrips_records(self, tmp_path: Path) -> None:
        """A rendered CSV reloads to the same records under the name map."""
        records = [make_record("Alpha", datetime(2025, 6, 9, 12, 0, tzinfo=UTC), channel_id="UCa", video_id="vid00000001")]
        csv_path = tmp_path / "t.csv"
        csv_path.write_text(render_timeseries_csv(records), encoding="utf-8")
        assert load_records_from_csv(csv_path, {"UCa": "Alpha"}) == records

    def test_unmapped_id_labelled_by_id(self, tmp_path: Path) -> None:
        """An ID missing from the map is labelled by the ID itself."""
        records = [make_record("Alpha", datetime(2025, 6, 9, tzinfo=UTC), channel_id="UCzzz")]
        csv_path = tmp_path / "t.csv"
        csv_path.write_text(render_timeseries_csv(records), encoding="utf-8")
        assert load_records_from_csv(csv_path, {})[0].channel == "UCzzz"

    def test_title_with_comma_roundtrips(self, tmp_path: Path) -> None:
        """A quoted title with a comma reloads intact."""
        records = [make_record("Alpha", datetime(2025, 6, 9, tzinfo=UTC), title="Rag, Explained", channel_id="UCa")]
        csv_path = tmp_path / "t.csv"
        csv_path.write_text(render_timeseries_csv(records), encoding="utf-8")
        assert load_records_from_csv(csv_path, {"UCa": "Alpha"})[0].title == "Rag, Explained"

    def test_unexpected_header_raises(self, tmp_path: Path) -> None:
        """A CSV whose header does not match is rejected, not misread."""
        csv_path = tmp_path / "t.csv"
        csv_path.write_text("wrong,header\n1,2\n", encoding="utf-8")
        with pytest.raises(AnalyticsError, match="unexpected header"):
            load_records_from_csv(csv_path, {})

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        """An absent CSV is an explicit error."""
        with pytest.raises(AnalyticsError, match="cannot be read"):
            load_records_from_csv(tmp_path / "absent.csv", {})


class TestBuildIncrementalRecords:
    """Reusing unchanged channels and rescanning only the changed ones."""

    def test_reuses_unchanged_channel(self, tmp_path: Path) -> None:
        """A channel whose count matches and has rows is reused, not reread."""
        alpha_dir = tmp_path / "Alpha" / "video"
        write_info_json(alpha_dir, "a", info_payload())
        existing = {"Alpha": [make_record("Alpha", datetime(2025, 1, 1, tzinfo=UTC))]}
        records, failures, rescanned = build_incremental_records(
            [("Alpha", alpha_dir)], {"Alpha": 1}, {"Alpha": 1}, existing, lambda _: None
        )
        assert rescanned == []
        assert records == existing["Alpha"]
        assert failures == []

    def test_rescans_changed_channel(self, tmp_path: Path) -> None:
        """A changed file count forces the channel to be reread from metadata."""
        alpha_dir = tmp_path / "Alpha" / "video"
        write_info_json(alpha_dir, "a", info_payload(id="aaaaaaaaaaa"))
        write_info_json(alpha_dir, "b", info_payload(id="bbbbbbbbbbb"))
        existing = {"Alpha": [make_record("Alpha", datetime(2025, 1, 1, tzinfo=UTC))]}
        records, _, rescanned = build_incremental_records([("Alpha", alpha_dir)], {"Alpha": 2}, {"Alpha": 1}, existing, lambda _: None)
        assert rescanned == ["Alpha"]
        assert len(records) == 2

    def test_new_channel_is_rescanned(self, tmp_path: Path) -> None:
        """A channel absent from the recorded counts is rescanned."""
        alpha_dir = tmp_path / "Alpha" / "video"
        write_info_json(alpha_dir, "a", info_payload())
        records, _, rescanned = build_incremental_records([("Alpha", alpha_dir)], {"Alpha": 1}, {}, {}, lambda _: None)
        assert rescanned == ["Alpha"]
        assert len(records) == 1

    def test_count_matches_but_no_existing_rows_rescans(self, tmp_path: Path) -> None:
        """Matching counts still rescan when the CSV holds no rows for the channel."""
        alpha_dir = tmp_path / "Alpha" / "video"
        write_info_json(alpha_dir, "a", info_payload())
        records, _, rescanned = build_incremental_records([("Alpha", alpha_dir)], {"Alpha": 1}, {"Alpha": 1}, {}, lambda _: None)
        assert rescanned == ["Alpha"]
        assert len(records) == 1

    def test_mixed_reuse_and_rescan(self, tmp_path: Path) -> None:
        """Only the changed channel is reread; the unchanged one is reused."""
        alpha_dir = tmp_path / "Alpha" / "video"
        beta_dir = tmp_path / "Beta" / "video"
        write_info_json(alpha_dir, "a", info_payload(id="aaaaaaaaaaa"))
        write_info_json(beta_dir, "b", info_payload(id="bbbbbbbbbbb"))
        write_info_json(beta_dir, "c", info_payload(id="ccccccccccc"))
        existing = {
            "Alpha": [make_record("Alpha", datetime(2025, 1, 1, tzinfo=UTC))],
            "Beta": [make_record("Beta", datetime(2025, 1, 1, tzinfo=UTC))],
        }
        records, _, rescanned = build_incremental_records(
            [("Alpha", alpha_dir), ("Beta", beta_dir)],
            {"Alpha": 1, "Beta": 2},
            {"Alpha": 1, "Beta": 1},
            existing,
            lambda _: None,
        )
        assert rescanned == ["Beta"]
        assert len(records) == 3
