"""Unit tests for the SQLite video metadata database."""

import json
import os
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from src.metadata_db import (
    RESERVED_COLUMNS,
    TABLE_NAME,
    MetadataDbError,
    UpdateSummary,
    count_rows,
    encode_value,
    list_channel_metadata_files,
    metadata_db_is_ready,
    open_metadata_db,
    open_metadata_db_readonly,
    read_updated_at,
    select_published_between,
    update_metadata_db,
)

# 2025-06-09T12:00:00+00:00
BASE_TIMESTAMP = 1749470400
NOW = datetime(2026, 9, 4, 10, 30, tzinfo=UTC)
VIDEO_SUBDIR = "video"


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


def write_info_json(metadata_dir: Path, channel: str, name: str, payload: dict[str, Any]) -> Path:
    """Write a payload as <channel>/video/<name>.info.json and return its path."""
    directory = metadata_dir / channel / VIDEO_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.info.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_update(db_path: Path, metadata_dir: Path) -> tuple[UpdateSummary, list[str]]:
    """Run one update pass and return the summary plus captured progress lines."""
    progress: list[str] = []
    with open_metadata_db(db_path) as connection:
        summary = update_metadata_db(connection, metadata_dir, VIDEO_SUBDIR, NOW, progress.append)
    return summary, progress


def fetch_rows(db_path: Path) -> list[sqlite3.Row]:
    """Read every row of the metadata table ordered by file path."""
    with open_metadata_db_readonly(db_path) as connection:
        return connection.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY file_path").fetchall()


def column_names(db_path: Path) -> set[str]:
    """Return the metadata table's column names."""
    with open_metadata_db_readonly(db_path) as connection:
        return {row["name"] for row in connection.execute(f"PRAGMA table_info({TABLE_NAME})")}


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    """An empty synthetic metadata root."""
    root = tmp_path / "metadata"
    root.mkdir()
    return root


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path for a fresh database file."""
    return tmp_path / "metadata.sqlite"


class TestEncodeValue:
    """Mapping JSON values onto SQLite storage classes."""

    def test_none_stays_null(self) -> None:
        """JSON null becomes SQL NULL."""
        assert encode_value(None) is None

    def test_bool_becomes_integer(self) -> None:
        """SQLite has no boolean; true is 1 and false is 0."""
        assert encode_value(True) == 1
        assert encode_value(False) == 0

    @pytest.mark.parametrize("scalar", [7, 2.5, "text"])
    def test_scalars_pass_through(self, scalar: int | float | str) -> None:
        """Numbers and strings are stored natively."""
        assert encode_value(scalar) == scalar

    def test_nested_values_become_json_text(self) -> None:
        """Lists and objects are stored as JSON text."""
        assert encode_value([1, {"a": "ü"}]) == '[1, {"a": "ü"}]'


class TestListChannelMetadataFiles:
    """Discovering the per-channel metadata files."""

    def test_missing_metadata_dir_raises(self, tmp_path: Path) -> None:
        """An absent metadata root is an error, not an empty scan."""
        with pytest.raises(MetadataDbError, match="not found"):
            list_channel_metadata_files(tmp_path / "missing", VIDEO_SUBDIR)

    def test_channels_and_files_are_sorted(self, metadata_dir: Path) -> None:
        """Channels and their files come back in name order."""
        write_info_json(metadata_dir, "Zeta", "b", info_payload())
        write_info_json(metadata_dir, "Zeta", "a", info_payload())
        write_info_json(metadata_dir, "Alpha", "c", info_payload())
        listing = list_channel_metadata_files(metadata_dir, VIDEO_SUBDIR)
        assert [channel for channel, _ in listing] == ["Alpha", "Zeta"]
        assert [path.name for path in listing[1][1]] == ["a.info.json", "b.info.json"]

    def test_appledouble_sidecars_and_dot_dirs_are_ignored(self, metadata_dir: Path) -> None:
        """macOS ._ sidecars and hidden directories are not metadata."""
        write_info_json(metadata_dir, "Alpha", "a", info_payload())
        write_info_json(metadata_dir, "Alpha", "._a", info_payload())
        write_info_json(metadata_dir, ".hidden", "h", info_payload())
        listing = list_channel_metadata_files(metadata_dir, VIDEO_SUBDIR)
        assert [channel for channel, _ in listing] == ["Alpha"]
        assert [path.name for path in listing[0][1]] == ["a.info.json"]

    def test_channel_without_video_subdir_is_skipped(self, metadata_dir: Path) -> None:
        """A channel directory with no video subdirectory contributes nothing."""
        (metadata_dir / "Empty").mkdir()
        write_info_json(metadata_dir, "Alpha", "a", info_payload())
        assert [channel for channel, _ in list_channel_metadata_files(metadata_dir, VIDEO_SUBDIR)] == ["Alpha"]


class TestSchema:
    """Table and index creation."""

    def test_open_creates_table_and_indexes(self, db_path: Path) -> None:
        """Opening a fresh database creates the table and both indexes."""
        with open_metadata_db(db_path) as connection:
            indexes = {row["name"] for row in connection.execute(f"PRAGMA index_list({TABLE_NAME})")}
            columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({TABLE_NAME})")}
        assert {"idx_video_metadata_channel_id", "idx_video_metadata_published_at"} <= indexes
        assert set(RESERVED_COLUMNS) <= columns
        assert {"id", "title", "channel_id", "timestamp", "duration"} <= columns

    def test_writer_keeps_no_rollback_journal(self, db_path: Path, metadata_dir: Path) -> None:
        """The writer runs with journal_mode OFF and leaves no journal file."""
        write_info_json(metadata_dir, "Alpha", "v", info_payload())
        with open_metadata_db(db_path) as connection:
            assert str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "off"
        run_update(db_path, metadata_dir)
        assert not db_path.with_name(db_path.name + "-journal").exists()

    def test_absent_database_is_not_ready(self, db_path: Path) -> None:
        """No file means not ready."""
        assert metadata_db_is_ready(db_path) is False

    def test_database_without_a_finished_pass_is_not_ready(self, db_path: Path) -> None:
        """A schema without a recorded update instant is not ready."""
        with open_metadata_db(db_path):
            pass
        assert metadata_db_is_ready(db_path) is False

    def test_database_after_a_finished_pass_is_ready(self, db_path: Path, metadata_dir: Path) -> None:
        """A finished update pass makes the database ready."""
        write_info_json(metadata_dir, "Alpha", "v", info_payload())
        run_update(db_path, metadata_dir)
        assert metadata_db_is_ready(db_path) is True

    def test_database_with_unknown_columns_is_not_ready(self, db_path: Path, metadata_dir: Path) -> None:
        """A table with columns the schema does not define is not ready."""
        write_info_json(metadata_dir, "Alpha", "v", info_payload())
        run_update(db_path, metadata_dir)
        with sqlite3.connect(db_path) as connection:
            connection.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN extra TEXT")
        assert metadata_db_is_ready(db_path) is False

    def test_readonly_open_of_missing_database_raises(self, db_path: Path) -> None:
        """Readers never create a database by accident."""
        with pytest.raises(MetadataDbError, match="not found"):
            open_metadata_db_readonly(db_path)

    def test_readonly_connection_rejects_writes(self, db_path: Path) -> None:
        """The read-only connection cannot modify the database."""
        with open_metadata_db(db_path):
            pass
        with open_metadata_db_readonly(db_path) as connection, pytest.raises(sqlite3.OperationalError):
            connection.execute(f"DELETE FROM {TABLE_NAME}")


class TestUpdateMetadataDb:
    """Mirroring the metadata files into the table."""

    def test_listing_fields_are_stored(self, db_path: Path, metadata_dir: Path) -> None:
        """The listing fields are stored with their values."""
        write_info_json(metadata_dir, "Alpha", "v", info_payload())
        summary, _ = run_update(db_path, metadata_dir)
        row = fetch_rows(db_path)[0]
        assert summary.inserted == 1
        assert row["id"] == "abc123XYZ09"
        assert row["title"] == "Building RAG Systems"
        assert row["channel_id"] == "UCtestchannelid0000000"
        assert row["timestamp"] == BASE_TIMESTAMP
        assert row["duration"] == 600

    def test_keys_outside_the_listing_fields_are_not_stored(self, db_path: Path, metadata_dir: Path) -> None:
        """Keys outside the listing fields do not become columns."""
        payload = info_payload(view_count=42, formats=[{"format_id": "251"}], thumbnails=[{"url": "https://example.test/t.jpg"}])
        write_info_json(metadata_dir, "Alpha", "v", payload)
        summary, _ = run_update(db_path, metadata_dir)
        assert summary.inserted == 1
        assert column_names(db_path) == {*RESERVED_COLUMNS, "id", "title", "channel_id", "timestamp", "duration"}

    def test_listing_field_absent_from_the_file_is_null(self, db_path: Path, metadata_dir: Path) -> None:
        """A file without one of the listing fields still yields a row, with NULL there."""
        payload = info_payload()
        del payload["duration"]
        write_info_json(metadata_dir, "Alpha", "v", payload)
        summary, _ = run_update(db_path, metadata_dir)
        assert summary.inserted == 1
        assert fetch_rows(db_path)[0]["duration"] is None

    def test_row_keeps_the_file_facts(self, db_path: Path, metadata_dir: Path) -> None:
        """The relative path, channel, mtime, and size are recorded."""
        path = write_info_json(metadata_dir, "Alpha", "v", info_payload())
        run_update(db_path, metadata_dir)
        row = fetch_rows(db_path)[0]
        stat = path.stat()
        assert row["file_path"] == "Alpha/video/v.info.json"
        assert row["file_channel"] == "Alpha"
        assert row["file_mtime_ns"] == stat.st_mtime_ns
        assert row["file_size"] == stat.st_size

    def test_published_at_is_derived_from_timestamp_in_utc(self, db_path: Path, metadata_dir: Path) -> None:
        """The publish instant column is the ISO UTC form of the epoch timestamp."""
        write_info_json(metadata_dir, "Alpha", "v", info_payload(timestamp=BASE_TIMESTAMP + 3661))
        run_update(db_path, metadata_dir)
        assert fetch_rows(db_path)[0]["published_at"] == "2025-06-09T13:01:01+00:00"

    def test_missing_timestamp_leaves_published_at_null(self, db_path: Path, metadata_dir: Path) -> None:
        """A file without a timestamp is stored, with no publish instant."""
        payload = info_payload()
        del payload["timestamp"]
        write_info_json(metadata_dir, "Alpha", "v", payload)
        summary, _ = run_update(db_path, metadata_dir)
        assert summary.inserted == 1
        assert fetch_rows(db_path)[0]["published_at"] is None

    def test_unrepresentable_timestamp_is_a_failure(self, db_path: Path, metadata_dir: Path) -> None:
        """A numeric timestamp outside the datetime range is reported, not stored."""
        write_info_json(metadata_dir, "Alpha", "v", info_payload(timestamp=10**18))
        summary, _ = run_update(db_path, metadata_dir)
        assert len(summary.failures) == 1
        assert "timestamp" in summary.failures[0]
        assert fetch_rows(db_path) == []

    def test_unchanged_file_is_skipped_on_second_run(self, db_path: Path, metadata_dir: Path) -> None:
        """A file with the same mtime and size is not re-read."""
        write_info_json(metadata_dir, "Alpha", "v", info_payload())
        run_update(db_path, metadata_dir)
        summary, progress = run_update(db_path, metadata_dir)
        assert (summary.inserted, summary.updated, summary.unchanged) == (0, 0, 1)
        assert progress == ["Skipping: Alpha (unchanged, 1 files)"]

    def test_changed_file_is_reread(self, db_path: Path, metadata_dir: Path) -> None:
        """A file whose mtime or size changed is stored again."""
        path = write_info_json(metadata_dir, "Alpha", "v", info_payload())
        run_update(db_path, metadata_dir)
        path.write_text(json.dumps(info_payload(title="Renamed Video")), encoding="utf-8")
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000_000))
        summary, progress = run_update(db_path, metadata_dir)
        assert summary.updated == 1
        assert fetch_rows(db_path)[0]["title"] == "Renamed Video"
        assert progress == ["Processing: Alpha (1 files: 0 new, 1 changed, 0 removed)", "done"]

    def test_row_for_removed_file_is_deleted(self, db_path: Path, metadata_dir: Path) -> None:
        """The table mirrors the files: a vanished file loses its row."""
        keep = write_info_json(metadata_dir, "Alpha", "keep", info_payload())
        gone = write_info_json(metadata_dir, "Alpha", "gone", info_payload())
        run_update(db_path, metadata_dir)
        gone.unlink()
        summary, _ = run_update(db_path, metadata_dir)
        assert summary.removed == 1
        assert [row["file_path"] for row in fetch_rows(db_path)] == [keep.relative_to(metadata_dir).as_posix()]

    def test_rows_of_a_vanished_channel_are_deleted(self, db_path: Path, metadata_dir: Path) -> None:
        """A channel directory that disappears takes its rows with it."""
        write_info_json(metadata_dir, "Alpha", "a", info_payload())
        write_info_json(metadata_dir, "Beta", "b", info_payload())
        run_update(db_path, metadata_dir)
        for path in sorted((metadata_dir / "Beta").rglob("*")):
            if path.is_file():
                path.unlink()
        (metadata_dir / "Beta" / VIDEO_SUBDIR).rmdir()
        (metadata_dir / "Beta").rmdir()
        summary, _ = run_update(db_path, metadata_dir)
        assert summary.removed == 1
        assert [row["file_channel"] for row in fetch_rows(db_path)] == ["Alpha"]

    def test_invalid_json_is_a_failure_and_not_stored(self, db_path: Path, metadata_dir: Path) -> None:
        """An unreadable file is reported and the scan continues."""
        directory = metadata_dir / "Alpha" / VIDEO_SUBDIR
        directory.mkdir(parents=True)
        (directory / "bad.info.json").write_text("{not json", encoding="utf-8")
        write_info_json(metadata_dir, "Alpha", "good", info_payload())
        summary, _ = run_update(db_path, metadata_dir)
        assert summary.inserted == 1
        assert len(summary.failures) == 1
        assert "bad.info.json" in summary.failures[0]

    def test_non_object_json_is_a_failure(self, db_path: Path, metadata_dir: Path) -> None:
        """A JSON array has no keys to store."""
        directory = metadata_dir / "Alpha" / VIDEO_SUBDIR
        directory.mkdir(parents=True)
        (directory / "list.info.json").write_text("[1, 2]", encoding="utf-8")
        summary, _ = run_update(db_path, metadata_dir)
        assert len(summary.failures) == 1
        assert "not a JSON object" in summary.failures[0]

    def test_stale_row_of_now_invalid_file_is_removed(self, db_path: Path, metadata_dir: Path) -> None:
        """When a stored file turns invalid, its old row does not linger."""
        path = write_info_json(metadata_dir, "Alpha", "v", info_payload())
        run_update(db_path, metadata_dir)
        path.write_text("{broken", encoding="utf-8")
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000_000))
        summary, _ = run_update(db_path, metadata_dir)
        assert len(summary.failures) == 1
        assert fetch_rows(db_path) == []

    @pytest.mark.parametrize("reserved", ["file_path", "published_at"])
    def test_key_named_like_a_bookkeeping_column_is_ignored(self, db_path: Path, metadata_dir: Path, reserved: str) -> None:
        """A metadata key sharing a bookkeeping column's name cannot overwrite it."""
        write_info_json(metadata_dir, "Alpha", "v", info_payload(**{reserved: "x"}))
        summary, _ = run_update(db_path, metadata_dir)
        assert summary.failures == []
        assert fetch_rows(db_path)[0][reserved] != "x"

    def test_summary_counts_channels_and_files(self, db_path: Path, metadata_dir: Path) -> None:
        """The summary reports what was scanned."""
        write_info_json(metadata_dir, "Alpha", "a", info_payload())
        write_info_json(metadata_dir, "Beta", "b", info_payload())
        write_info_json(metadata_dir, "Beta", "c", info_payload())
        summary, progress = run_update(db_path, metadata_dir)
        assert (summary.channels, summary.files_seen, summary.inserted) == (2, 3, 3)
        assert progress == [
            "Processing: Alpha (1 files: 1 new, 0 changed, 0 removed)",
            "done",
            "Processing: Beta (2 files: 2 new, 0 changed, 0 removed)",
            "done",
        ]

    def test_update_records_the_update_instant(self, db_path: Path, metadata_dir: Path) -> None:
        """The last update instant is stored for readers to display."""
        write_info_json(metadata_dir, "Alpha", "a", info_payload())
        run_update(db_path, metadata_dir)
        with open_metadata_db_readonly(db_path) as connection:
            assert read_updated_at(connection) == NOW.isoformat()

    def test_fresh_database_has_no_update_instant(self, db_path: Path) -> None:
        """Before any update there is no instant to report."""
        with open_metadata_db(db_path) as connection:
            assert read_updated_at(connection) is None


class TestSelectPublishedBetween:
    """The indexed date-window query used by list-videos."""

    def test_bounds_are_inclusive_by_utc_calendar_date(self, db_path: Path, metadata_dir: Path) -> None:
        """Rows on the first and last day are returned; neighbours are not."""
        day = 86400
        first = datetime(2026, 8, 1, 0, 0, tzinfo=UTC).timestamp()
        write_info_json(metadata_dir, "A", "before", info_payload(id="before", timestamp=first - 1))
        write_info_json(metadata_dir, "A", "first", info_payload(id="first", timestamp=first))
        write_info_json(metadata_dir, "A", "last", info_payload(id="last", timestamp=first + 31 * day - 1))
        write_info_json(metadata_dir, "A", "after", info_payload(id="after", timestamp=first + 31 * day))
        run_update(db_path, metadata_dir)
        with open_metadata_db_readonly(db_path) as connection:
            rows = select_published_between(connection, date(2026, 8, 1), date(2026, 8, 31))
        assert sorted(row["id"] for row in rows) == ["first", "last"]

    def test_rows_without_publish_instant_are_not_returned(self, db_path: Path, metadata_dir: Path) -> None:
        """A row with NULL published_at is in no window."""
        payload = info_payload()
        del payload["timestamp"]
        write_info_json(metadata_dir, "A", "nodate", payload)
        run_update(db_path, metadata_dir)
        with open_metadata_db_readonly(db_path) as connection:
            assert select_published_between(connection, date(2000, 1, 1), date(2100, 1, 1)) == []

    def test_rows_carry_the_listing_fields(self, db_path: Path, metadata_dir: Path) -> None:
        """The query returns the channel dir plus the fields a publication record needs."""
        write_info_json(metadata_dir, "Alpha", "v", info_payload())
        run_update(db_path, metadata_dir)
        with open_metadata_db_readonly(db_path) as connection:
            row = select_published_between(connection, date(2025, 6, 9), date(2025, 6, 9))[0]
        assert row["file_channel"] == "Alpha"
        assert row["id"] == "abc123XYZ09"
        assert row["channel_id"] == "UCtestchannelid0000000"
        assert row["title"] == "Building RAG Systems"
        assert row["published_at"] == "2025-06-09T12:00:00+00:00"
        assert row["duration"] == 600

    def test_count_rows_reports_total_and_undated(self, db_path: Path, metadata_dir: Path) -> None:
        """Readers can show how many rows exist and how many have no publish instant."""
        payload = info_payload()
        del payload["timestamp"]
        write_info_json(metadata_dir, "A", "nodate", payload)
        write_info_json(metadata_dir, "A", "dated", info_payload())
        run_update(db_path, metadata_dir)
        with open_metadata_db_readonly(db_path) as connection:
            assert count_rows(connection) == (2, 1)
