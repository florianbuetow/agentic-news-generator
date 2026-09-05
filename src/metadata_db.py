"""SQLite index of the yt-dlp video metadata files.

One row per ``.info.json`` file under the metadata directory, holding the
listing fields alone: the video id, title, channel id, timestamp and duration,
plus the publish instant derived from the timestamp. Bookkeeping columns record
where each row came from and when its file last changed, which lets an update
pass skip files that did not change and drop rows whose files are gone.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

TABLE_NAME: str = "video_metadata"
INFO_TABLE_NAME: str = "metadata_db_info"
UPDATED_AT_KEY: str = "updated_at"
METADATA_SUFFIX: str = ".info.json"
APPLEDOUBLE_PREFIX: str = "._"
FILE_PATH_COLUMN: str = "file_path"
FILE_CHANNEL_COLUMN: str = "file_channel"
FILE_MTIME_COLUMN: str = "file_mtime_ns"
FILE_SIZE_COLUMN: str = "file_size"
PUBLISHED_AT_COLUMN: str = "published_at"
RESERVED_COLUMNS: tuple[str, ...] = (
    FILE_PATH_COLUMN,
    FILE_CHANNEL_COLUMN,
    FILE_MTIME_COLUMN,
    FILE_SIZE_COLUMN,
    PUBLISHED_AT_COLUMN,
)
CORE_METADATA_COLUMNS: tuple[str, ...] = ("id", "title", "channel_id", "timestamp", "duration")
STORED_COLUMNS: tuple[str, ...] = (*RESERVED_COLUMNS, *CORE_METADATA_COLUMNS)
TIMESTAMP_KEY: str = "timestamp"


class MetadataDbError(Exception):
    """The metadata database or a metadata file cannot be read or written."""


class UpdateSummary(BaseModel):
    """What one update pass saw and changed."""

    channels: int = Field(..., ge=0, description="Channel directories that hold a video metadata subdirectory")
    files_seen: int = Field(..., ge=0, description="Metadata files found on disk")
    inserted: int = Field(..., ge=0, description="Files stored for the first time")
    updated: int = Field(..., ge=0, description="Files stored again because they changed")
    unchanged: int = Field(..., ge=0, description="Files skipped because mtime and size matched their row")
    removed: int = Field(..., ge=0, description="Rows deleted because their file is gone or unreadable")
    failures: list[str] = Field(..., description="Human-readable messages for files that could not be stored")

    model_config = ConfigDict(frozen=True, extra="forbid")


def encode_value(value: object) -> int | float | str | None:
    """Map a JSON value onto an SQLite storage class; nested values become JSON text."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        return value
    return json.dumps(value, ensure_ascii=False)


def ensure_schema(connection: sqlite3.Connection) -> None:
    """Create the metadata table, its indexes, and the info table if absent."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS video_metadata ("
        "file_path TEXT PRIMARY KEY, file_channel TEXT NOT NULL, file_mtime_ns INTEGER NOT NULL, "
        "file_size INTEGER NOT NULL, published_at TEXT, "
        '"id", "title", "channel_id", "timestamp", "duration")'
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_video_metadata_channel_id ON video_metadata (channel_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_video_metadata_published_at ON video_metadata (published_at)")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata_db_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.commit()


def open_metadata_db(db_path: Path) -> sqlite3.Connection:
    """Open the database for writing, creating the file and schema when needed.

    The writer keeps no rollback journal; an interrupted pass is built again.

    Raises:
        MetadataDbError: If the directory that should hold the database is absent.
    """
    if not db_path.parent.is_dir():
        raise MetadataDbError(f"Metadata database directory not found: {db_path.parent}")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = OFF")
    ensure_schema(connection)
    return connection


def open_metadata_db_readonly(db_path: Path) -> sqlite3.Connection:
    """Open an existing database read-only.

    Raises:
        MetadataDbError: If the database file does not exist.
    """
    if not db_path.is_file():
        raise MetadataDbError(f"Metadata database not found: {db_path}")
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def metadata_db_is_ready(db_path: Path) -> bool:
    """Report whether the database holds the result of a finished update pass."""
    if not db_path.is_file():
        return False
    try:
        with closing(open_metadata_db_readonly(db_path)) as connection:
            tables = {str(row["name"]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            if not {TABLE_NAME, INFO_TABLE_NAME} <= tables:
                return False
            stored = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({TABLE_NAME})")}
            if not stored <= set(STORED_COLUMNS):
                return False
            return read_updated_at(connection) is not None
    except (MetadataDbError, sqlite3.Error):
        return False


def list_channel_metadata_files(metadata_dir: Path, video_subdir: str) -> list[tuple[str, list[Path]]]:
    """List every channel's video metadata files, channels and files in name order.

    Hidden channel directories and macOS AppleDouble sidecars are ignored; a
    channel without the video subdirectory is not listed at all.

    Raises:
        MetadataDbError: If the metadata root is absent.
    """
    if not metadata_dir.is_dir():
        raise MetadataDbError(f"Metadata directory not found: {metadata_dir}")
    listing: list[tuple[str, list[Path]]] = []
    for channel_dir in sorted(metadata_dir.iterdir()):
        if not channel_dir.is_dir() or channel_dir.name.startswith("."):
            continue
        video_dir = channel_dir / video_subdir
        if not video_dir.is_dir():
            continue
        files = sorted(
            path
            for path in video_dir.iterdir()
            if path.is_file() and path.name.endswith(METADATA_SUFFIX) and not path.name.startswith(APPLEDOUBLE_PREFIX)
        )
        listing.append((channel_dir.name, files))
    return listing


def read_metadata_file(path: Path) -> dict[str, object]:
    """Read one metadata file as its parsed top-level object.

    Raises:
        MetadataDbError: If the file is unreadable, not JSON, or not a JSON object.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise MetadataDbError(f"Metadata file unreadable: {path}: {e}") from e
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise MetadataDbError(f"Metadata file is not valid JSON: {path}: {e}") from e
    if not isinstance(payload, dict):
        raise MetadataDbError(f"Metadata file is not a JSON object: {path}")
    return cast(dict[str, object], payload)


def derive_published_at(fields: dict[str, object], path: Path) -> str | None:
    """Return the ISO UTC publish instant from the epoch timestamp, or None without one.

    Raises:
        MetadataDbError: If the timestamp is a number that no datetime can represent.
    """
    if TIMESTAMP_KEY not in fields:
        return None
    value = fields[TIMESTAMP_KEY]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return datetime.fromtimestamp(value, tz=UTC).isoformat()
    except (OverflowError, OSError, ValueError) as e:
        raise MetadataDbError(f"Metadata 'timestamp' is not a representable instant: {value} in {path}") from e


def store_metadata_file(
    connection: sqlite3.Connection,
    metadata_dir: Path,
    channel: str,
    path: Path,
    stat: os.stat_result,
) -> None:
    """Read one metadata file and write its listing fields as a row, replacing any previous row.

    Raises:
        MetadataDbError: If the file cannot be read or one of its values cannot be stored.
    """
    fields = read_metadata_file(path)
    published_at = derive_published_at(fields, path)
    values: list[int | float | str | None] = [
        path.relative_to(metadata_dir).as_posix(),
        channel,
        stat.st_mtime_ns,
        stat.st_size,
        published_at,
        *(encode_value(fields.get(name)) for name in CORE_METADATA_COLUMNS),
    ]
    column_sql = ", ".join(f'"{name}"' for name in STORED_COLUMNS)
    placeholders = ", ".join("?" for _ in STORED_COLUMNS)
    try:
        connection.execute(f"INSERT OR REPLACE INTO {TABLE_NAME} ({column_sql}) VALUES ({placeholders})", values)
    except OverflowError as e:
        raise MetadataDbError(f"Metadata value too large for SQLite: {path}: {e}") from e


def existing_rows_by_channel(connection: sqlite3.Connection) -> dict[str, dict[str, tuple[int, int]]]:
    """Map channel to its stored file paths and their recorded (mtime_ns, size)."""
    existing: dict[str, dict[str, tuple[int, int]]] = {}
    for row in connection.execute("SELECT file_path, file_channel, file_mtime_ns, file_size FROM video_metadata"):
        channel = str(row[FILE_CHANNEL_COLUMN])
        if channel not in existing:
            existing[channel] = {}
        existing[channel][str(row[FILE_PATH_COLUMN])] = (int(row[FILE_MTIME_COLUMN]), int(row[FILE_SIZE_COLUMN]))
    return existing


def delete_rows(connection: sqlite3.Connection, keys: Iterable[str]) -> None:
    """Delete the rows with the given file path keys."""
    connection.executemany("DELETE FROM video_metadata WHERE file_path = ?", [(key,) for key in keys])


def write_updated_at(connection: sqlite3.Connection, now: datetime) -> None:
    """Record the instant of the update pass that just finished."""
    connection.execute("INSERT OR REPLACE INTO metadata_db_info (key, value) VALUES (?, ?)", (UPDATED_AT_KEY, now.isoformat()))


def read_updated_at(connection: sqlite3.Connection) -> str | None:
    """Return the ISO instant of the last update pass, or None before the first one."""
    row = connection.execute("SELECT value FROM metadata_db_info WHERE key = ?", (UPDATED_AT_KEY,)).fetchone()
    if row is None:
        return None
    return str(row["value"])


def classify_channel_files(
    metadata_dir: Path,
    files: list[Path],
    channel_rows: dict[str, tuple[int, int]],
) -> tuple[list[tuple[Path, str, os.stat_result]], int, list[str], set[str]]:
    """Sort a channel's files into those to (re)read and those already current.

    Returns:
        ``(pending, unchanged, failures, seen)``: files to store with their row
        key and stat, the count of files whose row is current, messages for
        files that could not be inspected, and the row keys of every file seen.
    """
    pending: list[tuple[Path, str, os.stat_result]] = []
    failures: list[str] = []
    seen: set[str] = set()
    unchanged = 0
    for path in files:
        key = path.relative_to(metadata_dir).as_posix()
        seen.add(key)
        try:
            stat = path.stat()
        except OSError as e:
            failures.append(f"Metadata file cannot be inspected: {path}: {e}")
            continue
        if key in channel_rows and channel_rows[key] == (stat.st_mtime_ns, stat.st_size):
            unchanged += 1
            continue
        pending.append((path, key, stat))
    return pending, unchanged, failures, seen


def sync_channel(
    connection: sqlite3.Connection,
    metadata_dir: Path,
    channel: str,
    files: list[Path],
    channel_rows: dict[str, tuple[int, int]],
    progress: Callable[[str], None],
) -> UpdateSummary:
    """Bring one channel's rows in line with its files and commit the result.

    Returns:
        The counts for this channel alone, with ``channels`` set to one.
    """
    pending, unchanged, failures, seen = classify_channel_files(metadata_dir, files, channel_rows)
    stale = sorted(set(channel_rows) - seen)
    inserted = updated = 0
    processing = bool(files) and (bool(pending) or bool(stale))
    if not files:
        progress(f"Skipping: {channel} (no video metadata)")
    elif not processing:
        progress(f"Skipping: {channel} (unchanged, {len(files)} files)")
    else:
        new_count = sum(1 for _, key, _ in pending if key not in channel_rows)
        progress(f"Processing: {channel} ({len(files)} files: {new_count} new, {len(pending) - new_count} changed, {len(stale)} removed)")
        for path, key, stat in pending:
            try:
                store_metadata_file(connection, metadata_dir, channel, path, stat)
            except MetadataDbError as e:
                failures.append(str(e))
                delete_rows(connection, [key])
                continue
            if key in channel_rows:
                updated += 1
            else:
                inserted += 1
    delete_rows(connection, stale)
    connection.commit()
    if processing:
        progress("done")
    return UpdateSummary(
        channels=1,
        files_seen=len(files),
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        removed=len(stale),
        failures=failures,
    )


def update_metadata_db(
    connection: sqlite3.Connection,
    metadata_dir: Path,
    video_subdir: str,
    now: datetime,
    progress: Callable[[str], None],
) -> UpdateSummary:
    """Bring the table in line with the metadata files on disk.

    A file whose recorded mtime and size still match is skipped without being
    read. New and changed files are read and stored; rows whose file is gone or
    can no longer be read are deleted. One failure never stops the pass: every
    failure is collected into the summary. Each channel is committed as soon as
    it is done so an interrupted pass keeps the channels it finished.

    Args:
        connection: Writable connection with the schema in place.
        metadata_dir: Root metadata directory holding one subdirectory per channel.
        video_subdir: Name of the per-channel subdirectory holding video metadata.
        now: Instant recorded as the time of this update.
        progress: Sink for per-channel progress lines.

    Raises:
        MetadataDbError: If the metadata root is absent.
    """
    listing = list_channel_metadata_files(metadata_dir, video_subdir)
    existing = existing_rows_by_channel(connection)

    results: list[UpdateSummary] = []
    for channel, files in listing:
        channel_rows: dict[str, tuple[int, int]] = {}
        if channel in existing:
            channel_rows = existing.pop(channel)
        results.append(sync_channel(connection, metadata_dir, channel, files, channel_rows, progress))

    vanished = 0
    for channel_rows in existing.values():
        delete_rows(connection, sorted(channel_rows))
        vanished += len(channel_rows)

    write_updated_at(connection, now)
    connection.commit()
    return UpdateSummary(
        channels=len(listing),
        files_seen=sum(result.files_seen for result in results),
        inserted=sum(result.inserted for result in results),
        updated=sum(result.updated for result in results),
        unchanged=sum(result.unchanged for result in results),
        removed=sum(result.removed for result in results) + vanished,
        failures=[failure for result in results for failure in result.failures],
    )


def select_published_between(connection: sqlite3.Connection, start: date, end: date) -> list[sqlite3.Row]:
    """Return the listing fields of every row published on a UTC date inside [start, end].

    The bounds are compared as ISO text against the indexed publish column, so
    the query is a plain index range scan. Rows without a publish instant are
    never returned.
    """
    lower = datetime(start.year, start.month, start.day, tzinfo=UTC).isoformat()
    upper = (datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)).isoformat()
    return connection.execute(
        "SELECT file_path, file_channel, id, channel_id, title, published_at, duration "
        "FROM video_metadata WHERE published_at >= ? AND published_at < ? "
        "ORDER BY file_channel, published_at, id",
        (lower, upper),
    ).fetchall()


def count_rows(connection: sqlite3.Connection) -> tuple[int, int]:
    """Return (rows in the table, rows without a publish instant)."""
    row = connection.execute("SELECT COUNT(*), COUNT(published_at) FROM video_metadata").fetchone()
    total = int(row[0])
    return total, total - int(row[1])
