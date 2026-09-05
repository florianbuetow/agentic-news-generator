#!/usr/bin/env python3
"""Mirror every video metadata file into the SQLite metadata database.

Scans the per-channel ``.info.json`` files written by the download pipeline
and stores each one as a row with a column per metadata key plus the exact
file text. The pass is incremental: a file whose mtime and size are unchanged
since the last pass is skipped without being read, and rows whose file is gone
are deleted. Files that cannot be stored are listed at the end and make the
script exit non-zero.
"""

from contextlib import closing
from datetime import UTC, datetime

from src.config import Config
from src.metadata_db import MetadataDbError, count_rows, open_metadata_db, update_metadata_db


def main() -> int:
    """Update the metadata database from the metadata files and print the summary."""
    config = Config.load_default()
    metadata_dir = config.get_data_downloads_metadata_dir()
    video_subdir = config.get_transcription_metadata_video_subdir()
    db_path = config.get_data_downloads_metadata_db_path()

    print(f"Metadata directory: {metadata_dir}")
    print(f"Metadata database:  {db_path}")
    print()
    try:
        with closing(open_metadata_db(db_path)) as connection:
            summary = update_metadata_db(connection, metadata_dir, video_subdir, datetime.now(UTC), print)
            total, undated = count_rows(connection)
    except MetadataDbError as e:
        print(f"ERROR: {e}")
        return 1

    print()
    print(
        f"Channels: {summary.channels} | Files: {summary.files_seen:,} | New: {summary.inserted:,} | "
        f"Changed: {summary.updated:,} | Unchanged: {summary.unchanged:,} | Removed: {summary.removed:,}"
    )
    print(f"Rows in database: {total:,} ({undated:,} without a publish timestamp)")

    if summary.failures:
        print()
        print("--- Failure Summary ---")
        for failure in summary.failures:
            print(f"❌ {failure}")
        print(f"Encountered {len(summary.failures)} metadata file(s) that could not be stored")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
