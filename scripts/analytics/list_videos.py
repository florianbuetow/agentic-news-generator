#!/usr/bin/env python3
"""List the videos published inside a date window, grouped by channel.

Reads the SQLite metadata database (see ``update-metadata-db``) and prints
every video whose publish date (the metadata ``timestamp``, in UTC) falls
inside the requested window, both bounds included. Channels print
alphabetically; videos inside a channel print oldest first. No metadata file
is read, so the listing is a single indexed query.

The window comes from the words passed by the just target: ``since <date>``,
``<from> <to>``, or ``last <days>``, with dates written as YYYY-MM-DD.
"""

import sys
from contextlib import closing
from datetime import UTC, datetime

from src.analytics.errors import WindowArgumentError
from src.analytics.video_listing import (
    USAGE,
    group_by_channel_chronological,
    parse_window_args,
    records_from_rows,
    render_video_listing,
)
from src.config import Config
from src.metadata_db import MetadataDbError, count_rows, open_metadata_db_readonly, read_updated_at, select_published_between


def main() -> int:
    """Parse the window words, query the metadata database, and print the listing."""
    today = datetime.now(UTC).date()
    try:
        window = parse_window_args(sys.argv[1:], today)
    except WindowArgumentError as e:
        print(f"ERROR: {e}")
        print(USAGE)
        return 1

    config = Config.load_default()
    db_path = config.get_data_downloads_metadata_db_path()
    try:
        with closing(open_metadata_db_readonly(db_path)) as connection:
            updated_at = read_updated_at(connection)
            total, undated = count_rows(connection)
            rows = [dict(row) for row in select_published_between(connection, window.start, window.end)]
    except MetadataDbError as e:
        print(f"ERROR: {e}")
        print("Run 'just update-metadata-db' to build the metadata database first.")
        return 1

    records, failures = records_from_rows(rows)
    groups = group_by_channel_chronological(records)
    print(f"Metadata database: {db_path} (last updated: {updated_at})")
    print(f"Videos in database: {total:,} ({undated:,} without publish date) | Today (UTC): {today.isoformat()} | Publish times are UTC")
    print()
    print(render_video_listing(groups, window))

    if failures:
        print()
        print("--- Failure Summary ---")
        for failure in failures:
            print(f"❌ {failure}")
        print(f"Encountered {len(failures)} database row(s) missing a required field")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
