#!/usr/bin/env python3
"""Download raw content from the configured URL inbox."""

import sys
from datetime import date
from pathlib import Path

from src.config import Config
from src.url_ingestion.classifier import UrlClassifier
from src.url_ingestion.download_pipeline import InboxArchive, UrlDownloadPipeline
from src.url_ingestion.downloader import DownloaderFactory
from src.url_ingestion.normalizer import UrlNormalizer
from src.url_ingestion.queue_reader import UrlInboxQueueReader
from src.url_ingestion.reachability import CurlReachabilityProbe


def main() -> int:
    """Load production config, read the URL inbox queue, and print summary counts."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"

    try:
        print(f"Loading config: {config_path}", flush=True)
        config = Config(config_path)
        print(f"URL inbox directory: {config.get_url_inbox_dir()}", flush=True)
        print(f"Raw URL output directory: {config.get_url_raw_dir()}", flush=True)

        normalizer = UrlNormalizer()
        classifier = UrlClassifier()
        print("Reading URL inbox queue...", flush=True)
        summary = UrlInboxQueueReader(config, normalizer, classifier).read_queue()
        print(f"Found {len(summary.inbox_file_snapshots)} inbox file(s)", flush=True)
        print(f"Found {summary.total_url_lines_read} URL line(s)", flush=True)
        print(f"Queued {len(summary.queued_urls)} unique classified URL(s)", flush=True)
        print(f"Duplicate URL entries: {summary.duplicate_url_entry_count}", flush=True)
        print(f"Unprocessable URL entries: {len(summary.unprocessable_urls)}", flush=True)
        for classified_type, count in summary.type_counts.items():
            print(f"  {classified_type}: {count}", flush=True)

        downloader_factory = DownloaderFactory.default(config)
        archive = InboxArchive(config, date.today)
        print("Downloading queued URLs...", flush=True)
        run_summary = UrlDownloadPipeline(config, downloader_factory, archive, CurlReachabilityProbe()).run(summary)
    except (FileNotFoundError, KeyError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nSummary:")
    print(f"total_url_lines_read: {summary.total_url_lines_read}")
    print(f"unique_normalized_url_count: {summary.unique_normalized_url_count}")
    print(f"duplicate_url_entry_count: {summary.duplicate_url_entry_count}")
    print(f"unprocessable_url_count: {len(summary.unprocessable_urls)}")
    for classified_type, count in summary.type_counts.items():
        print(f"{classified_type}: {count}")
    print(f"successful_download_count: {run_summary.successful_download_count}")
    print(f"skipped_download_count: {run_summary.skipped_download_count}")
    print(f"unprocessed_count: {run_summary.unprocessed_count}")
    print(f"failure_count: {run_summary.failure_count}")
    if run_summary.failures:
        print("", file=sys.stderr)
        for failure in run_summary.failures:
            print(f"ERROR: {failure.original_url}: {failure.reason}", file=sys.stderr)
        print(f"Encountered {len(run_summary.failures)} errors", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
