#!/usr/bin/env python3
"""Read configured URL inbox files and print queue statistics."""

import sys
from datetime import date
from pathlib import Path

from src.config import Config
from src.url_ingestion.classifier import UrlClassifier
from src.url_ingestion.download_pipeline import InboxArchive, UrlDownloadPipeline
from src.url_ingestion.downloader import DownloaderFactory
from src.url_ingestion.normalizer import UrlNormalizer
from src.url_ingestion.queue_reader import UrlInboxQueueReader


def main() -> int:
    """Load production config, read the URL inbox queue, and print summary counts."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"

    try:
        config = Config(config_path)
        normalizer = UrlNormalizer()
        classifier = UrlClassifier()
        summary = UrlInboxQueueReader(config, normalizer, classifier).read_queue()
        downloader_factory = DownloaderFactory.default(config)
        archive = InboxArchive(config, date.today)
        run_summary = UrlDownloadPipeline(config, downloader_factory, archive).run(summary)
    except (FileNotFoundError, KeyError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"total_url_lines_read: {summary.total_url_lines_read}")
    print(f"unique_normalized_url_count: {summary.unique_normalized_url_count}")
    print(f"duplicate_url_entry_count: {summary.duplicate_url_entry_count}")
    print(f"unprocessable_url_count: {len(summary.unprocessable_urls)}")
    for classified_type, count in summary.type_counts.items():
        print(f"{classified_type}: {count}")
    print(f"successful_download_count: {run_summary.successful_download_count}")
    print(f"skipped_download_count: {run_summary.skipped_download_count}")
    print(f"failure_count: {run_summary.failure_count}")
    if run_summary.failures:
        print("\n--- Failure Summary ---", file=sys.stderr)
        for failure in run_summary.failures:
            print(f"{failure.original_url}: {failure.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
