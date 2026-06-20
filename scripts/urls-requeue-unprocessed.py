#!/usr/bin/env python3
"""Requeue previously unprocessed URL archive entries that are recoverable now."""

import argparse
import sys
from datetime import date

from src.config import Config
from src.url_ingestion.classifier import UrlClassifier
from src.url_ingestion.normalizer import UrlNormalizer
from src.url_ingestion.reachability import CurlReachabilityProbe
from src.url_ingestion.requeue_unprocessed import UnprocessedUrlRequeuer, select_requeue_candidates


def main() -> int:
    """Scan unprocessed URL archives and optionally write a recovery inbox file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write recoverable URLs to a new inbox file")
    parser.add_argument("--limit", type=positive_int, help="limit the number of recoverable URLs to show or write")
    parser.add_argument("--offset", type=non_negative_int, default=0, help="skip this many recoverable URLs before showing or writing")
    args = parser.parse_args()

    config_path = Config.repo_config_path()

    try:
        print(f"Loading config: {config_path}", flush=True)
        config = Config(config_path)
        requeuer = UnprocessedUrlRequeuer(config, UrlNormalizer(), UrlClassifier(), CurlReachabilityProbe())
        if args.write:
            summary = requeuer.write(date.today(), limit=args.limit, offset=args.offset)
            candidates = ()
        else:
            summary, candidates = requeuer.scan()
    except (FileNotFoundError, KeyError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"unprocessed_archive_line_count: {summary.archive_line_count}")
    print(f"recoverable_count: {summary.recoverable_count}")
    print(f"duplicate_count: {summary.duplicate_count}")
    print(f"already_downloaded_count: {summary.already_downloaded_count}")
    print(f"unsupported_count: {summary.unsupported_count}")
    print(f"unprocessable_count: {summary.unprocessable_count}")
    print(f"unreachable_count: {summary.unreachable_count}")
    selected_count = (
        summary.selected_count if args.write else len(select_requeue_candidates(candidates, limit=args.limit, offset=args.offset))
    )
    print(f"selected_count: {selected_count}")
    print(f"written_count: {summary.written_count}")
    if summary.output_file is not None:
        print(f"output_file: {summary.output_file}")
    if not args.write:
        selected_candidates = select_requeue_candidates(candidates, limit=args.limit, offset=args.offset)
        for index, candidate in enumerate(selected_candidates, start=1):
            print(f"candidate_{index}: {candidate.classified_type} {candidate.normalized_url}")
    if not args.write and summary.recoverable_count:
        print("dry_run: true")
        print("Run with --write to create a recovery inbox file.")
    return 0


def positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed_value


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI argument."""
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed_value


if __name__ == "__main__":
    sys.exit(main())
