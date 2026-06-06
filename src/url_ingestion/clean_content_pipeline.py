"""URL raw-to-cleaned Markdown processing pipeline."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.config import Config
from src.url_ingestion.formatting import OversizedDocumentError
from src.url_ingestion.raw_processing import RawContentItem, RawContentScanner, RawProcessorFactory


@dataclass(frozen=True)
class RawContentProcessingFailure:
    """One raw content processing failure."""

    raw_path: str
    reason: str


@dataclass(frozen=True)
class CleanContentRunSummary:
    """Summary of a URL clean-content processing run."""

    raw_pending_count: int
    total_pending_count: int
    cleaned_count: int
    skipped_existing_count: int
    oversized_count: int
    failure_count: int
    failures: tuple[RawContentProcessingFailure, ...]


class CleaningErrorLog:
    """Append clean-content failures to a durable daily processing-error log for human review."""

    def __init__(self, config: Config, today_provider: Callable[[], date]) -> None:
        """Initialize the cleaning error log."""
        self._config = config
        self._today_provider = today_provider

    def append_failure(self, raw_path: str, reason: str) -> None:
        """Append one raw path and failure reason as a single processing-error line."""
        errors_dir = self._config.get_url_cleaned_dir() / "errors"
        errors_dir.mkdir(parents=True, exist_ok=True)
        log_path = errors_dir / f"{self._today_provider().isoformat()}.txt"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{self._single_line(raw_path)}\t{self._single_line(reason)}\n")

    def _single_line(self, value: str) -> str:
        """Collapse newlines and whitespace runs so one failure stays on one line."""
        return " ".join(value.split())


def format_eta(seconds_remaining: float) -> str:
    """Format remaining duration as hours and minutes."""
    total_minutes = max(0, int(seconds_remaining // 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours}h{minutes}m"


class UrlCleanContentPipeline:
    """Scan raw URL content and process it into cleaned Markdown."""

    def __init__(self, scanner: RawContentScanner, processor_factory: RawProcessorFactory, error_log: CleaningErrorLog) -> None:
        """Initialize the clean-content pipeline."""
        self._scanner = scanner
        self._processor_factory = processor_factory
        self._error_log = error_log

    def run(
        self,
        *,
        limit: int | None,
        raw_path: Path | None,
        raw_paths: tuple[Path, ...] | None,
        force: bool,
    ) -> CleanContentRunSummary:
        """Run raw content processing and collect failures."""
        scan_result = self._scanner.scan(include_existing_cleaned=force)
        pending_items = select_pending_items(scan_result.pending_items, limit=limit, raw_path=raw_path, raw_paths=raw_paths)
        cleaned_count = 0
        oversized_count = 0
        failures: list[RawContentProcessingFailure] = []

        print(f"raw_files_pending: {len(scan_result.pending_items)}", flush=True)
        print(f"raw_files_selected: {len(pending_items)}", flush=True)
        print(f"skipped_existing_cleaned_count: {scan_result.skipped_existing_count}", flush=True)

        total_pending = len(pending_items)
        if len(scan_result.pending_items) == 0:
            print("No raw URL files to clean.", flush=True)
        elif total_pending == 0:
            print("No raw URL files matched the selected processing bounds.", flush=True)
        run_start = time.monotonic()
        for index, item in enumerate(pending_items, start=1):
            completed = index - 1
            elapsed = time.monotonic() - run_start
            avg = (elapsed / completed) if completed > 0 else 0.0
            eta_seconds = avg * (total_pending - completed)
            print(
                f"[{index}/{total_pending}] {index / total_pending * 100:5.1f}% ETA {format_eta(eta_seconds)}  "
                f"{item.content_type}: {item.raw_path}",
                flush=True,
            )
            try:
                started_at = time.monotonic()
                processor = self._processor_factory.create(item.content_type)
                process_result = processor.process(item)
                processing_seconds = time.monotonic() - started_at
                cleaned_count += 1
                print(f"  raw_bytes: {process_result.raw_bytes}", flush=True)
                print(f"  extracted_chars: {process_result.extracted_chars}", flush=True)
                print(f"  prompt_tokens: {process_result.prompt_tokens}", flush=True)
                print(f"  llm_attempts: {process_result.llm_attempts}", flush=True)
                print(f"  formatting_seconds: {process_result.formatting_seconds:.2f}", flush=True)
                print(f"  processing_seconds: {processing_seconds:.2f}", flush=True)
                print(f"  cleaned_chars: {process_result.cleaned_chars}", flush=True)
                if process_result.extraction_type is not None:
                    print(f"  extraction_type: {process_result.extraction_type}", flush=True)
                print(f"  cleaned: {process_result.cleaned_path}", flush=True)
            except OversizedDocumentError as exc:
                oversized_count += 1
                failures.append(RawContentProcessingFailure(raw_path=str(item.raw_path), reason=str(exc)))
                self._error_log.append_failure(str(item.raw_path), str(exc))
                print(f"  oversized: {exc}", flush=True)
            except Exception as exc:
                failures.append(RawContentProcessingFailure(raw_path=str(item.raw_path), reason=str(exc)))
                self._error_log.append_failure(str(item.raw_path), str(exc))
                print(f"  failed: {exc}", flush=True)

        return CleanContentRunSummary(
            raw_pending_count=len(scan_result.pending_items),
            total_pending_count=len(pending_items),
            cleaned_count=cleaned_count,
            skipped_existing_count=scan_result.skipped_existing_count,
            oversized_count=oversized_count,
            failure_count=len(failures),
            failures=tuple(failures),
        )


def select_pending_items(
    pending_items: tuple[RawContentItem, ...],
    *,
    limit: int | None,
    raw_path: Path | None,
    raw_paths: tuple[Path, ...] | None,
) -> tuple[RawContentItem, ...]:
    """Apply optional raw-path and count bounds to pending raw items."""
    selected_items = pending_items
    if raw_path is not None and raw_paths is not None:
        raise ValueError("Use either raw_path or raw_paths, not both")
    if raw_path is not None:
        selected_raw_path = raw_path.expanduser().resolve()
        selected_items = tuple(item for item in selected_items if item.raw_path.resolve() == selected_raw_path)
        if not selected_items:
            raise ValueError(f"Selected raw path is not pending for cleaning: {raw_path}")
    if raw_paths is not None:
        selected_raw_paths = tuple(path.expanduser().resolve() for path in raw_paths)
        selected_raw_path_set = set(selected_raw_paths)
        selected_items = tuple(item for item in selected_items if item.raw_path.resolve() in selected_raw_path_set)
        missing_paths = tuple(path for path in selected_raw_paths if all(item.raw_path.resolve() != path for item in selected_items))
        if missing_paths:
            missing_display = ", ".join(str(path) for path in missing_paths)
            raise ValueError(f"Selected raw paths are not pending for cleaning: {missing_display}")
    if limit is not None:
        selected_items = selected_items[:limit]
    return selected_items
