"""URL raw-to-cleaned Markdown processing pipeline."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.config import Config
from src.url_ingestion.formatting import OutputWindowExceededError, OversizedDocumentError
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
    skipped_uncleanable_count: int
    uncleanable_count: int
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


class UncleanableRegistry:
    """Persistent record of raw files that cannot be cleaned within a model output window."""

    def __init__(self, status_path: Path) -> None:
        """Load any previously recorded uncleanable entries from disk."""
        self._status_path = status_path
        self._entries = self._load()

    def _load(self) -> dict[str, dict[str, int]]:
        """Read recorded uncleanable entries, returning an empty map when no file exists."""
        if not self._status_path.is_file():
            return {}
        parsed = json.loads(self._status_path.read_text(encoding="utf-8"))
        return {
            str(raw_name): {"failed_output_window_tokens": int(record["failed_output_window_tokens"])}
            for raw_name, record in parsed.items()
        }

    def is_uncleanable_at(self, raw_name: str, output_window_tokens: int) -> bool:
        """Return whether a raw file already failed at an output window this size or smaller."""
        record = self._entries.get(raw_name)
        if record is None:
            return False
        return output_window_tokens <= record["failed_output_window_tokens"]

    def record(self, raw_name: str, *, output_window_tokens: int) -> None:
        """Persist that a raw file could not be cleaned within an output window, never lowering a recorded window."""
        existing = self._entries.get(raw_name)
        recorded_window = output_window_tokens
        if existing is not None and existing["failed_output_window_tokens"] > recorded_window:
            recorded_window = existing["failed_output_window_tokens"]
        self._entries[raw_name] = {"failed_output_window_tokens": recorded_window}
        self._save()

    def _save(self) -> None:
        """Write the uncleanable entries to disk as sorted JSON for human inspection."""
        self._status_path.parent.mkdir(parents=True, exist_ok=True)
        self._status_path.write_text(json.dumps(self._entries, indent=2, sort_keys=True), encoding="utf-8")


def format_eta(seconds_remaining: float) -> str:
    """Format remaining duration as hours and minutes."""
    total_minutes = max(0, int(seconds_remaining // 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours}h{minutes}m"


class UrlCleanContentPipeline:
    """Scan raw URL content and process it into cleaned Markdown."""

    def __init__(
        self,
        scanner: RawContentScanner,
        processor_factory: RawProcessorFactory,
        error_log: CleaningErrorLog,
        uncleanable_registry: UncleanableRegistry,
        max_output_tokens: int,
    ) -> None:
        """Initialize the clean-content pipeline."""
        self._scanner = scanner
        self._processor_factory = processor_factory
        self._error_log = error_log
        self._uncleanable_registry = uncleanable_registry
        self._max_output_tokens = max_output_tokens

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
        selected_items = select_pending_items(scan_result.pending_items, limit=limit, raw_path=raw_path, raw_paths=raw_paths)

        processable_items: list[RawContentItem] = []
        skipped_uncleanable_count = 0
        for candidate in selected_items:
            if not force and self._uncleanable_registry.is_uncleanable_at(candidate.raw_path.name, self._max_output_tokens):
                skipped_uncleanable_count += 1
                print(f"Skipping (recorded uncleanable until output window grows): {candidate.raw_path}", flush=True)
                continue
            processable_items.append(candidate)

        cleaned_count = 0
        uncleanable_count = 0
        failures: list[RawContentProcessingFailure] = []

        print(f"raw_files_pending: {len(scan_result.pending_items)}", flush=True)
        print(f"raw_files_selected: {len(selected_items)}", flush=True)
        print(f"skipped_existing_cleaned_count: {scan_result.skipped_existing_count}", flush=True)
        print(f"skipped_uncleanable_count: {skipped_uncleanable_count}", flush=True)

        total_pending = len(processable_items)
        if len(scan_result.pending_items) == 0:
            print("No raw URL files to clean.", flush=True)
        elif total_pending == 0:
            print("No raw URL files matched the selected processing bounds.", flush=True)
        run_start = time.monotonic()
        for index, item in enumerate(processable_items, start=1):
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
            except (OversizedDocumentError, OutputWindowExceededError) as exc:
                # Too big for the output window or truncated mid-generation: same outcome - cannot be
                # cleaned at this output window. Record it in the uncleanable ledger and skip it until
                # the window grows, so nothing is silently lost and it is not re-attempted every run.
                item.cleaned_path.unlink(missing_ok=True)
                self._uncleanable_registry.record(item.raw_path.name, output_window_tokens=self._max_output_tokens)
                uncleanable_count += 1
                failures.append(RawContentProcessingFailure(raw_path=str(item.raw_path), reason=str(exc)))
                self._error_log.append_failure(str(item.raw_path), str(exc))
                print(f"  uncleanable: {exc}", flush=True)
            except Exception as exc:
                failures.append(RawContentProcessingFailure(raw_path=str(item.raw_path), reason=str(exc)))
                self._error_log.append_failure(str(item.raw_path), str(exc))
                print(f"  failed: {exc}", flush=True)

        return CleanContentRunSummary(
            raw_pending_count=len(scan_result.pending_items),
            total_pending_count=len(selected_items),
            cleaned_count=cleaned_count,
            skipped_existing_count=scan_result.skipped_existing_count,
            skipped_uncleanable_count=skipped_uncleanable_count,
            uncleanable_count=uncleanable_count,
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
