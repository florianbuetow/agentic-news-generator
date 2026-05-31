"""URL raw-to-cleaned Markdown processing pipeline."""

from dataclasses import dataclass

from src.url_ingestion.formatting import OversizedDocumentError
from src.url_ingestion.raw_processing import RawContentScanner, RawProcessorFactory


@dataclass(frozen=True)
class RawContentProcessingFailure:
    """One raw content processing failure."""

    raw_path: str
    reason: str


@dataclass(frozen=True)
class CleanContentRunSummary:
    """Summary of a URL clean-content processing run."""

    total_pending_count: int
    cleaned_count: int
    skipped_existing_count: int
    oversized_count: int
    failure_count: int
    failures: tuple[RawContentProcessingFailure, ...]


class UrlCleanContentPipeline:
    """Scan raw URL content and process it into cleaned Markdown."""

    def __init__(self, scanner: RawContentScanner, processor_factory: RawProcessorFactory) -> None:
        """Initialize the clean-content pipeline."""
        self._scanner = scanner
        self._processor_factory = processor_factory

    def run(self) -> CleanContentRunSummary:
        """Run raw content processing and collect failures."""
        scan_result = self._scanner.scan()
        cleaned_count = 0
        oversized_count = 0
        failures: list[RawContentProcessingFailure] = []

        for item in scan_result.pending_items:
            try:
                processor = self._processor_factory.create(item.content_type)
                processor.process(item)
                cleaned_count += 1
            except OversizedDocumentError as exc:
                oversized_count += 1
                failures.append(RawContentProcessingFailure(raw_path=str(item.raw_path), reason=str(exc)))
            except Exception as exc:
                failures.append(RawContentProcessingFailure(raw_path=str(item.raw_path), reason=str(exc)))

        return CleanContentRunSummary(
            total_pending_count=len(scan_result.pending_items),
            cleaned_count=cleaned_count,
            skipped_existing_count=scan_result.skipped_existing_count,
            oversized_count=oversized_count,
            failure_count=len(failures),
            failures=tuple(failures),
        )
