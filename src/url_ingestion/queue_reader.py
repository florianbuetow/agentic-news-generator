"""Read URL inbox files for the URL ingestion pipeline."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.config import Config
from src.url_ingestion.classifier import UrlClassifier, UrlContentType
from src.url_ingestion.identity import sanitize_normalized_url_to_stem
from src.url_ingestion.normalizer import NormalizedUrl, UnprocessableUrl, UrlNormalizer


@dataclass(frozen=True)
class InboxFileSnapshot:
    """Inbox file path and content hash captured before processing."""

    path: Path
    content_hash: str


@dataclass(frozen=True)
class QueuedUrl:
    """A normalized and classified URL ready for download routing."""

    original_url: str
    normalized_url: str
    sanitized_url_stem: str
    classified_type: UrlContentType


@dataclass(frozen=True)
class UrlQueueSummary:
    """Summary of URL inbox queue contents."""

    total_url_lines_read: int
    unique_normalized_url_count: int
    duplicate_url_entry_count: int
    unique_urls: tuple[str, ...]
    type_counts: dict[UrlContentType, int]
    queued_urls: tuple[QueuedUrl, ...]
    unprocessable_urls: tuple[UnprocessableUrl, ...]
    inbox_file_snapshots: tuple[InboxFileSnapshot, ...]


class UrlInboxQueueReader:
    """Read, merge, deduplicate, and sort configured URL inbox files."""

    def __init__(self, config: Config, normalizer: UrlNormalizer, classifier: UrlClassifier) -> None:
        """Initialize the reader with an explicit configuration object."""
        self._config = config
        self._normalizer = normalizer
        self._classifier = classifier

    def read_queue(self) -> UrlQueueSummary:
        """Read all configured URL inbox files and return deterministic queue statistics."""
        inbox_dir = self._config.get_url_inbox_dir()
        self._validate_configured_folders(inbox_dir)
        inbox_file_snapshots = self._snapshot_inbox_files(inbox_dir)

        url_lines: list[str] = []
        for inbox_file_snapshot in inbox_file_snapshots:
            url_lines.extend(self._read_non_empty_lines(inbox_file_snapshot.path))

        normalized_urls: list[str] = []
        normalized_results_by_url: dict[str, NormalizedUrl] = {}
        unprocessable_urls: list[UnprocessableUrl] = []
        for url_line in url_lines:
            if "http" not in url_line.lower():
                unprocessable_urls.append(UnprocessableUrl(original_url=url_line, reason="line does not contain an http URL"))
                continue
            normalized_result = self._normalizer.normalize(url_line)
            if isinstance(normalized_result, NormalizedUrl):
                normalized_urls.append(normalized_result.normalized_url)
                if normalized_result.normalized_url not in normalized_results_by_url:
                    normalized_results_by_url[normalized_result.normalized_url] = normalized_result
            else:
                unprocessable_urls.append(normalized_result)

        unique_urls = tuple(sorted(set(normalized_urls)))
        total_url_lines_read = len(url_lines)
        unique_normalized_url_count = len(unique_urls)
        duplicate_url_entry_count = len(normalized_urls) - unique_normalized_url_count
        queued_urls, collision_failures = self._build_queued_urls(unique_urls, normalized_results_by_url)
        unprocessable_urls.extend(collision_failures)
        type_counts = self._count_url_types(queued_urls)

        return UrlQueueSummary(
            total_url_lines_read=total_url_lines_read,
            unique_normalized_url_count=unique_normalized_url_count,
            duplicate_url_entry_count=duplicate_url_entry_count,
            unique_urls=unique_urls,
            type_counts=type_counts,
            queued_urls=queued_urls,
            unprocessable_urls=tuple(unprocessable_urls),
            inbox_file_snapshots=inbox_file_snapshots,
        )

    def _validate_configured_folders(self, inbox_dir: Path) -> None:
        """Fail clearly when configured URL queue folders are missing or invalid."""
        base_dir = self._config.get_url_base_dir()
        if not base_dir.exists():
            raise FileNotFoundError(f"Configured URL base folder does not exist: {base_dir}")
        if not base_dir.is_dir():
            raise NotADirectoryError(f"Configured URL base path is not a folder: {base_dir}")
        if not inbox_dir.exists():
            raise FileNotFoundError(f"Configured URL inbox folder does not exist: {inbox_dir}")
        if not inbox_dir.is_dir():
            raise NotADirectoryError(f"Configured URL inbox path is not a folder: {inbox_dir}")

    def _snapshot_inbox_files(self, inbox_dir: Path) -> tuple[InboxFileSnapshot, ...]:
        """Return direct inbox file snapshots, excluding operational archive folders."""
        excluded_inbox_directories = {"done", "unprocessed"}
        inbox_file_snapshots: list[InboxFileSnapshot] = []
        for path in sorted(inbox_dir.iterdir(), key=lambda inbox_path: inbox_path.name):
            if path.is_dir() and path.name in excluded_inbox_directories:
                continue
            if path.is_file():
                inbox_file_snapshots.append(InboxFileSnapshot(path=path, content_hash=self._hash_file(path)))
        return tuple(inbox_file_snapshots)

    def _read_non_empty_lines(self, inbox_file: Path) -> tuple[str, ...]:
        """Read stripped non-empty URL lines from one inbox file."""
        url_lines: list[str] = []
        with inbox_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped_line = line.strip()
                if stripped_line:
                    url_lines.append(self._extract_url(stripped_line))
        return tuple(url_lines)

    def _extract_url(self, line: str) -> str:
        """Return the embedded URL from a category-prefixed inbox line.

        Inbox lines may carry a category path before the URL, for example
        ``Category->Subcategory:https://example.com``. Such a line does not start
        with a scheme and contains ``:http`` before the real URL. Lines that
        already start with a scheme are returned unchanged.
        """
        if line.startswith("http"):
            return line
        if ":http" in line:
            _, _, url_remainder = line.partition(":http")
            return f"http{url_remainder}"
        return line

    def _build_queued_urls(
        self,
        unique_urls: tuple[str, ...],
        normalized_results_by_url: dict[str, NormalizedUrl],
    ) -> tuple[tuple[QueuedUrl, ...], tuple[UnprocessableUrl, ...]]:
        """Build queued URL records and detect sanitized-stem collisions."""
        stem_to_urls: dict[str, list[str]] = {}
        for unique_url in unique_urls:
            sanitized_stem = sanitize_normalized_url_to_stem(unique_url)
            stem_to_urls.setdefault(sanitized_stem, []).append(unique_url)

        queued_urls: list[QueuedUrl] = []
        collision_failures: list[UnprocessableUrl] = []
        for unique_url in unique_urls:
            sanitized_stem = sanitize_normalized_url_to_stem(unique_url)
            normalized_result = normalized_results_by_url[unique_url]
            if len(stem_to_urls[sanitized_stem]) > 1:
                collision_failures.append(
                    UnprocessableUrl(
                        original_url=normalized_result.original_url,
                        reason=f"sanitized URL stem collision: {sanitized_stem}",
                    )
                )
                continue
            queued_urls.append(
                QueuedUrl(
                    original_url=normalized_result.original_url,
                    normalized_url=unique_url,
                    sanitized_url_stem=sanitized_stem,
                    classified_type=self._classifier.classify(unique_url),
                )
            )
        return tuple(queued_urls), tuple(collision_failures)

    def _count_url_types(self, queued_urls: tuple[QueuedUrl, ...]) -> dict[UrlContentType, int]:
        """Count classified URL types in deterministic key order."""
        type_counts = dict.fromkeys(self._classifier.classification_order(), 0)
        for queued_url in queued_urls:
            type_counts[queued_url.classified_type] += 1
        return type_counts

    def _hash_file(self, path: Path) -> str:
        """Calculate a stable hash for an inbox file."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
