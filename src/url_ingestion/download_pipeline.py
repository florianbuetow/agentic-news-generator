"""Batch URL download pipeline."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from src.config import Config
from src.url_ingestion.downloader import DownloaderFactory, UnsupportedUrlTypeError
from src.url_ingestion.metadata import Metadata, MetadataHelper
from src.url_ingestion.normalizer import UnprocessableUrl
from src.url_ingestion.queue_reader import InboxFileSnapshot, QueuedUrl, UrlQueueSummary


@dataclass(frozen=True)
class UrlDownloadFailure:
    """One URL download or routing failure."""

    original_url: str
    reason: str


@dataclass(frozen=True)
class UrlDownloadRunSummary:
    """Summary of a URL download batch run."""

    queue_summary: UrlQueueSummary
    successful_download_count: int
    skipped_download_count: int
    failure_count: int
    failures: tuple[UrlDownloadFailure, ...]


class InboxArchive:
    """Append URL processing outcomes and remove unchanged consumed inbox files."""

    def __init__(self, config: Config, today_provider: Callable[[], date]) -> None:
        """Initialize the archive helper."""
        self._config = config
        self._today_provider = today_provider

    def append_done(self, original_url: str) -> None:
        """Append a successful original URL to today's done archive."""
        self._append_line("done", original_url)

    def append_unprocessed(self, original_url: str, reason: str) -> None:
        """Append an unprocessed original URL and reason to today's archive."""
        self._append_line("unprocessed", f"{original_url}\t{reason}")

    def remove_unchanged_files(self, snapshots: tuple[InboxFileSnapshot, ...]) -> None:
        """Remove consumed inbox files only when their content hash is unchanged."""
        for snapshot in snapshots:
            if not snapshot.path.exists():
                continue
            if self._hash_file(snapshot.path) == snapshot.content_hash:
                snapshot.path.unlink()

    def _append_line(self, archive_dir_name: str, line: str) -> None:
        """Append one line to a daily archive file."""
        archive_dir = self._config.get_url_inbox_dir() / archive_dir_name
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{self._today_provider().isoformat()}.txt"
        with archive_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def _hash_file(self, path: Path) -> str:
        """Calculate the current content hash for an inbox file."""
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class UrlDownloadPipeline:
    """Download queued URLs, write metadata, and archive inbox outcomes."""

    def __init__(
        self,
        config: Config,
        downloader_factory: DownloaderFactory,
        archive: InboxArchive,
    ) -> None:
        """Initialize the URL download pipeline."""
        self._config = config
        self._downloader_factory = downloader_factory
        self._archive = archive

    def run(self, queue_summary: UrlQueueSummary) -> UrlDownloadRunSummary:
        """Run downloads for a prepared queue summary."""
        failures: list[UrlDownloadFailure] = []
        successful_download_count = 0
        skipped_download_count = 0

        for unprocessable_url in queue_summary.unprocessable_urls:
            failure = self._record_unprocessable(unprocessable_url)
            failures.append(failure)

        for queued_url in queue_summary.queued_urls:
            try:
                downloader = self._downloader_factory.create(queued_url.classified_type)
                download_result = downloader.download(queued_url)
                self._save_metadata(
                    queued_url, download_result.raw_path, download_result.final_url, download_result.http_status, download_result.status
                )
                self._archive.append_done(queued_url.original_url)
                if download_result.status == "skipped_existing":
                    skipped_download_count += 1
                else:
                    successful_download_count += 1
            except UnsupportedUrlTypeError as exc:
                failure = UrlDownloadFailure(original_url=queued_url.original_url, reason=str(exc))
                failures.append(failure)
                self._archive.append_unprocessed(queued_url.original_url, failure.reason)
            except Exception as exc:
                failure = UrlDownloadFailure(original_url=queued_url.original_url, reason=str(exc))
                failures.append(failure)
                self._archive.append_unprocessed(queued_url.original_url, failure.reason)

        self._archive.remove_unchanged_files(queue_summary.inbox_file_snapshots)
        return UrlDownloadRunSummary(
            queue_summary=queue_summary,
            successful_download_count=successful_download_count,
            skipped_download_count=skipped_download_count,
            failure_count=len(failures),
            failures=tuple(failures),
        )

    def _record_unprocessable(self, unprocessable_url: UnprocessableUrl) -> UrlDownloadFailure:
        """Record one normalization or collision failure."""
        failure = UrlDownloadFailure(original_url=unprocessable_url.original_url, reason=unprocessable_url.reason)
        self._archive.append_unprocessed(unprocessable_url.original_url, unprocessable_url.reason)
        return failure

    def _save_metadata(
        self,
        queued_url: QueuedUrl,
        raw_path: Path,
        final_url: str | None,
        http_status: int | None,
        status: str,
    ) -> None:
        """Write metadata beside a successful raw download."""
        metadata_path = raw_path.with_name(f"{raw_path.stem}.metadata.json")
        metadata = Metadata(
            source_url=queued_url.original_url,
            normalized_url=queued_url.normalized_url,
            final_url=final_url,
            sanitized_url_stem=queued_url.sanitized_url_stem,
            classified_type=queued_url.classified_type,
            downloaded_at=datetime.now(UTC).isoformat(),
            http_status=http_status,
            raw_path=str(raw_path),
            metadata_path=str(metadata_path),
            status=status,
            source_kind="url_download",
        )
        MetadataHelper(metadata).save(metadata_path)
