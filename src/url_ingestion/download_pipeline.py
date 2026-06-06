"""Batch URL download pipeline."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

from src.config import Config
from src.url_ingestion.downloader import Downloader, DownloaderFactory, DownloadResult, NonHtmlContentError, UnsupportedUrlTypeError
from src.url_ingestion.metadata import Metadata, MetadataHelper
from src.url_ingestion.normalizer import UnprocessableUrl
from src.url_ingestion.queue_reader import InboxFileSnapshot, QueuedUrl, UrlQueueSummary
from src.url_ingestion.reachability import ReachabilityProbe


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
    unprocessed_count: int
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

    def append_unprocessed(self, source_line: str, reason: str) -> None:
        """Append an unprocessed source inbox line and reason as a single archive line."""
        self._append_line("unprocessed", f"{self._single_line(source_line)}\t{self._single_line(reason)}")

    def _single_line(self, value: str) -> str:
        """Collapse newlines and whitespace runs so one archived outcome stays on one line."""
        return " ".join(value.split())

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
        reachability_probe: ReachabilityProbe | None,
    ) -> None:
        """Initialize the URL download pipeline."""
        self._config = config
        self._downloader_factory = downloader_factory
        self._archive = archive
        self._reachability_probe = reachability_probe

    def run(self, queue_summary: UrlQueueSummary) -> UrlDownloadRunSummary:
        """Run downloads for a prepared queue summary."""
        failures: list[UrlDownloadFailure] = []
        successful_download_count = 0
        unprocessed_count = 0

        for unprocessable_url in queue_summary.unprocessable_urls:
            self._record_unprocessable(unprocessable_url)
            unprocessed_count += 1
            self._emit(f"unprocessable: {unprocessable_url.original_url} | {unprocessable_url.reason}")

        pending_urls = tuple(url for url in queue_summary.queued_urls if not self._is_already_downloaded(url))
        skipped_download_count = len(queue_summary.queued_urls) - len(pending_urls)
        if skipped_download_count:
            self._emit(f"Skipping {skipped_download_count} previously downloaded urls.")
        if not queue_summary.queued_urls:
            self._emit("No queued URLs to download.")

        total_pending = len(pending_urls)
        for index, queued_url in enumerate(pending_urls, start=1):
            progress = f"[{index}/{total_pending}]"
            try:
                downloader = self._downloader_factory.create(queued_url.classified_type)
            except UnsupportedUrlTypeError as exc:
                unprocessed_count += 1
                self._archive.append_unprocessed(self._failure_source_line(queued_url), str(exc))
                self._emit(f"{progress} Unsupported type, archived for review: {queued_url.normalized_url}")
            else:
                self._emit(f"{progress} Processing: {queued_url.normalized_url}")
                try:
                    rerouted = False
                    try:
                        download_result = self._download_and_record(queued_url, downloader)
                    except NonHtmlContentError:
                        rerouted = True
                        self._emit("  re-routing to PDF (served non-HTML content)...")
                        pdf_downloader = self._downloader_factory.create("pdf")
                        download_result = self._download_and_record(replace(queued_url, classified_type="pdf"), pdf_downloader)
                    successful_download_count += 1
                    size = download_result.raw_path.stat().st_size
                    resolved_type = "pdf (re-routed from html)" if rerouted else queued_url.classified_type
                    self._emit(f"  done: {resolved_type} ({size:,} bytes)")
                except Exception as exc:
                    reason = " ".join(self._failure_reason_with_reachability(queued_url, str(exc)).split())
                    failures.append(UrlDownloadFailure(original_url=self._failure_source_line(queued_url), reason=reason))
                    self._archive.append_unprocessed(self._failure_source_line(queued_url), reason)
                    self._emit(f"  failed: {reason}")

        self._archive.remove_unchanged_files(queue_summary.inbox_file_snapshots)
        return UrlDownloadRunSummary(
            queue_summary=queue_summary,
            successful_download_count=successful_download_count,
            skipped_download_count=skipped_download_count,
            unprocessed_count=unprocessed_count,
            failure_count=len(failures),
            failures=tuple(failures),
        )

    def _is_already_downloaded(self, queued_url: QueuedUrl) -> bool:
        """Return whether a supported URL already has a non-empty raw file on disk."""
        try:
            downloader = self._downloader_factory.create(queued_url.classified_type)
        except UnsupportedUrlTypeError:
            return False
        return downloader.already_downloaded(queued_url)

    def _download_and_record(self, queued_url: QueuedUrl, downloader: Downloader) -> DownloadResult:
        """Download one queued URL with the given downloader, persist metadata, and archive the source line."""
        download_result = downloader.download(queued_url)
        self._save_metadata(
            queued_url, download_result.raw_path, download_result.final_url, download_result.http_status, download_result.status
        )
        self._archive.append_done(queued_url.original_url)
        return download_result

    def _emit(self, message: str) -> None:
        """Report URL download progress."""
        print(message, flush=True)

    def _record_unprocessable(self, unprocessable_url: UnprocessableUrl) -> UrlDownloadFailure:
        """Record one normalization or collision failure."""
        failure = UrlDownloadFailure(original_url=unprocessable_url.original_url, reason=unprocessable_url.reason)
        self._archive.append_unprocessed(unprocessable_url.original_url, unprocessable_url.reason)
        return failure

    def _failure_source_line(self, queued_url: QueuedUrl) -> str:
        """Return the pre-normalization inbox line to archive for failures."""
        return queued_url.source_line or queued_url.original_url

    def _failure_reason_with_reachability(self, queued_url: QueuedUrl, reason: str) -> str:
        """Append curl-style reachability context after a download failure."""
        if self._reachability_probe is None:
            return reason
        diagnostic = self._reachability_probe.check(queued_url.normalized_url)
        return f"{reason} | {diagnostic.summary()}"

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
