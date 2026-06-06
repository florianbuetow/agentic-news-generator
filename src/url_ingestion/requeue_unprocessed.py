"""Recover previously unprocessed URL inbox lines under current routing rules."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.config import Config
from src.url_ingestion.classifier import UrlClassifier, UrlContentType
from src.url_ingestion.identity import sanitize_normalized_url_to_stem
from src.url_ingestion.normalizer import NormalizedUrl, UrlNormalizer
from src.url_ingestion.reachability import ReachabilityProbe

RECOVERABLE_TYPES: frozenset[UrlContentType] = frozenset({"html", "markdown", "pdf", "text"})
RAW_SUFFIX_BY_TYPE: dict[UrlContentType, str] = {
    "html": ".html",
    "markdown": ".md",
    "pdf": ".pdf",
    "text": ".txt",
    "unknown": "",
}


@dataclass(frozen=True)
class RequeueSummary:
    """Summary of a previously-unprocessed URL recovery scan."""

    archive_line_count: int
    recoverable_count: int
    duplicate_count: int
    already_downloaded_count: int
    unsupported_count: int
    unprocessable_count: int
    unreachable_count: int
    selected_count: int
    written_count: int
    output_file: Path | None


@dataclass(frozen=True)
class RequeueCandidate:
    """One URL line that can be retried by the current downloader pipeline."""

    source_line: str
    normalized_url: str
    classified_type: UrlContentType


class UnprocessedUrlRequeuer:
    """Scan unprocessed archives and optionally write recoverable URLs back to inbox."""

    def __init__(self, config: Config, normalizer: UrlNormalizer, classifier: UrlClassifier, reachability_probe: ReachabilityProbe) -> None:
        """Initialize the requeuer with explicit processing dependencies."""
        self._config = config
        self._normalizer = normalizer
        self._classifier = classifier
        self._reachability_probe = reachability_probe

    def scan(self) -> tuple[RequeueSummary, tuple[RequeueCandidate, ...]]:
        """Return currently recoverable URL candidates without writing an inbox file."""
        archive_paths = self._archive_paths()
        archive_line_count = 0
        unsupported_count = 0
        unprocessable_count = 0
        already_downloaded_count = 0
        unreachable_count = 0
        candidates_by_normalized_url: dict[str, RequeueCandidate] = {}
        duplicate_count = 0

        for archive_path in archive_paths:
            for raw_line in archive_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                archive_line_count += 1
                source_line = raw_line.partition("\t")[0].strip()
                normalized_result = self._normalizer.normalize(self._extract_url(source_line))
                if not isinstance(normalized_result, NormalizedUrl):
                    unprocessable_count += 1
                    continue
                classified_type = self._classifier.classify(normalized_result.normalized_url)
                if classified_type not in RECOVERABLE_TYPES:
                    unsupported_count += 1
                    continue
                if self._raw_output_exists(normalized_result.normalized_url, classified_type):
                    already_downloaded_count += 1
                    continue
                if normalized_result.normalized_url in candidates_by_normalized_url:
                    duplicate_count += 1
                    continue
                if self._is_definitively_unreachable(normalized_result.normalized_url):
                    unreachable_count += 1
                    continue
                candidates_by_normalized_url[normalized_result.normalized_url] = RequeueCandidate(
                    source_line=source_line,
                    normalized_url=normalized_result.normalized_url,
                    classified_type=classified_type,
                )

        candidates = tuple(sorted(candidates_by_normalized_url.values(), key=lambda candidate: candidate.normalized_url))
        summary = RequeueSummary(
            archive_line_count=archive_line_count,
            recoverable_count=len(candidates),
            duplicate_count=duplicate_count,
            already_downloaded_count=already_downloaded_count,
            unsupported_count=unsupported_count,
            unprocessable_count=unprocessable_count,
            unreachable_count=unreachable_count,
            selected_count=len(candidates),
            written_count=0,
            output_file=None,
        )
        return summary, candidates

    def write(self, today: date, *, limit: int | None, offset: int) -> RequeueSummary:
        """Write recoverable URL lines to a new inbox file and return a summary."""
        summary, candidates = self.scan()
        selected_candidates = select_requeue_candidates(candidates, limit=limit, offset=offset)
        if not selected_candidates:
            return RequeueSummary(
                archive_line_count=summary.archive_line_count,
                recoverable_count=summary.recoverable_count,
                duplicate_count=summary.duplicate_count,
                already_downloaded_count=summary.already_downloaded_count,
                unsupported_count=summary.unsupported_count,
                unprocessable_count=summary.unprocessable_count,
                unreachable_count=summary.unreachable_count,
                selected_count=0,
                written_count=0,
                output_file=None,
            )

        output_file = self._next_output_file(today)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join(candidate.source_line for candidate in selected_candidates) + "\n", encoding="utf-8")
        return RequeueSummary(
            archive_line_count=summary.archive_line_count,
            recoverable_count=summary.recoverable_count,
            duplicate_count=summary.duplicate_count,
            already_downloaded_count=summary.already_downloaded_count,
            unsupported_count=summary.unsupported_count,
            unprocessable_count=summary.unprocessable_count,
            unreachable_count=summary.unreachable_count,
            selected_count=len(selected_candidates),
            written_count=len(selected_candidates),
            output_file=output_file,
        )

    def _archive_paths(self) -> tuple[Path, ...]:
        """Return configured unprocessed archive files."""
        unprocessed_dir = self._config.get_url_inbox_dir() / "unprocessed"
        if not unprocessed_dir.exists():
            return ()
        return tuple(path for path in sorted(unprocessed_dir.glob("*.txt")) if path.is_file())

    def _extract_url(self, line: str) -> str:
        """Return the embedded URL from a category-prefixed inbox line."""
        if line.startswith("http"):
            return line
        if ":http" in line:
            _, _, url_remainder = line.partition(":http")
            return f"http{url_remainder}"
        return line

    def _raw_output_exists(self, normalized_url: str, classified_type: UrlContentType) -> bool:
        """Return whether the URL already has a non-empty raw output file."""
        stem = sanitize_normalized_url_to_stem(normalized_url)
        suffix = RAW_SUFFIX_BY_TYPE[classified_type]
        raw_path = self._config.get_url_raw_dir() / classified_type / f"{stem}{suffix}"
        return raw_path.is_file() and raw_path.stat().st_size > 0

    def _is_definitively_unreachable(self, normalized_url: str) -> bool:
        """Return whether curl proves a URL is permanently gone via DNS failure or HTTP 404/410."""
        result = self._reachability_probe.check(normalized_url)
        if result.error is not None and "could not resolve host" in result.error.lower():
            return True
        return result.http_status in {"404", "410"}

    def _next_output_file(self, today: date) -> Path:
        """Return a recovery inbox path without overwriting an existing file."""
        inbox_dir = self._config.get_url_inbox_dir()
        base_name = f"requeued-unprocessed-{today.isoformat()}"
        candidate_path = inbox_dir / f"{base_name}.txt"
        if not candidate_path.exists():
            return candidate_path
        suffix = 2
        while True:
            candidate_path = inbox_dir / f"{base_name}-{suffix}.txt"
            if not candidate_path.exists():
                return candidate_path
            suffix += 1


def select_requeue_candidates(
    candidates: tuple[RequeueCandidate, ...],
    *,
    limit: int | None,
    offset: int,
) -> tuple[RequeueCandidate, ...]:
    """Apply optional count bounds to recoverable requeue candidates."""
    offset_candidates = candidates[offset:]
    if limit is None:
        return offset_candidates
    return offset_candidates[:limit]
