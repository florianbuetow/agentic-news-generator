"""Tests for recovering previously unprocessed URL archive entries."""

from datetime import date
from pathlib import Path

from src.config import Config
from src.url_ingestion.classifier import UrlClassifier
from src.url_ingestion.normalizer import UrlNormalizer
from src.url_ingestion.reachability import ReachabilityResult
from src.url_ingestion.requeue_unprocessed import RequeueCandidate, UnprocessedUrlRequeuer, select_requeue_candidates
from tests.test_url_queue_reader import write_config


class StubReachabilityProbe:
    """Reachability probe that returns canned results for requeue tests."""

    def __init__(self, results_by_url: dict[str, ReachabilityResult], default_result: ReachabilityResult) -> None:
        """Initialize with per-URL results and a default result for unlisted URLs."""
        self._results_by_url = results_by_url
        self._default_result = default_result

    def check(self, url: str) -> ReachabilityResult:
        """Return the canned reachability result for a URL."""
        return self._results_by_url.get(url, self._default_result)


def reachable_probe() -> StubReachabilityProbe:
    """Return a probe that treats every URL as reachable."""
    return StubReachabilityProbe({}, ReachabilityResult(http_status="200", final_url=None, content_type="text/html"))


def make_requeuer(tmp_path: Path) -> tuple[Config, UnprocessedUrlRequeuer]:
    """Build a requeuer with a test configuration and an all-reachable probe."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    return config, UnprocessedUrlRequeuer(config, UrlNormalizer(), UrlClassifier(), reachable_probe())


def test_requeuer_scans_recoverable_unprocessed_urls_without_writing(tmp_path: Path) -> None:
    """Report recoverable document URLs and leave inbox untouched in dry-run mode."""
    config, requeuer = make_requeuer(tmp_path)
    inbox_dir = config.get_url_inbox_dir()
    unprocessed_dir = inbox_dir / "unprocessed"
    unprocessed_dir.mkdir(parents=True)
    (unprocessed_dir / "2026-06-01.txt").write_text(
        "\n".join(
            [
                "Category:https://example.com/article?utm_source=email\tURL contains a query string",
                "https://example.com/report.pdf\ttransient failure",
                "https://example.com/report.pdf\tduplicate failure",
                "https://example.com/readme.md\tUnsupported URL type for download: markdown",
                "https://example.com/notes.txt\tUnsupported URL type for download: text",
                "https://example.com/image.png\tUnsupported URL type for download: unknown",
                "not a url\tURL contains whitespace",
            ]
        ),
        encoding="utf-8",
    )

    summary, candidates = requeuer.scan()

    assert summary.archive_line_count == 7
    assert summary.recoverable_count == 4
    assert summary.duplicate_count == 1
    assert summary.unsupported_count == 1
    assert summary.unprocessable_count == 1
    assert summary.selected_count == 4
    assert summary.written_count == 0
    assert [candidate.source_line for candidate in candidates] == [
        "Category:https://example.com/article?utm_source=email",
        "https://example.com/notes.txt",
        "https://example.com/readme.md",
        "https://example.com/report.pdf",
    ]
    assert not (inbox_dir / "requeued-unprocessed-2026-06-01.txt").exists()


def test_requeuer_skips_already_downloaded_raw_outputs(tmp_path: Path) -> None:
    """Avoid requeueing URLs that already have a non-empty raw file."""
    config, requeuer = make_requeuer(tmp_path)
    unprocessed_dir = config.get_url_inbox_dir() / "unprocessed"
    unprocessed_dir.mkdir(parents=True)
    (unprocessed_dir / "2026-06-01.txt").write_text(
        "\n".join(
            [
                "https://example.com/report.pdf\ttransient failure",
                "https://example.com/readme.md\ttransient failure",
                "https://example.com/notes.txt\ttransient failure",
            ]
        ),
        encoding="utf-8",
    )
    raw_pdf = config.get_url_raw_dir() / "pdf" / "https_example_com_report_pdf.pdf"
    raw_pdf.parent.mkdir(parents=True)
    raw_pdf.write_bytes(b"%PDF already downloaded")
    raw_markdown = config.get_url_raw_dir() / "markdown" / "https_example_com_readme_md.md"
    raw_markdown.parent.mkdir(parents=True)
    raw_markdown.write_text("# Already downloaded", encoding="utf-8")
    raw_text = config.get_url_raw_dir() / "text" / "https_example_com_notes_txt.txt"
    raw_text.parent.mkdir(parents=True)
    raw_text.write_text("Already downloaded", encoding="utf-8")

    summary, candidates = requeuer.scan()

    assert summary.recoverable_count == 0
    assert summary.already_downloaded_count == 3
    assert candidates == ()


def test_requeuer_skips_definitively_unreachable_urls(tmp_path: Path) -> None:
    """Skip candidates that curl proves are gone via DNS failure or HTTP 404/410."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    unprocessed_dir = config.get_url_inbox_dir() / "unprocessed"
    unprocessed_dir.mkdir(parents=True)
    (unprocessed_dir / "2026-06-01.txt").write_text(
        "\n".join(
            [
                "https://gone.example.com/a.html\tHTTP 404",
                "https://dns-dead.example.com/b.html\tERR_NAME_NOT_RESOLVED",
                "https://live.example.com/c.html\ttransient failure",
            ]
        ),
        encoding="utf-8",
    )
    probe = StubReachabilityProbe(
        {
            "https://gone.example.com/a.html": ReachabilityResult(http_status="404", final_url=None, content_type="text/html"),
            "https://dns-dead.example.com/b.html": ReachabilityResult(
                http_status=None, final_url=None, content_type=None, error="curl: (6) Could not resolve host: dns-dead.example.com"
            ),
        },
        ReachabilityResult(http_status="200", final_url=None, content_type="text/html"),
    )
    requeuer = UnprocessedUrlRequeuer(config, UrlNormalizer(), UrlClassifier(), probe)

    summary, candidates = requeuer.scan()

    assert summary.unreachable_count == 2
    assert summary.recoverable_count == 1
    assert [candidate.normalized_url for candidate in candidates] == ["https://live.example.com/c.html"]


def test_requeuer_writes_recovery_inbox_file(tmp_path: Path) -> None:
    """Write recoverable URLs to a deterministic new inbox file."""
    config, requeuer = make_requeuer(tmp_path)
    inbox_dir = config.get_url_inbox_dir()
    unprocessed_dir = inbox_dir / "unprocessed"
    unprocessed_dir.mkdir(parents=True)
    (unprocessed_dir / "2026-06-01.txt").write_text("https://example.com/article?utm_source=email\told failure\n", encoding="utf-8")

    summary = requeuer.write(date(2026, 6, 1), limit=None, offset=0)

    output_file = inbox_dir / "requeued-unprocessed-2026-06-01.txt"
    assert summary.written_count == 1
    assert summary.selected_count == 1
    assert summary.output_file == output_file
    assert output_file.read_text(encoding="utf-8") == "https://example.com/article?utm_source=email\n"


def test_requeuer_writes_limited_recovery_inbox_file(tmp_path: Path) -> None:
    """Write only the requested number of recoverable candidates."""
    config, requeuer = make_requeuer(tmp_path)
    inbox_dir = config.get_url_inbox_dir()
    unprocessed_dir = inbox_dir / "unprocessed"
    unprocessed_dir.mkdir(parents=True)
    (unprocessed_dir / "2026-06-01.txt").write_text(
        "\n".join(
            [
                "https://example.com/a.html\told failure",
                "https://example.com/b.html\told failure",
            ]
        ),
        encoding="utf-8",
    )

    summary = requeuer.write(date(2026, 6, 1), limit=1, offset=0)

    output_file = inbox_dir / "requeued-unprocessed-2026-06-01.txt"
    assert summary.recoverable_count == 2
    assert summary.selected_count == 1
    assert summary.written_count == 1
    assert output_file.read_text(encoding="utf-8") == "https://example.com/a.html\n"


def test_requeuer_writes_offset_limited_recovery_inbox_file(tmp_path: Path) -> None:
    """Skip earlier candidates when writing a bounded recovery inbox."""
    config, requeuer = make_requeuer(tmp_path)
    inbox_dir = config.get_url_inbox_dir()
    unprocessed_dir = inbox_dir / "unprocessed"
    unprocessed_dir.mkdir(parents=True)
    (unprocessed_dir / "2026-06-01.txt").write_text(
        "\n".join(
            [
                "https://example.com/a.html\told failure",
                "https://example.com/b.html\told failure",
            ]
        ),
        encoding="utf-8",
    )

    summary = requeuer.write(date(2026, 6, 1), limit=1, offset=1)

    assert summary.recoverable_count == 2
    assert summary.selected_count == 1
    assert summary.written_count == 1
    assert (inbox_dir / "requeued-unprocessed-2026-06-01.txt").read_text(encoding="utf-8") == "https://example.com/b.html\n"


def test_requeuer_write_avoids_overwriting_existing_recovery_file(tmp_path: Path) -> None:
    """Create a numbered recovery inbox file when today's file already exists."""
    config, requeuer = make_requeuer(tmp_path)
    inbox_dir = config.get_url_inbox_dir()
    inbox_dir.mkdir(parents=True, exist_ok=True)
    (inbox_dir / "requeued-unprocessed-2026-06-01.txt").write_text("existing\n", encoding="utf-8")
    unprocessed_dir = inbox_dir / "unprocessed"
    unprocessed_dir.mkdir(parents=True)
    (unprocessed_dir / "2026-06-01.txt").write_text("https://example.com/a.html\told failure\n", encoding="utf-8")

    summary = requeuer.write(date(2026, 6, 1), limit=None, offset=0)

    assert summary.output_file == inbox_dir / "requeued-unprocessed-2026-06-01-2.txt"
    assert summary.output_file is not None
    assert (inbox_dir / "requeued-unprocessed-2026-06-01.txt").read_text(encoding="utf-8") == "existing\n"
    assert summary.output_file.read_text(encoding="utf-8") == "https://example.com/a.html\n"


def test_select_requeue_candidates_limits_candidates() -> None:
    """Select a bounded prefix for smoke retries."""
    candidates = (
        RequeueCandidate("https://example.com/a.html", "https://example.com/a.html", "html"),
        RequeueCandidate("https://example.com/b.html", "https://example.com/b.html", "html"),
    )

    assert select_requeue_candidates(candidates, limit=1, offset=0) == candidates[:1]
    assert select_requeue_candidates(candidates, limit=1, offset=1) == candidates[1:]
    assert select_requeue_candidates(candidates, limit=None, offset=0) == candidates
