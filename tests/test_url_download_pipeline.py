"""Integration tests for the URL download pipeline."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import yaml

from src.config import Config
from src.url_ingestion.classifier import UrlClassifier
from src.url_ingestion.download_pipeline import InboxArchive, UrlDownloadPipeline
from src.url_ingestion.downloader import DownloaderFactory, RenderedHtml
from src.url_ingestion.metadata import MetadataHelper
from src.url_ingestion.normalizer import UrlNormalizer
from src.url_ingestion.queue_reader import UrlInboxQueueReader
from src.url_ingestion.reachability import ReachabilityResult
from tests.test_url_downloader import FakeHtmlRenderer, FakeHttpClient, FakeResponse


def write_config(config_path: Path, data_dir: Path) -> Config:
    """Write a minimal URL ingestion config."""
    paths = {
        "data_dir": str(data_dir),
        "data_models_dir": str(data_dir / "models"),
        "data_downloads_dir": str(data_dir / "downloads"),
        "data_downloads_videos_dir": str(data_dir / "downloads" / "videos"),
        "data_downloads_transcripts_dir": str(data_dir / "downloads" / "transcripts"),
        "data_downloads_transcripts_hallucinations_dir": str(data_dir / "downloads" / "transcripts-hallucinations"),
        "data_downloads_transcripts_cleaned_dir": str(data_dir / "downloads" / "transcripts_cleaned"),
        "data_downloads_transcripts_summaries_dir": str(data_dir / "downloads" / "transcripts_summaries"),
        "data_downloads_audio_dir": str(data_dir / "downloads" / "audio"),
        "data_downloads_metadata_dir": str(data_dir / "downloads" / "metadata"),
        "data_output_dir": str(data_dir / "output"),
        "data_input_dir": str(data_dir / "input"),
        "data_temp_dir": str(data_dir / "temp"),
        "data_archive_dir": str(data_dir / "archive"),
        "data_archive_videos_dir": str(data_dir / "archive" / "videos"),
        "data_logs_dir": str(data_dir / "logs"),
        "reports_dir": "reports",
    }
    config_data: dict[str, Any] = {
        "paths": paths,
        "channels": [],
        "url_processing": {
            "base_dir": ".test/urls",
            "inbox_dir": "inbox_urls",
            "raw_dir": "data_raw",
            "cleaned_dir": "data_cleaned",
            "test_base_dir": ".test/urls",
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return Config(config_path)


def read_queue(config: Config):
    """Read the configured URL test queue."""
    return UrlInboxQueueReader(config, UrlNormalizer(), UrlClassifier()).read_queue()


def fixed_today() -> date:
    """Return a fixed archive date for deterministic tests."""
    return date(2026, 5, 31)


def make_pipeline(config: Config, http_client: FakeHttpClient, html_renderer: FakeHtmlRenderer) -> UrlDownloadPipeline:
    """Create a URL download pipeline with fake download dependencies."""
    factory = DownloaderFactory(config=config, http_client=http_client, html_renderer=html_renderer)
    archive = InboxArchive(config, fixed_today)
    return UrlDownloadPipeline(config, factory, archive, reachability_probe=None)


class QuietSimpleHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses per-request stderr logging in tests."""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs."""


class FailingHtmlRenderer:
    """HTML renderer that always fails."""

    def render(self, url: str) -> RenderedHtml:
        """Raise a deterministic failure."""
        raise RuntimeError("renderer blocked")


class FakeReachabilityProbe:
    """Reachability probe with deterministic diagnostics."""

    def __init__(self) -> None:
        """Initialize captured URLs."""
        self.urls: list[str] = []

    def check(self, url: str) -> ReachabilityResult:
        """Record and return a fake diagnostic."""
        self.urls.append(url)
        return ReachabilityResult(http_status="403", final_url=url, content_type="text/html")


@contextmanager
def serve_directory(directory: Path) -> Iterator[str]:
    """Serve a directory over localhost HTTP for real downloader integration tests."""
    handler_class = partial(QuietSimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_download_pipeline_downloads_real_local_http_pdf_and_rendered_html(tmp_path: Path) -> None:
    """Download real localhost PDF/HTML responses using production downloader dependencies."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    pdf_bytes = b"%PDF-1.4\n% local integration fixture\n"
    (server_root / "report.pdf").write_bytes(pdf_bytes)
    (server_root / "article.html").write_text(
        """
        <!doctype html>
        <html>
          <head><title>Local article</title></head>
          <body>
            <main id="article">Initial article text.</main>
            <script>
              document.getElementById("article").insertAdjacentHTML("beforeend", "<p>Rendered by JavaScript.</p>");
            </script>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    inbox_dir = config.get_url_inbox_dir()
    inbox_dir.mkdir(parents=True)

    with serve_directory(server_root) as base_url:
        inbox_file = inbox_dir / "links.txt"
        inbox_file.write_text(f"{base_url}/report.pdf\n{base_url}/article.html\n", encoding="utf-8")

        summary = UrlDownloadPipeline(
            config,
            DownloaderFactory.default(config),
            InboxArchive(config, fixed_today),
            reachability_probe=None,
        ).run(read_queue(config))

        host_stem = f"http_127_0_0_1_{base_url.rsplit(':', maxsplit=1)[1]}"
        pdf_path = config.get_url_raw_dir() / "pdf" / f"{host_stem}_report_pdf.pdf"
        html_path = config.get_url_raw_dir() / "html" / f"{host_stem}_article_html.html"
        assert pdf_path.read_bytes() == pdf_bytes
        html_text = html_path.read_text(encoding="utf-8")
        assert "Initial article text." in html_text
        assert "Rendered by JavaScript." in html_text
        assert MetadataHelper.load(pdf_path.with_name(f"{pdf_path.stem}.metadata.json")).metadata.http_status == 200
        assert MetadataHelper.load(html_path.with_name(f"{html_path.stem}.metadata.json")).metadata.http_status == 200
        assert summary.successful_download_count == 2
        assert summary.skipped_download_count == 0
        assert summary.failure_count == 0
        assert not inbox_file.exists()


def test_download_pipeline_creates_raw_pdf_html_metadata_archives_and_removes_unchanged_inbox(tmp_path: Path) -> None:
    """Download PDF and HTML URLs into raw folders with metadata and done archive entries."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    inbox_dir = config.get_url_inbox_dir()
    inbox_dir.mkdir(parents=True)
    inbox_file = inbox_dir / "links.txt"
    inbox_file.write_text("https://example.com/report.pdf\nhttps://example.com/article.html\n", encoding="utf-8")
    http_client = FakeHttpClient(FakeResponse(content=b"%PDF fake", status_code=200, url="https://cdn.example.com/report.pdf"))
    html_renderer = FakeHtmlRenderer(
        RenderedHtml(html="<html><body>Rendered article</body></html>", final_url="https://www.example.com/article.html", http_status=200)
    )

    summary = make_pipeline(config, http_client, html_renderer).run(read_queue(config))

    pdf_path = config.get_url_raw_dir() / "pdf" / "https_example_com_report_pdf.pdf"
    html_path = config.get_url_raw_dir() / "html" / "https_example_com_article_html.html"
    assert pdf_path.read_bytes() == b"%PDF fake"
    assert html_path.read_text(encoding="utf-8") == "<html><body>Rendered article</body></html>"
    assert MetadataHelper.load(pdf_path.with_name("https_example_com_report_pdf.metadata.json")).metadata.http_status == 200
    assert (
        MetadataHelper.load(html_path.with_name("https_example_com_article_html.metadata.json")).metadata.final_url
        == "https://www.example.com/article.html"
    )
    done_path = inbox_dir / "done" / "2026-05-31.txt"
    assert done_path.read_text(encoding="utf-8").splitlines() == [
        "https://example.com/article.html",
        "https://example.com/report.pdf",
    ]
    assert not inbox_file.exists()
    assert summary.successful_download_count == 2
    assert summary.skipped_download_count == 0
    assert summary.failure_count == 0


def test_download_pipeline_second_run_skips_existing_raw_files(tmp_path: Path) -> None:
    """Skip already-downloaded raw files on a second run."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    inbox_dir = config.get_url_inbox_dir()
    inbox_dir.mkdir(parents=True)
    inbox_file = inbox_dir / "links.txt"
    inbox_file.write_text("https://example.com/report.pdf\nhttps://example.com/article.html\n", encoding="utf-8")
    first_http_client = FakeHttpClient(FakeResponse(content=b"%PDF fake", status_code=200, url="https://cdn.example.com/report.pdf"))
    first_html_renderer = FakeHtmlRenderer(
        RenderedHtml(html="<html><body>Rendered article</body></html>", final_url="https://www.example.com/article.html", http_status=200)
    )
    make_pipeline(config, first_http_client, first_html_renderer).run(read_queue(config))
    inbox_file.write_text("https://example.com/report.pdf\nhttps://example.com/article.html\n", encoding="utf-8")
    second_http_client = FakeHttpClient(FakeResponse(content=b"new pdf", status_code=200, url="https://cdn.example.com/report.pdf"))
    second_html_renderer = FakeHtmlRenderer(
        RenderedHtml(html="<html>new</html>", final_url="https://example.com/article.html", http_status=200)
    )

    second_summary = make_pipeline(config, second_http_client, second_html_renderer).run(read_queue(config))

    assert second_summary.successful_download_count == 0
    assert second_summary.skipped_download_count == 2
    assert second_summary.failure_count == 0
    assert second_http_client.calls == []
    assert second_html_renderer.calls == []


def test_download_pipeline_archives_unprocessed_source_lines_and_keeps_changed_inbox_file(tmp_path: Path) -> None:
    """Archive unprocessed pre-normalization source lines and avoid removing changed inbox files."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    inbox_dir = config.get_url_inbox_dir()
    inbox_dir.mkdir(parents=True)
    inbox_file = inbox_dir / "links.txt"
    inbox_file.write_text(
        "Docs->Markdown:https://example.com/readme.md\nAsset->Image:https://example.com/image.png\n",
        encoding="utf-8",
    )
    queue_summary = read_queue(config)
    inbox_file.write_text(
        "Docs->Markdown:https://example.com/readme.md\nAsset->Image:https://example.com/image.png\nhttps://example.com/new.pdf\n",
        encoding="utf-8",
    )
    http_client = FakeHttpClient(FakeResponse(content=b"%PDF fake", status_code=200, url="https://cdn.example.com/report.pdf"))
    html_renderer = FakeHtmlRenderer(RenderedHtml(html="<html></html>", final_url="https://example.com/article.html", http_status=200))

    summary = make_pipeline(config, http_client, html_renderer).run(queue_summary)

    unprocessed_path = inbox_dir / "unprocessed" / "2026-05-31.txt"
    assert unprocessed_path.read_text(encoding="utf-8").splitlines() == [
        "Asset->Image:https://example.com/image.png\tUnsupported URL type for download: unknown",
    ]
    markdown_path = config.get_url_raw_dir() / "markdown" / "https_example_com_readme_md.md"
    assert markdown_path.read_bytes() == b"%PDF fake"
    assert inbox_file.exists()
    assert summary.successful_download_count == 1
    assert summary.unprocessed_count == 1
    assert summary.failure_count == 0


def test_download_pipeline_adds_reachability_diagnostic_to_download_failures(tmp_path: Path) -> None:
    """Append curl-style reachability context when a downloader fails."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    inbox_dir = config.get_url_inbox_dir()
    inbox_dir.mkdir(parents=True)
    inbox_file = inbox_dir / "links.txt"
    inbox_file.write_text("https://example.com/article.html\n", encoding="utf-8")
    probe = FakeReachabilityProbe()
    pipeline = UrlDownloadPipeline(
        config,
        DownloaderFactory(config=config, http_client=None, html_renderer=FailingHtmlRenderer()),
        InboxArchive(config, fixed_today),
        probe,
    )

    summary = pipeline.run(read_queue(config))

    assert probe.urls == ["https://example.com/article.html"]
    assert summary.failure_count == 1
    assert summary.failures[0].reason == (
        "renderer blocked | curl_http_status=403, curl_content_type=text/html, curl_final_url=https://example.com/article.html"
    )
    assert (inbox_dir / "unprocessed" / "2026-05-31.txt").read_text(encoding="utf-8").splitlines() == [
        "https://example.com/article.html\trenderer blocked | curl_http_status=403, curl_content_type=text/html, "
        "curl_final_url=https://example.com/article.html"
    ]
    assert not inbox_file.exists()
