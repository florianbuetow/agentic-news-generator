"""Unit tests for URL downloaders and factory routing."""

from pathlib import Path
from typing import Any, cast

import pytest
import requests
import yaml

from src.config import Config
from src.url_ingestion.classifier import UrlContentType
from src.url_ingestion.downloader import (
    CONTENT_FETCH_TIMEOUT_SECONDS,
    HTML_RENDER_WAIT_UNTIL,
    DownloaderFactory,
    HtmlDownloader,
    PdfDownloader,
    PlaywrightHtmlRenderer,
    RenderedHtml,
    TextDocumentDownloader,
    UnsupportedUrlTypeError,
    UrlDownloadError,
)
from src.url_ingestion.queue_reader import QueuedUrl


class FakeResponse:
    """Fake HTTP response for downloader tests."""

    def __init__(self, *, content: bytes, status_code: int, url: str) -> None:
        """Initialize the fake response."""
        self._content = content
        self._status_code = status_code
        self._url = url

    @property
    def status_code(self) -> int:
        """Return the fake status code."""
        return self._status_code

    @property
    def content(self) -> bytes:
        """Return fake response bytes."""
        return self._content

    @property
    def url(self) -> str:
        """Return the fake final URL."""
        return self._url

    def raise_for_status(self) -> None:
        """Raise for failed fake status."""
        if self._status_code >= 400:
            raise requests.HTTPError(f"status {self._status_code}")


class FakeHttpClient:
    """Fake HTTP client for downloader tests."""

    def __init__(self, response: FakeResponse) -> None:
        """Initialize the fake client."""
        self.response = response
        self.calls: list[str] = []
        self.timeouts: list[int] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: int, allow_redirects: bool) -> FakeResponse:
        """Record and return a fake HTTP response."""
        self.calls.append(url)
        self.timeouts.append(timeout)
        return self.response


class FakeHtmlRenderer:
    """Fake HTML renderer for downloader tests."""

    def __init__(self, rendered_html: RenderedHtml) -> None:
        """Initialize the fake renderer."""
        self.rendered_html = rendered_html
        self.calls: list[str] = []

    def render(self, url: str) -> RenderedHtml:
        """Record and return fake rendered HTML."""
        self.calls.append(url)
        return self.rendered_html


class FakePageResponse:
    """Fake Playwright page response."""

    status = 200


class FakePage:
    """Fake Playwright page that records navigation arguments."""

    url = "https://example.com/final"

    def __init__(self) -> None:
        """Initialize captured navigation state."""
        self.goto_kwargs: dict[str, Any] = {}

    def goto(self, url: str, **kwargs: Any) -> FakePageResponse:
        """Record navigation arguments and return a fake response."""
        self.goto_kwargs = kwargs
        return FakePageResponse()

    def evaluate(self, expression: str) -> str:
        """Return fake rendered HTML."""
        return "<html>rendered</html>"


class FakeBrowser:
    """Fake Playwright browser."""

    def __init__(self, page: FakePage) -> None:
        """Initialize with one fake page."""
        self._page = page
        self.closed = False

    def new_context(self, **kwargs: Any) -> "FakeBrowserContext":
        """Return a fake browser context."""
        return FakeBrowserContext(self._page)

    def close(self) -> None:
        """Record browser closure."""
        self.closed = True


class FakeBrowserContext:
    """Fake Playwright browser context."""

    def __init__(self, page: FakePage) -> None:
        """Initialize with one fake page."""
        self._page = page

    def new_page(self) -> FakePage:
        """Return the fake page."""
        return self._page


class FakeChromium:
    """Fake Playwright Chromium launcher."""

    def __init__(self, browser: FakeBrowser) -> None:
        """Initialize with one fake browser."""
        self._browser = browser

    def launch(self, *, headless: bool) -> FakeBrowser:
        """Return the fake browser."""
        return self._browser


class FakePlaywright:
    """Fake Playwright root object."""

    def __init__(self, browser: FakeBrowser) -> None:
        """Initialize fake Chromium access."""
        self.chromium = FakeChromium(browser)


class FakeSyncPlaywright:
    """Fake sync_playwright context manager."""

    def __init__(self, playwright: FakePlaywright) -> None:
        """Initialize with a fake Playwright object."""
        self._playwright = playwright

    def __enter__(self) -> FakePlaywright:
        """Return the fake Playwright object."""
        return self._playwright

    def __exit__(self, *args: object) -> None:
        """Exit the fake context manager."""
        return None


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


def queued_url(classified_type: UrlContentType, sanitized_url_stem: str) -> QueuedUrl:
    """Build a queued URL for downloader tests."""
    return QueuedUrl(
        original_url=f"https://example.com/{sanitized_url_stem}",
        normalized_url=f"https://example.com/{sanitized_url_stem}",
        sanitized_url_stem=sanitized_url_stem,
        classified_type=classified_type,
    )


def test_downloader_factory_routes_supported_types(tmp_path: Path) -> None:
    """Return type-specific downloaders for supported classes."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    factory = DownloaderFactory(
        config=config,
        http_client=FakeHttpClient(FakeResponse(content=b"pdf", status_code=200, url="https://example.com/file.pdf")),
        html_renderer=FakeHtmlRenderer(RenderedHtml(html="<html></html>", final_url="https://example.com/file.html", http_status=200)),
    )

    assert isinstance(factory.create("pdf"), PdfDownloader)
    assert isinstance(factory.create("html"), HtmlDownloader)
    assert isinstance(factory.create("markdown"), TextDocumentDownloader)
    assert isinstance(factory.create("text"), TextDocumentDownloader)


def test_downloader_factory_rejects_unsupported_types(tmp_path: Path) -> None:
    """Raise a clear error for unsupported URL classes."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    factory = DownloaderFactory(config=config, http_client=None, html_renderer=None)

    with pytest.raises(UnsupportedUrlTypeError) as exc_info:
        factory.create(cast(UrlContentType, "unknown"))

    assert "Unsupported URL type for download: unknown" in str(exc_info.value)


def test_pdf_downloader_expected_path_and_skip_existing(tmp_path: Path) -> None:
    """Calculate expected PDF output path and skip non-empty existing raw files."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    http_client = FakeHttpClient(FakeResponse(content=b"new pdf", status_code=200, url="https://example.com/file.pdf"))
    downloader = PdfDownloader(config, http_client)
    item = queued_url("pdf", "https_example_com_file_pdf")
    expected_path = config.get_url_raw_dir() / "pdf" / "https_example_com_file_pdf.pdf"
    expected_path.parent.mkdir(parents=True)
    expected_path.write_bytes(b"existing pdf")

    result = downloader.download(item)

    assert downloader.expected_raw_path(item) == expected_path
    assert result.raw_path == expected_path
    assert result.status == "skipped_existing"
    assert http_client.calls == []


def test_pdf_downloader_uses_content_fetch_timeout(tmp_path: Path) -> None:
    """Limit PDF fetches to the shared URL content timeout."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    http_client = FakeHttpClient(FakeResponse(content=b"%PDF fake", status_code=200, url="https://example.com/file.pdf"))
    downloader = PdfDownloader(config, http_client)

    downloader.download(queued_url("pdf", "https_example_com_file_pdf"))

    assert http_client.timeouts == [CONTENT_FETCH_TIMEOUT_SECONDS]


def test_text_document_downloader_expected_path_and_skip_existing(tmp_path: Path) -> None:
    """Calculate expected Markdown/text output paths and skip existing raw files."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    http_client = FakeHttpClient(FakeResponse(content=b"# Markdown", status_code=200, url="https://example.com/readme.md"))
    markdown_downloader = TextDocumentDownloader(config, http_client, "markdown")
    text_downloader = TextDocumentDownloader(config, http_client, "text")
    markdown_item = queued_url("markdown", "https_example_com_readme_md")
    text_item = queued_url("text", "https_example_com_notes_txt")
    expected_markdown_path = config.get_url_raw_dir() / "markdown" / "https_example_com_readme_md.md"
    expected_text_path = config.get_url_raw_dir() / "text" / "https_example_com_notes_txt.txt"
    expected_markdown_path.parent.mkdir(parents=True)
    expected_markdown_path.write_text("existing markdown", encoding="utf-8")

    result = markdown_downloader.download(markdown_item)

    assert markdown_downloader.expected_raw_path(markdown_item) == expected_markdown_path
    assert text_downloader.expected_raw_path(text_item) == expected_text_path
    assert result.status == "skipped_existing"
    assert http_client.calls == []


def test_text_document_downloader_downloads_with_content_fetch_timeout(tmp_path: Path) -> None:
    """Download text-like documents with the shared URL content timeout."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    http_client = FakeHttpClient(FakeResponse(content=b"# Markdown", status_code=200, url="https://example.com/readme.md"))
    downloader = TextDocumentDownloader(config, http_client, "markdown")

    result = downloader.download(queued_url("markdown", "https_example_com_readme_md"))

    assert result.raw_path.read_bytes() == b"# Markdown"
    assert http_client.timeouts == [CONTENT_FETCH_TIMEOUT_SECONDS]


def test_html_downloader_expected_path_and_skip_existing(tmp_path: Path) -> None:
    """Calculate expected HTML output path and skip non-empty existing raw files."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    renderer = FakeHtmlRenderer(RenderedHtml(html="<html>new</html>", final_url="https://example.com/file.html", http_status=200))
    downloader = HtmlDownloader(config, renderer)
    item = queued_url("html", "https_example_com_file_html")
    expected_path = config.get_url_raw_dir() / "html" / "https_example_com_file_html.html"
    expected_path.parent.mkdir(parents=True)
    expected_path.write_text("<html>existing</html>", encoding="utf-8")

    result = downloader.download(item)

    assert downloader.expected_raw_path(item) == expected_path
    assert result.raw_path == expected_path
    assert result.status == "skipped_existing"
    assert renderer.calls == []


def test_html_downloader_rejects_http_error_status(tmp_path: Path) -> None:
    """Do not persist rendered error pages as successful HTML downloads."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    renderer = FakeHtmlRenderer(RenderedHtml(html="<html>missing</html>", final_url="https://example.com/missing.html", http_status=404))
    downloader = HtmlDownloader(config, renderer)
    item = queued_url("html", "https_example_com_missing_html")

    with pytest.raises(UrlDownloadError, match="HTTP 404"):
        downloader.download(item)

    assert not downloader.expected_raw_path(item).exists()


def test_playwright_html_renderer_uses_content_fetch_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limit Playwright page navigation to the shared URL content timeout."""
    page = FakePage()
    browser = FakeBrowser(page)
    monkeypatch.setattr("src.url_ingestion.downloader.sync_playwright", lambda: FakeSyncPlaywright(FakePlaywright(browser)))

    result = PlaywrightHtmlRenderer().render("https://example.com/file.html")

    assert result.html == "<html>rendered</html>"
    assert page.goto_kwargs["wait_until"] == HTML_RENDER_WAIT_UNTIL
    assert page.goto_kwargs["timeout"] == CONTENT_FETCH_TIMEOUT_SECONDS * 1000
    assert browser.closed is True
