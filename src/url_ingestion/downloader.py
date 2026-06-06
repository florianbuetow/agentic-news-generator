"""Type-specific raw content downloaders for URL ingestion."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import requests
from playwright.sync_api import sync_playwright

from src.config import Config
from src.url_ingestion.classifier import UrlContentType
from src.url_ingestion.queue_reader import QueuedUrl

CONTENT_FETCH_TIMEOUT_SECONDS: int = 10
HtmlRenderWaitUntil = Literal["commit", "domcontentloaded", "load", "networkidle"]
HTML_RENDER_WAIT_UNTIL: HtmlRenderWaitUntil = "domcontentloaded"


class UnsupportedUrlTypeError(ValueError):
    """Raised when no downloader exists for a classified URL type."""


class UrlDownloadError(RuntimeError):
    """Raised when a URL download fails validation."""


class NonHtmlContentError(ValueError):
    """Raised when an HTML-classified URL serves non-HTML content that must be re-routed."""


@dataclass(frozen=True)
class DownloadResult:
    """Result returned by a type-specific downloader."""

    raw_path: Path
    final_url: str | None
    http_status: int | None
    status: str


class Downloader(Protocol):
    """Downloader interface for a classified URL."""

    def download(self, queued_url: QueuedUrl) -> DownloadResult:
        """Download a queued URL and return the raw file path."""
        ...


class HttpResponse(Protocol):
    """Minimal HTTP response protocol used by the PDF downloader."""

    @property
    def status_code(self) -> int:
        """Return the HTTP status code."""
        ...

    @property
    def content(self) -> bytes:
        """Return the response content bytes."""
        ...

    @property
    def url(self) -> str:
        """Return the final response URL after redirects."""
        ...

    def raise_for_status(self) -> None:
        """Raise an exception for failed HTTP statuses."""


class HttpClient(Protocol):
    """Minimal HTTP client protocol used by the PDF downloader."""

    def get(self, url: str, *, headers: dict[str, str], timeout: int, allow_redirects: bool) -> HttpResponse:
        """Fetch one URL."""
        ...


class RequestsHttpClient:
    """Requests-backed HTTP client."""

    def get(self, url: str, *, headers: dict[str, str], timeout: int, allow_redirects: bool) -> HttpResponse:
        """Fetch one URL using requests."""
        return requests.get(url, headers=headers, timeout=timeout, allow_redirects=allow_redirects)


@dataclass(frozen=True)
class RenderedHtml:
    """Rendered HTML browser output."""

    html: str
    final_url: str | None
    http_status: int | None


class HtmlRenderer(Protocol):
    """Browser renderer protocol for HTML downloads."""

    def render(self, url: str) -> RenderedHtml:
        """Render one HTML URL and return the rendered document."""
        ...


class PlaywrightHtmlRenderer:
    """Playwright-backed HTML renderer."""

    def __init__(self) -> None:
        """Initialize the renderer with a hard content-fetch timeout."""
        self._timeout_ms = CONTENT_FETCH_TIMEOUT_SECONDS * 1000

    def render(self, url: str) -> RenderedHtml:
        """Render HTML in headless browser mode and return the document element HTML."""
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    user_agent=user_agent,
                    locale="en-US",
                    extra_http_headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )
                page = context.new_page()
                response = page.goto(url, wait_until=HTML_RENDER_WAIT_UNTIL, timeout=self._timeout_ms)
                html = page.evaluate("document.documentElement.outerHTML")
                final_url = page.url
                http_status = response.status if response is not None else None
                return RenderedHtml(html=html, final_url=final_url, http_status=http_status)
            finally:
                browser.close()


class PdfDownloader:
    """Download PDF URLs to the configured raw PDF folder."""

    def __init__(self, config: Config, http_client: HttpClient) -> None:
        """Initialize the PDF downloader."""
        self._config = config
        self._http_client = http_client

    def download(self, queued_url: QueuedUrl) -> DownloadResult:
        """Download one PDF URL or skip an existing non-empty raw file."""
        raw_path = self.expected_raw_path(queued_url)
        if self._is_non_empty_file(raw_path):
            return DownloadResult(raw_path=raw_path, final_url=queued_url.normalized_url, http_status=None, status="skipped_existing")

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = self._http_client.get(
                queued_url.normalized_url,
                headers=self._browser_headers(),
                timeout=CONTENT_FETCH_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise UrlDownloadError(f"PDF download failed for {queued_url.normalized_url}: {exc}") from exc

        raw_path.write_bytes(response.content)
        self._validate_raw_file(raw_path)
        return DownloadResult(
            raw_path=raw_path,
            final_url=response.url,
            http_status=response.status_code,
            status="downloaded",
        )

    def expected_raw_path(self, queued_url: QueuedUrl) -> Path:
        """Return the expected raw PDF output path."""
        return self._config.get_url_raw_dir() / "pdf" / f"{queued_url.sanitized_url_stem}.pdf"

    def _browser_headers(self) -> dict[str, str]:
        """Return browser-like headers for HTTP downloads."""
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _validate_raw_file(self, raw_path: Path) -> None:
        """Validate the raw PDF output exists and is non-empty."""
        if not self._is_non_empty_file(raw_path):
            raise UrlDownloadError(f"PDF download produced an empty or missing file: {raw_path}")

    def _is_non_empty_file(self, raw_path: Path) -> bool:
        """Return whether a raw output file already exists and is non-empty."""
        return raw_path.is_file() and raw_path.stat().st_size > 0


class TextDocumentDownloader:
    """Download text-like document URLs to the configured raw folders."""

    def __init__(self, config: Config, http_client: HttpClient, classified_type: Literal["markdown", "text"]) -> None:
        """Initialize the text document downloader."""
        self._config = config
        self._http_client = http_client
        self._classified_type = classified_type

    def download(self, queued_url: QueuedUrl) -> DownloadResult:
        """Download one text-like URL or skip an existing non-empty raw file."""
        raw_path = self.expected_raw_path(queued_url)
        if self._is_non_empty_file(raw_path):
            return DownloadResult(raw_path=raw_path, final_url=queued_url.normalized_url, http_status=None, status="skipped_existing")

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = self._http_client.get(
                queued_url.normalized_url,
                headers=self._browser_headers(),
                timeout=CONTENT_FETCH_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise UrlDownloadError(f"{self._classified_type} download failed for {queued_url.normalized_url}: {exc}") from exc

        raw_path.write_bytes(response.content)
        self._validate_raw_file(raw_path)
        return DownloadResult(
            raw_path=raw_path,
            final_url=response.url,
            http_status=response.status_code,
            status="downloaded",
        )

    def expected_raw_path(self, queued_url: QueuedUrl) -> Path:
        """Return the expected raw text document output path."""
        suffix = ".md" if self._classified_type == "markdown" else ".txt"
        return self._config.get_url_raw_dir() / self._classified_type / f"{queued_url.sanitized_url_stem}{suffix}"

    def _browser_headers(self) -> dict[str, str]:
        """Return browser-like headers for text downloads."""
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/markdown,text/plain,text/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _validate_raw_file(self, raw_path: Path) -> None:
        """Validate the raw text output exists and is non-empty."""
        if not self._is_non_empty_file(raw_path):
            raise UrlDownloadError(f"{self._classified_type} download produced an empty or missing file: {raw_path}")

    def _is_non_empty_file(self, raw_path: Path) -> bool:
        """Return whether a raw output file already exists and is non-empty."""
        return raw_path.is_file() and raw_path.stat().st_size > 0


class HtmlDownloader:
    """Download rendered HTML URLs, falling back to a direct HTTP fetch when rendering is blocked."""

    def __init__(self, config: Config, renderer: HtmlRenderer, http_client: HttpClient) -> None:
        """Initialize the HTML downloader."""
        self._config = config
        self._renderer = renderer
        self._http_client = http_client

    def download(self, queued_url: QueuedUrl) -> DownloadResult:
        """Render and save one HTML URL, fall back to a direct HTTP fetch when blocked, or skip an existing raw file."""
        raw_path = self.expected_raw_path(queued_url)
        if self._is_non_empty_file(raw_path):
            return DownloadResult(raw_path=raw_path, final_url=queued_url.normalized_url, http_status=None, status="skipped_existing")

        render_outcome = self._render(queued_url.normalized_url, raw_path)
        if isinstance(render_outcome, DownloadResult):
            return render_outcome
        return self._download_via_http_fallback(queued_url, raw_path, render_outcome)

    def expected_raw_path(self, queued_url: QueuedUrl) -> Path:
        """Return the expected raw HTML output path."""
        return self._config.get_url_raw_dir() / "html" / f"{queued_url.sanitized_url_stem}.html"

    def _render(self, normalized_url: str, raw_path: Path) -> DownloadResult | str:
        """Render HTML and persist it, or return the reason rendering did not yield usable HTML."""
        try:
            rendered_html = self._renderer.render(normalized_url)
        except Exception as exc:
            return f"HTML render failed for {normalized_url}: {exc}"
        if rendered_html.http_status is not None and rendered_html.http_status >= 400:
            return f"HTML render returned HTTP {rendered_html.http_status} for {normalized_url}"
        if not rendered_html.html.strip():
            return f"HTML render returned empty content for {normalized_url}"
        return self._persist_html(raw_path, rendered_html.html, rendered_html.final_url, rendered_html.http_status)

    def _download_via_http_fallback(self, queued_url: QueuedUrl, raw_path: Path, render_error: str) -> DownloadResult:
        """Fetch HTML directly when the headless renderer is blocked, or signal non-HTML content for re-routing."""
        try:
            response = self._http_client.get(
                queued_url.normalized_url,
                headers=self._browser_headers(),
                timeout=CONTENT_FETCH_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            raise UrlDownloadError(f"{render_error}; HTTP fallback failed for {queued_url.normalized_url}: {exc}") from exc
        if response.status_code >= 400:
            raise UrlDownloadError(f"{render_error}; HTTP fallback returned HTTP {response.status_code} for {queued_url.normalized_url}")
        content = response.content
        if content.startswith(b"%PDF"):
            raise NonHtmlContentError(f"{queued_url.normalized_url} served PDF content for an HTML-classified URL")
        decoded_html = content.decode("utf-8", errors="replace")
        if "<" not in decoded_html or not decoded_html.strip():
            raise UrlDownloadError(f"{render_error}; HTTP fallback returned non-HTML content for {queued_url.normalized_url}")
        return self._persist_html(raw_path, decoded_html, response.url, response.status_code)

    def _persist_html(self, raw_path: Path, html: str, final_url: str | None, http_status: int | None) -> DownloadResult:
        """Write rendered or fetched HTML to the configured raw HTML folder."""
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(html, encoding="utf-8")
        self._validate_raw_file(raw_path)
        return DownloadResult(raw_path=raw_path, final_url=final_url, http_status=http_status, status="downloaded")

    def _browser_headers(self) -> dict[str, str]:
        """Return browser-like headers for the HTTP fallback fetch."""
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _validate_raw_file(self, raw_path: Path) -> None:
        """Validate the raw HTML output exists and is non-empty."""
        if not self._is_non_empty_file(raw_path):
            raise UrlDownloadError(f"HTML download produced an empty or missing file: {raw_path}")

    def _is_non_empty_file(self, raw_path: Path) -> bool:
        """Return whether a raw output file already exists and is non-empty."""
        return raw_path.is_file() and raw_path.stat().st_size > 0


class DownloaderFactory:
    """Create type-specific downloaders for classified URLs."""

    def __init__(self, config: Config, http_client: HttpClient | None, html_renderer: HtmlRenderer | None) -> None:
        """Initialize the factory with explicit dependencies."""
        self._config = config
        self._http_client = http_client
        self._html_renderer = html_renderer

    @classmethod
    def default(cls, config: Config) -> "DownloaderFactory":
        """Create a production downloader factory."""
        return cls(config=config, http_client=RequestsHttpClient(), html_renderer=PlaywrightHtmlRenderer())

    def create(self, classified_type: UrlContentType) -> Downloader:
        """Return the downloader for a classified URL type."""
        if classified_type == "pdf":
            if self._http_client is None:
                raise UrlDownloadError("PDF downloader requires an HTTP client")
            return PdfDownloader(self._config, self._http_client)
        if classified_type == "markdown" or classified_type == "text":
            if self._http_client is None:
                raise UrlDownloadError(f"{classified_type} downloader requires an HTTP client")
            return TextDocumentDownloader(self._config, self._http_client, classified_type)
        if classified_type == "html":
            if self._html_renderer is None:
                raise UrlDownloadError("HTML downloader requires a renderer")
            if self._http_client is None:
                raise UrlDownloadError("HTML downloader requires an HTTP client")
            return HtmlDownloader(self._config, self._html_renderer, self._http_client)
        raise UnsupportedUrlTypeError(f"Unsupported URL type for download: {classified_type}")
