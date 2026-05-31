"""Raw URL content scanning and type-specific processing."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, Protocol

from pypdf import PdfReader

from src.config import Config
from src.url_ingestion.formatting import FormattingAgent
from src.url_ingestion.metadata import Metadata, MetadataHelper

RawContentType = Literal["pdf", "html"]


class RawProcessingError(RuntimeError):
    """Raised when raw content processing fails."""


@dataclass(frozen=True)
class RawContentItem:
    """One raw file and its expected cleaned Markdown output path."""

    raw_path: Path
    cleaned_path: Path
    metadata_path: Path
    content_type: RawContentType


@dataclass(frozen=True)
class RawScanResult:
    """Raw content scan result."""

    pending_items: tuple[RawContentItem, ...]
    skipped_existing_count: int


class RawContentScanner:
    """Scan configured raw URL folders directly for pending processing work."""

    def __init__(self, config: Config) -> None:
        """Initialize the scanner."""
        self._config = config

    def scan(self) -> RawScanResult:
        """Scan raw PDF and HTML folders and skip already-cleaned files."""
        raw_dir = self._config.get_url_raw_dir()
        if not raw_dir.exists():
            raise FileNotFoundError(f"Configured URL raw folder does not exist: {raw_dir}")
        if not raw_dir.is_dir():
            raise NotADirectoryError(f"Configured URL raw path is not a folder: {raw_dir}")

        pending_items: list[RawContentItem] = []
        skipped_existing_count = 0
        raw_type_specs: tuple[tuple[RawContentType, str], ...] = (("html", ".html"), ("pdf", ".pdf"))
        for content_type, suffix in raw_type_specs:
            type_dir = raw_dir / content_type
            if not type_dir.exists():
                continue
            for raw_path in sorted(type_dir.glob(f"*{suffix}")):
                if raw_path.name.startswith("._") or not raw_path.is_file():
                    continue
                item = self._build_item(raw_path, content_type)
                if item.cleaned_path.is_file() and item.cleaned_path.stat().st_size > 0:
                    skipped_existing_count += 1
                    continue
                pending_items.append(item)
        return RawScanResult(pending_items=tuple(pending_items), skipped_existing_count=skipped_existing_count)

    def _build_item(self, raw_path: Path, content_type: RawContentType) -> RawContentItem:
        """Build one raw content item."""
        cleaned_path = self._config.get_url_cleaned_dir() / content_type / f"{raw_path.stem}.md"
        metadata_path = raw_path.with_name(f"{raw_path.stem}.metadata.json")
        return RawContentItem(raw_path=raw_path, cleaned_path=cleaned_path, metadata_path=metadata_path, content_type=content_type)


class PdfTextExtractor(Protocol):
    """PDF text extractor protocol."""

    def extract_text(self, raw_path: Path) -> str:
        """Extract text from a PDF file."""
        ...


class PypdfTextExtractor:
    """pypdf-backed PDF text extractor."""

    def extract_text(self, raw_path: Path) -> str:
        """Extract text from a PDF file using pypdf."""
        try:
            reader = PdfReader(raw_path)
            page_text = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            raise RawProcessingError(f"PDF text extraction failed for {raw_path}: {exc}") from exc

        extracted_text = "\n\n".join(text.strip() for text in page_text if text.strip()).strip()
        if not extracted_text:
            raise RawProcessingError(f"PDF text extraction produced empty text for {raw_path}")
        return extracted_text


class HtmlTextExtractor:
    """Extract readable text from raw rendered HTML snapshots."""

    def extract_text(self, raw_html: str) -> str:
        """Extract text from the body when present and strip obvious tags."""
        body_html = self._extract_body(raw_html)
        parser = _HtmlBodyTextParser()
        parser.feed(body_html)
        return parser.get_text()

    def _extract_body(self, raw_html: str) -> str:
        """Return body HTML when a body tag is present."""
        match = re.search(r"<body\b[^>]*>(?P<body>.*?)</body>", raw_html, flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            return raw_html
        return match.group("body")


class _HtmlBodyTextParser(HTMLParser):
    """Small HTML-to-text parser for raw URL snapshots."""

    def __init__(self) -> None:
        """Initialize parser state."""
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track skipped tags and add structure for block tags."""
        if tag in {"script", "style", "noscript", "nav"}:
            self._skip_depth += 1
            return
        if tag in {"p", "div", "section", "article", "header", "footer", "h1", "h2", "h3", "li", "br"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Track skipped tag exits and add structure for block tags."""
        if tag in {"script", "style", "noscript", "nav"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag in {"p", "div", "section", "article", "header", "footer", "h1", "h2", "h3", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Collect visible text."""
        if self._skip_depth == 0:
            stripped_data = data.strip()
            if stripped_data:
                self._parts.append(stripped_data)
                self._parts.append(" ")

    def get_text(self) -> str:
        """Return normalized extracted text."""
        lines = [" ".join(line.split()) for line in "".join(self._parts).splitlines()]
        return "\n\n".join(line for line in lines if line)


class RawProcessor(Protocol):
    """Type-specific raw processor protocol."""

    def process(self, item: RawContentItem) -> None:
        """Process one raw item into cleaned Markdown."""
        ...


class HtmlRawProcessor:
    """Process raw HTML files into cleaned Markdown."""

    def __init__(self, html_extractor: HtmlTextExtractor, formatting_agent: FormattingAgent) -> None:
        """Initialize the HTML processor."""
        self._html_extractor = html_extractor
        self._formatting_agent = formatting_agent

    def process(self, item: RawContentItem) -> None:
        """Process one raw HTML file."""
        self._ensure_manual_drop_metadata(item)
        raw_html = item.raw_path.read_text(encoding="utf-8")
        extracted_text = self._html_extractor.extract_text(raw_html)
        if not extracted_text.strip():
            raise RawProcessingError(f"HTML extraction produced empty text for {item.raw_path}")
        cleaned_markdown = self._formatting_agent.format_markdown(extracted_text)
        item.cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        item.cleaned_path.write_text(cleaned_markdown, encoding="utf-8")

    def _ensure_manual_drop_metadata(self, item: RawContentItem) -> None:
        """Create minimal metadata for manually dropped raw files."""
        ensure_manual_drop_metadata(item)


class PdfRawProcessor:
    """Process raw PDF files into cleaned Markdown."""

    def __init__(self, pdf_extractor: PdfTextExtractor, formatting_agent: FormattingAgent) -> None:
        """Initialize the PDF processor."""
        self._pdf_extractor = pdf_extractor
        self._formatting_agent = formatting_agent

    def process(self, item: RawContentItem) -> None:
        """Process one raw PDF file."""
        ensure_manual_drop_metadata(item)
        extracted_text = self._pdf_extractor.extract_text(item.raw_path)
        if not extracted_text.strip():
            raise RawProcessingError(f"PDF extraction produced empty text for {item.raw_path}")
        cleaned_markdown = self._formatting_agent.format_markdown(extracted_text)
        item.cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        item.cleaned_path.write_text(cleaned_markdown, encoding="utf-8")


class RawProcessorFactory:
    """Route raw content items to type-specific processors."""

    def __init__(self, html_processor: HtmlRawProcessor, pdf_processor: PdfRawProcessor) -> None:
        """Initialize the factory."""
        self._html_processor = html_processor
        self._pdf_processor = pdf_processor

    def create(self, content_type: RawContentType) -> RawProcessor:
        """Return the processor for a raw content type."""
        if content_type == "html":
            return self._html_processor
        if content_type == "pdf":
            return self._pdf_processor
        raise RawProcessingError(f"Unsupported raw content type: {content_type}")


def ensure_manual_drop_metadata(item: RawContentItem) -> None:
    """Create minimal metadata for raw files that were manually dropped without metadata."""
    if item.metadata_path.exists():
        return
    metadata = Metadata(
        source_url=None,
        normalized_url=None,
        final_url=None,
        sanitized_url_stem=item.raw_path.stem,
        classified_type=item.content_type,
        downloaded_at=datetime.now(UTC).isoformat(),
        http_status=None,
        raw_path=str(item.raw_path),
        metadata_path=str(item.metadata_path),
        status="manual_drop",
        source_kind="manual_drop",
    )
    MetadataHelper(metadata).save(item.metadata_path)
