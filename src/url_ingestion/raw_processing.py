"""Raw URL content scanning and type-specific processing."""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, Protocol

from pypdf import PdfReader

from src.config import Config
from src.url_ingestion.formatting import FormattingAgent
from src.url_ingestion.metadata import Metadata, MetadataHelper

RawContentType = Literal["pdf", "html", "markdown", "text"]
HtmlPageType = Literal["generic", "substack"]


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


@dataclass(frozen=True)
class RawProcessResult:
    """Metrics from processing one raw URL file into cleaned Markdown."""

    raw_bytes: int
    extracted_chars: int
    prompt_tokens: int
    llm_attempts: int
    formatting_seconds: float
    cleaned_chars: int
    cleaned_path: Path
    extraction_type: str | None = None


class RawContentScanner:
    """Scan configured raw URL folders directly for pending processing work."""

    def __init__(self, config: Config) -> None:
        """Initialize the scanner."""
        self._config = config

    def scan(self, *, include_existing_cleaned: bool) -> RawScanResult:
        """Scan raw PDF and HTML folders and skip already-cleaned files."""
        raw_dir = self._config.get_url_raw_dir()
        if not raw_dir.exists():
            raise FileNotFoundError(f"Configured URL raw folder does not exist: {raw_dir}")
        if not raw_dir.is_dir():
            raise NotADirectoryError(f"Configured URL raw path is not a folder: {raw_dir}")

        pending_items: list[RawContentItem] = []
        skipped_existing_count = 0
        raw_type_specs: tuple[tuple[RawContentType, str], ...] = (
            ("html", ".html"),
            ("markdown", ".md"),
            ("pdf", ".pdf"),
            ("text", ".txt"),
        )
        for content_type, suffix in raw_type_specs:
            type_dir = raw_dir / content_type
            if not type_dir.exists():
                continue
            for raw_path in sorted(type_dir.glob(f"*{suffix}")):
                if raw_path.name.startswith("._") or not raw_path.is_file():
                    continue
                item = self._build_item(raw_path, content_type)
                if not include_existing_cleaned and item.cleaned_path.is_file() and item.cleaned_path.stat().st_size > 0:
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


@dataclass(frozen=True)
class HtmlExtractionResult:
    """Readable text extracted from one HTML page with page-type routing metadata."""

    text: str
    page_type: HtmlPageType


class HtmlPageClassifier:
    """Detect page families that need specialized HTML extraction."""

    def classify(self, raw_html: str) -> HtmlPageType:
        """Classify a raw rendered HTML snapshot by page family."""
        raw_html_lower = raw_html[:250_000].lower()
        if self._is_substack(raw_html_lower):
            return "substack"
        return "generic"

    def _is_substack(self, raw_html_lower: str) -> bool:
        """Return whether the page has Substack-specific structure."""
        substack_markers = (
            "substackcdn.com",
            "substack.com/channel-frame",
            '"is_substack"',
            '"pub_community_enabled"',
        )
        has_substack_marker = any(marker in raw_html_lower for marker in substack_markers)
        has_post_article = "<article" in raw_html_lower and "newsletter-post" in raw_html_lower and "post" in raw_html_lower
        return has_substack_marker and has_post_article


class HtmlTextExtractor:
    """Route rendered HTML snapshots to page-family-specific text extractors."""

    def __init__(self) -> None:
        """Initialize the page-type router."""
        self._page_classifier = HtmlPageClassifier()
        self._generic_extractor = GenericHtmlTextExtractor(skip_container=None, line_filter=None)
        self._substack_extractor = SubstackHtmlTextExtractor()

    def extract_text(self, raw_html: str) -> str:
        """Extract readable text from the best available page-family extractor."""
        return self.extract_with_result(raw_html).text

    def extract_with_result(self, raw_html: str) -> HtmlExtractionResult:
        """Extract readable text and return the detected page family."""
        page_type = self._page_classifier.classify(raw_html)
        if page_type == "substack":
            return HtmlExtractionResult(text=self._substack_extractor.extract_text(raw_html), page_type=page_type)
        return HtmlExtractionResult(text=self._generic_extractor.extract_text(raw_html), page_type=page_type)


class GenericHtmlTextExtractor:
    """Extract readable text from generic raw rendered HTML snapshots."""

    def __init__(
        self,
        *,
        skip_container: Callable[[list[tuple[str, str | None]]], bool] | None,
        line_filter: Callable[[Iterable[str]], tuple[str, ...]] | None,
    ) -> None:
        """Initialize generic body extraction with optional page-specific cleanup hooks."""
        self._skip_container = skip_container
        self._line_filter = line_filter

    def extract_text(self, raw_html: str) -> str:
        """Extract text from the body when present and strip HTML tags."""
        body_html = self._extract_body(raw_html)
        parser = _HtmlBodyTextParser(skip_container=self._skip_container, line_filter=self._line_filter)
        parser.feed(body_html)
        return parser.get_text()

    def _extract_body(self, raw_html: str) -> str:
        """Return body HTML when a body tag is present."""
        match = re.search(r"<body\b[^>]*>(?P<body>.*?)</body>", raw_html, flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            return raw_html
        return match.group("body")


class SubstackHtmlTextExtractor:
    """Extract article text from rendered Substack post pages."""

    def __init__(self) -> None:
        """Initialize a Substack-specific article extractor."""
        self._article_extractor = GenericHtmlTextExtractor(
            skip_container=should_skip_substack_html_container,
            line_filter=filter_substack_boilerplate_lines,
        )

    def extract_text(self, raw_html: str) -> str:
        """Extract only the Substack post article region, excluding comments and page chrome."""
        article_html = self._extract_post_article(raw_html)
        return self._article_extractor.extract_text(article_html)

    def _extract_post_article(self, raw_html: str) -> str:
        """Return the Substack newsletter article HTML when it can be found."""
        article_match = re.search(
            r"<article\b(?=[^>]*\bnewsletter-post\b)(?=[^>]*\bpost\b)[^>]*>.*?</article>",
            raw_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if article_match is not None:
            return article_match.group(0)

        article_match = re.search(r"<article\b[^>]*>.*?</article>", raw_html, flags=re.IGNORECASE | re.DOTALL)
        if article_match is not None:
            return article_match.group(0)
        return raw_html


class _HtmlBodyTextParser(HTMLParser):
    """Small HTML-to-text parser for raw URL snapshots."""

    _void_tags = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(
        self,
        *,
        skip_container: Callable[[list[tuple[str, str | None]]], bool] | None,
        line_filter: Callable[[Iterable[str]], tuple[str, ...]] | None,
    ) -> None:
        """Initialize parser state."""
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0
        self._skip_container = skip_container
        self._line_filter = line_filter

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track skipped tags and add structure for block tags."""
        if self._skip_depth > 0:
            if tag not in self._void_tags:
                self._skip_depth += 1
            return
        if tag in {"script", "style", "noscript"} or (self._skip_container is not None and self._skip_container(attrs)):
            self._skip_depth += 1
            return
        if tag in {"p", "div", "section", "article", "header", "footer", "h1", "h2", "h3", "li", "br"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Track skipped tag exits and add structure for block tags."""
        if self._skip_depth > 0:
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
        filtered_lines: Iterable[str] = (line for line in lines if line)
        if self._line_filter is not None:
            filtered_lines = self._line_filter(filtered_lines)
        return "\n\n".join(filtered_lines)


def should_skip_substack_html_container(attrs: list[tuple[str, str | None]]) -> bool:
    """Return whether a Substack element is known page chrome rather than post content."""
    attr_map = {name.lower(): (value or "").lower() for name, value in attrs}
    if attr_map.get("role") == "dialog":
        return True
    if attr_map.get("data-testid") == "navbar":
        return True
    aria_label = attr_map.get("aria-label")
    if aria_label is not None and "subscribe modal" in aria_label:
        return True
    class_names = attr_map.get("class")
    if class_names is None:
        return False
    if "post-ufi" in class_names or "post-footer" in class_names:
        return True
    return "button-wrapper" in class_names


def filter_substack_boilerplate_lines(lines: Iterable[str]) -> tuple[str, ...]:
    """Remove rendered Substack chrome from extracted article text."""
    boilerplate_exact = {
        "accept all",
        "already have an account? sign in",
        "code of conduct",
        "communities",
        "content",
        "cookie policy",
        "education",
        "events",
        "government",
        "help",
        "home",
        "join the community",
        "legal",
        "manage preferences",
        "navigation",
        "news organizations",
        "nonprofits",
        "privacy policy",
        "reject all",
        "search",
        "share",
        "sign in",
        "small business",
        "stories",
        "subscribe",
        "subscribe now",
        "subscribe sign in",
        "terms of service",
        "watch more",
        "what's new",
        "work",
        "your privacy choices",
    }
    boilerplate_prefixes = (
        "ahead of ai is a reader-supported publication. to receive new posts",
        "by subscribing, you agree substack",
        "discover more from ",
        "over ",
        "sign in or join ",
        "we use cookies ",
    )
    filtered_lines: list[str] = []
    for line in lines:
        normalized_line = " ".join(line.split())
        lower_line = normalized_line.lower()
        if lower_line in boilerplate_exact:
            continue
        if "accept all reject all manage preferences" in lower_line:
            continue
        if "subscribers" in lower_line and len(lower_line.split()) <= 4:
            continue
        if any(lower_line.startswith(prefix) for prefix in boilerplate_prefixes):
            continue
        filtered_lines.append(normalized_line)
    return tuple(filtered_lines)


class RawProcessor(Protocol):
    """Type-specific raw processor protocol."""

    def process(self, item: RawContentItem) -> RawProcessResult:
        """Process one raw item into cleaned Markdown and return metrics."""
        ...


class HtmlRawProcessor:
    """Process raw HTML files into cleaned Markdown."""

    def __init__(self, html_extractor: HtmlTextExtractor, formatting_agent: FormattingAgent) -> None:
        """Initialize the HTML processor."""
        self._html_extractor = html_extractor
        self._formatting_agent = formatting_agent

    def process(self, item: RawContentItem) -> RawProcessResult:
        """Process one raw HTML file."""
        self._ensure_manual_drop_metadata(item)
        raw_html = item.raw_path.read_text(encoding="utf-8")
        extraction_result = self._html_extractor.extract_with_result(raw_html)
        extracted_text = extraction_result.text
        if not extracted_text.strip():
            raise RawProcessingError(f"HTML extraction produced empty text for {item.raw_path}")
        formatting_result = self._formatting_agent.format_markdown_with_stats(extracted_text)
        item.cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        item.cleaned_path.write_text(formatting_result.cleaned_markdown, encoding="utf-8")
        return RawProcessResult(
            raw_bytes=item.raw_path.stat().st_size,
            extracted_chars=len(extracted_text),
            prompt_tokens=formatting_result.prompt_tokens,
            llm_attempts=formatting_result.attempts,
            formatting_seconds=formatting_result.elapsed_seconds,
            cleaned_chars=formatting_result.output_chars,
            cleaned_path=item.cleaned_path,
            extraction_type=f"html:{extraction_result.page_type}",
        )

    def _ensure_manual_drop_metadata(self, item: RawContentItem) -> None:
        """Create minimal metadata for manually dropped raw files."""
        ensure_manual_drop_metadata(item)


class PdfRawProcessor:
    """Process raw PDF files into cleaned Markdown."""

    def __init__(self, pdf_extractor: PdfTextExtractor, formatting_agent: FormattingAgent) -> None:
        """Initialize the PDF processor."""
        self._pdf_extractor = pdf_extractor
        self._formatting_agent = formatting_agent

    def process(self, item: RawContentItem) -> RawProcessResult:
        """Process one raw PDF file."""
        ensure_manual_drop_metadata(item)
        extracted_text = self._pdf_extractor.extract_text(item.raw_path)
        if not extracted_text.strip():
            raise RawProcessingError(f"PDF extraction produced empty text for {item.raw_path}")
        formatting_result = self._formatting_agent.format_markdown_with_stats(extracted_text)
        item.cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        item.cleaned_path.write_text(formatting_result.cleaned_markdown, encoding="utf-8")
        return RawProcessResult(
            raw_bytes=item.raw_path.stat().st_size,
            extracted_chars=len(extracted_text),
            prompt_tokens=formatting_result.prompt_tokens,
            llm_attempts=formatting_result.attempts,
            formatting_seconds=formatting_result.elapsed_seconds,
            cleaned_chars=formatting_result.output_chars,
            cleaned_path=item.cleaned_path,
        )


class PlainTextRawProcessor:
    """Process raw Markdown or plain text files into cleaned Markdown."""

    def __init__(self, formatting_agent: FormattingAgent) -> None:
        """Initialize the plain-text processor."""
        self._formatting_agent = formatting_agent

    def process(self, item: RawContentItem) -> RawProcessResult:
        """Process one raw text-like file."""
        ensure_manual_drop_metadata(item)
        extracted_text = item.raw_path.read_text(encoding="utf-8").strip()
        if not extracted_text:
            raise RawProcessingError(f"{item.content_type} extraction produced empty text for {item.raw_path}")
        formatting_result = self._formatting_agent.format_markdown_with_stats(extracted_text)
        item.cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        item.cleaned_path.write_text(formatting_result.cleaned_markdown, encoding="utf-8")
        return RawProcessResult(
            raw_bytes=item.raw_path.stat().st_size,
            extracted_chars=len(extracted_text),
            prompt_tokens=formatting_result.prompt_tokens,
            llm_attempts=formatting_result.attempts,
            formatting_seconds=formatting_result.elapsed_seconds,
            cleaned_chars=formatting_result.output_chars,
            cleaned_path=item.cleaned_path,
        )


class RawProcessorFactory:
    """Route raw content items to type-specific processors."""

    def __init__(
        self,
        html_processor: HtmlRawProcessor,
        pdf_processor: PdfRawProcessor,
        plain_text_processor: PlainTextRawProcessor,
    ) -> None:
        """Initialize the factory."""
        self._html_processor = html_processor
        self._pdf_processor = pdf_processor
        self._plain_text_processor = plain_text_processor

    def create(self, content_type: RawContentType) -> RawProcessor:
        """Return the processor for a raw content type."""
        if content_type == "html":
            return self._html_processor
        if content_type == "pdf":
            return self._pdf_processor
        if content_type in {"markdown", "text"}:
            return self._plain_text_processor
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
