"""Unit tests for raw URL content scanning and processors."""

from pathlib import Path
from typing import Any

import pytest
import tiktoken
import yaml

from src.config import Config
from src.url_ingestion.formatting import FormattingAgent
from src.url_ingestion.metadata import MetadataHelper
from src.url_ingestion.raw_processing import (
    HtmlPageClassifier,
    HtmlRawProcessor,
    HtmlTextExtractor,
    PdfRawProcessor,
    PlainTextRawProcessor,
    RawContentItem,
    RawContentScanner,
    RawProcessingError,
    RawProcessorFactory,
)
from tests.test_url_formatting import FakeLlmClient, make_llm_config


class FakePdfExtractor:
    """Fake PDF extractor for processor tests."""

    def __init__(self, text: str) -> None:
        """Initialize fake extracted text."""
        self.text = text

    def extract_text(self, raw_path: Path) -> str:
        """Return fake extracted text."""
        return self.text


def write_config(config_path: Path, data_dir: Path) -> Config:
    """Write a minimal config for raw processing tests."""
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


def make_formatting_agent(response: str) -> FormattingAgent:
    """Build a formatting agent with a fake LLM client."""
    return FormattingAgent(
        llm=make_llm_config(),
        prompt_template="{source_text}",
        encoder=tiktoken.get_encoding("o200k_base"),
        skip_threshold_pct=80,
        llm_client=FakeLlmClient(response),
    )


def test_raw_content_scanner_discovers_pending_and_skips_existing_outputs(tmp_path: Path) -> None:
    """Scan raw folders directly and compute cleaned Markdown output paths."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_markdown = config.get_url_raw_dir() / "markdown" / "readme.md"
    raw_pdf = config.get_url_raw_dir() / "pdf" / "report.pdf"
    raw_text = config.get_url_raw_dir() / "text" / "notes.txt"
    skipped_pdf = config.get_url_raw_dir() / "pdf" / "done.pdf"
    raw_html.parent.mkdir(parents=True)
    raw_markdown.parent.mkdir(parents=True)
    raw_pdf.parent.mkdir(parents=True)
    raw_text.parent.mkdir(parents=True)
    raw_html.write_text("<html>article</html>", encoding="utf-8")
    raw_markdown.write_text("# Markdown", encoding="utf-8")
    raw_pdf.write_bytes(b"%PDF")
    raw_text.write_text("Plain text", encoding="utf-8")
    skipped_pdf.write_bytes(b"%PDF")
    skipped_cleaned = config.get_url_cleaned_dir() / "pdf" / "done.md"
    skipped_cleaned.parent.mkdir(parents=True)
    skipped_cleaned.write_text("already cleaned", encoding="utf-8")

    scan_result = RawContentScanner(config).scan(include_existing_cleaned=False)

    assert scan_result.skipped_existing_count == 1
    assert [(item.content_type, item.raw_path.name, item.cleaned_path.name) for item in scan_result.pending_items] == [
        ("html", "article.html", "article.md"),
        ("markdown", "readme.md", "readme.md"),
        ("pdf", "report.pdf", "report.md"),
        ("text", "notes.txt", "notes.md"),
    ]


def test_html_text_extractor_uses_body_and_strips_non_visible_tags() -> None:
    """Extract body text while skipping head, script, and style content."""
    html = """
    <html><head><title>Ignore</title></head>
    <body><nav>Menu</nav><h1>Title</h1><p>Useful <strong>text</strong>.</p><script>bad()</script></body></html>
    """

    text = HtmlTextExtractor().extract_text(html)

    assert "Ignore" not in text
    assert "Menu" in text
    assert "bad()" not in text
    assert "Title" in text
    assert "Useful" in text


def test_generic_html_text_extractor_keeps_body_content_without_page_specific_cleanup() -> None:
    """Generic HTML extraction strips tags but does not apply platform-specific cleanup."""
    html = """
    <html><body>
    <div>Navigation</div>
    <div>Home</div>
    <div>Search</div>
    <div>Government</div>
    <main><h1>Evals Build Hour</h1><p>Noah builds evals with OpenAI o1.</p></main>
    <div>Legal</div>
    <div>Terms of Service</div>
    <div>We use cookies to keep the platform secure and improve your experience.</div>
    <div>Accept All</div>
    <div>Accept All Reject All Manage Preferences</div>
    </body></html>
    """

    text = HtmlTextExtractor().extract_text(html)

    assert "Navigation" in text
    assert "Home" in text
    assert "Search" in text
    assert "Government" in text
    assert "Legal" in text
    assert "Terms of Service" in text
    assert "We use cookies" in text
    assert "Accept All" in text
    assert "Manage Preferences" in text
    assert "Evals Build Hour" in text
    assert "Noah builds evals" in text


def test_html_text_extractor_filters_substack_subscription_chrome() -> None:
    """Remove rendered Substack navigation and subscription modal chrome."""
    html = """
    <html><head><link href="https://substackcdn.com/bundle/theme/main.css" rel="stylesheet"></head><body>
    <div data-testid="navbar"><picture><source srcset="logo.webp"><img src="logo.png"></picture>\
<button>Subscribe</button><button>Sign in</button></div>
    <div role="dialog" aria-label="Subscribe modal">
      <div>Discover more from Ahead of AI</div>
      <div>Over 192,000 subscribers</div>
      <div>By subscribing, you agree Substack's Terms of Use.</div>
      <div>Already have an account? Sign in</div>
    </div>
    <article class="typography newsletter-post post">
      <h1>Noteworthy AI Research Papers of 2024</h1>
      <p>Useful article text.</p>
      <p>Ahead of AI is a reader-supported publication. To receive new posts and support my work, consider subscribing.</p>
    </article>
    </body></html>
    """

    text = HtmlTextExtractor().extract_text(html)

    assert "Subscribe" not in text
    assert "Sign in" not in text
    assert "Discover more" not in text
    assert "subscribers" not in text
    assert "By subscribing" not in text
    assert "reader-supported publication" not in text
    assert "Noteworthy AI Research Papers" in text
    assert "Useful article text" in text


def test_html_page_classifier_detects_substack_posts() -> None:
    """Classify rendered Substack post pages for specialized extraction."""
    html = """
    <html><head><link href="https://substackcdn.com/bundle/theme/main.css" rel="stylesheet"></head>
    <body>
    <iframe src="https://substack.com/channel-frame"></iframe>
    <article class="typography newsletter-post post"><h1>Post title</h1></article>
    </body></html>
    """

    assert HtmlPageClassifier().classify(html) == "substack"
    assert HtmlPageClassifier().classify("<html><body><article><h1>Generic</h1></article></body></html>") == "generic"


def test_substack_html_extractor_uses_post_article_and_excludes_comments() -> None:
    """Use the Substack article boundary instead of extracting comments below the post."""
    html = """
    <html><head><link href="https://substackcdn.com/bundle/theme/main.css" rel="stylesheet"></head>
    <body>
      <div data-testid="navbar"><button>Subscribe</button></div>
      <article class="typography newsletter-post post">
        <div role="region" aria-label="Post header"><h1>Substack Article</h1></div>
        <div role="region" aria-label="Post UFI">
          <div class="byline-wrapper">Sebastian Raschka, PhD</div>
          <div class="post-ufi"><button><div class="label">373</div></button></div>
        </div>
        <p>Useful article body.</p>
      </article>
      <section aria-label="Comments">
        <h2>Comments</h2>
        <p>This reader comment should not be extracted.</p>
      </section>
    </body></html>
    """

    result = HtmlTextExtractor().extract_with_result(html)

    assert result.page_type == "substack"
    assert "Substack Article" in result.text
    assert "Sebastian Raschka" in result.text
    assert "373" not in result.text
    assert "Useful article body" in result.text
    assert "This reader comment" not in result.text
    assert "Comments" not in result.text


def test_pdf_processor_reports_extractor_failure(tmp_path: Path) -> None:
    """Raise when PDF extraction produces no text."""
    raw_path = tmp_path / "raw.pdf"
    raw_path.write_bytes(b"%PDF")
    item = RawContentItem(
        raw_path=raw_path,
        cleaned_path=tmp_path / "cleaned.md",
        metadata_path=tmp_path / "raw.metadata.json",
        content_type="pdf",
    )
    processor = PdfRawProcessor(FakePdfExtractor(""), make_formatting_agent("unused"))

    with pytest.raises(RawProcessingError) as exc_info:
        processor.process(item)

    assert "PDF extraction produced empty text" in str(exc_info.value)


def test_raw_processor_factory_routes_supported_processors(tmp_path: Path) -> None:
    """Route raw content types to their configured processors."""
    html_processor = HtmlRawProcessor(HtmlTextExtractor(), make_formatting_agent("html"))
    pdf_processor = PdfRawProcessor(FakePdfExtractor("pdf"), make_formatting_agent("pdf"))
    plain_text_processor = PlainTextRawProcessor(make_formatting_agent("text"))
    factory = RawProcessorFactory(html_processor, pdf_processor, plain_text_processor)

    assert factory.create("html") is html_processor
    assert factory.create("pdf") is pdf_processor
    assert factory.create("markdown") is plain_text_processor
    assert factory.create("text") is plain_text_processor


def test_processor_creates_minimal_metadata_for_manual_drop(tmp_path: Path) -> None:
    """Create minimal metadata for a raw file without existing metadata."""
    raw_path = tmp_path / "raw.html"
    raw_path.write_text("<html><body>Manual drop text</body></html>", encoding="utf-8")
    item = RawContentItem(
        raw_path=raw_path,
        cleaned_path=tmp_path / "cleaned.md",
        metadata_path=tmp_path / "raw.metadata.json",
        content_type="html",
    )
    processor = HtmlRawProcessor(HtmlTextExtractor(), make_formatting_agent("Manual drop text"))

    processor.process(item)

    metadata = MetadataHelper.load(item.metadata_path).metadata
    assert metadata.source_kind == "manual_drop"
    assert metadata.source_url is None
    assert metadata.classified_type == "html"
    assert item.cleaned_path.read_text(encoding="utf-8") == "Manual drop text"


def test_plain_text_processor_writes_cleaned_markdown(tmp_path: Path) -> None:
    """Process raw Markdown/text files with the formatting agent."""
    raw_path = tmp_path / "raw.md"
    raw_path.write_text("# Raw markdown\n\nUseful text.", encoding="utf-8")
    item = RawContentItem(
        raw_path=raw_path,
        cleaned_path=tmp_path / "cleaned.md",
        metadata_path=tmp_path / "raw.metadata.json",
        content_type="markdown",
    )
    processor = PlainTextRawProcessor(make_formatting_agent("# Cleaned\n\nUseful text."))

    result = processor.process(item)

    assert result.extracted_chars == len("# Raw markdown\n\nUseful text.")
    assert item.cleaned_path.read_text(encoding="utf-8") == "# Cleaned\n\nUseful text."
    assert MetadataHelper.load(item.metadata_path).metadata.classified_type == "markdown"
