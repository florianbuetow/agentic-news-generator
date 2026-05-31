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
    HtmlRawProcessor,
    HtmlTextExtractor,
    PdfRawProcessor,
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
    raw_pdf = config.get_url_raw_dir() / "pdf" / "report.pdf"
    skipped_pdf = config.get_url_raw_dir() / "pdf" / "done.pdf"
    raw_html.parent.mkdir(parents=True)
    raw_pdf.parent.mkdir(parents=True)
    raw_html.write_text("<html>article</html>", encoding="utf-8")
    raw_pdf.write_bytes(b"%PDF")
    skipped_pdf.write_bytes(b"%PDF")
    skipped_cleaned = config.get_url_cleaned_dir() / "pdf" / "done.md"
    skipped_cleaned.parent.mkdir(parents=True)
    skipped_cleaned.write_text("already cleaned", encoding="utf-8")

    scan_result = RawContentScanner(config).scan()

    assert scan_result.skipped_existing_count == 1
    assert [(item.content_type, item.raw_path.name, item.cleaned_path.name) for item in scan_result.pending_items] == [
        ("html", "article.html", "article.md"),
        ("pdf", "report.pdf", "report.md"),
    ]


def test_html_text_extractor_uses_body_and_strips_obvious_tags() -> None:
    """Extract useful body text while skipping head, nav, script, and style content."""
    html = """
    <html><head><title>Ignore</title></head>
    <body><nav>Menu</nav><h1>Title</h1><p>Useful <strong>text</strong>.</p><script>bad()</script></body></html>
    """

    text = HtmlTextExtractor().extract_text(html)

    assert "Ignore" not in text
    assert "Menu" not in text
    assert "bad()" not in text
    assert "Title" in text
    assert "Useful" in text


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
    factory = RawProcessorFactory(html_processor, pdf_processor)

    assert factory.create("html") is html_processor
    assert factory.create("pdf") is pdf_processor


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
