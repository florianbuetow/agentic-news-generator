"""Integration tests for URL raw-to-cleaned Markdown processing."""

from pathlib import Path

import pytest
import tiktoken

from src.url_ingestion.clean_content_pipeline import UrlCleanContentPipeline, select_pending_items
from src.url_ingestion.formatting import FormattingAgent
from src.url_ingestion.raw_processing import (
    HtmlRawProcessor,
    HtmlTextExtractor,
    PdfRawProcessor,
    PlainTextRawProcessor,
    RawContentScanner,
    RawProcessorFactory,
)
from tests.test_url_formatting import FakeLlmClient, make_llm_config
from tests.test_url_raw_processing import FakePdfExtractor, write_config


def make_pipeline(tmp_path: Path, *, formatter_response: str, pdf_text: str) -> UrlCleanContentPipeline:
    """Create a clean-content pipeline with fake extraction and formatting."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    formatting_agent = FormattingAgent(
        llm=make_llm_config(),
        prompt_template="{source_text}",
        encoder=tiktoken.get_encoding("o200k_base"),
        skip_threshold_pct=80,
        llm_client=FakeLlmClient(formatter_response),
    )
    processor_factory = RawProcessorFactory(
        HtmlRawProcessor(HtmlTextExtractor(), formatting_agent),
        PdfRawProcessor(FakePdfExtractor(pdf_text), formatting_agent),
        PlainTextRawProcessor(formatting_agent),
    )
    return UrlCleanContentPipeline(RawContentScanner(config), processor_factory)


def test_clean_content_pipeline_writes_cleaned_markdown_for_raw_documents(tmp_path: Path) -> None:
    """Process raw HTML, PDF, Markdown, and text fixtures into cleaned Markdown."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_markdown = config.get_url_raw_dir() / "markdown" / "readme.md"
    raw_pdf = config.get_url_raw_dir() / "pdf" / "report.pdf"
    raw_text = config.get_url_raw_dir() / "text" / "notes.txt"
    raw_html.parent.mkdir(parents=True)
    raw_markdown.parent.mkdir(parents=True)
    raw_pdf.parent.mkdir(parents=True)
    raw_text.parent.mkdir(parents=True)
    raw_html.write_text("<html><body><h1>Article</h1><p>Useful content.</p></body></html>", encoding="utf-8")
    raw_markdown.write_text("# Useful markdown", encoding="utf-8")
    raw_pdf.write_bytes(b"%PDF fixture")
    raw_text.write_text("Useful text", encoding="utf-8")
    pipeline = make_pipeline(tmp_path, formatter_response="# Cleaned\n\nUseful content.", pdf_text="PDF useful content")

    summary = pipeline.run(limit=None, raw_path=None, raw_paths=None, force=False)

    assert summary.cleaned_count == 4
    assert summary.failure_count == 0
    assert (config.get_url_cleaned_dir() / "html" / "article.md").read_text(encoding="utf-8") == "# Cleaned\n\nUseful content."
    assert (config.get_url_cleaned_dir() / "markdown" / "readme.md").read_text(encoding="utf-8") == "# Cleaned\n\nUseful content."
    assert (config.get_url_cleaned_dir() / "pdf" / "report.md").read_text(encoding="utf-8") == "# Cleaned\n\nUseful content."
    assert (config.get_url_cleaned_dir() / "text" / "notes.md").read_text(encoding="utf-8") == "# Cleaned\n\nUseful content."


def test_clean_content_pipeline_second_run_skips_existing_cleaned_files(tmp_path: Path) -> None:
    """Skip already-cleaned raw files on a second run."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_html.parent.mkdir(parents=True)
    raw_html.write_text("<html><body>Useful content.</body></html>", encoding="utf-8")
    first_pipeline = make_pipeline(tmp_path, formatter_response="Useful content.", pdf_text="unused")
    first_summary = first_pipeline.run(limit=None, raw_path=None, raw_paths=None, force=False)

    second_pipeline = make_pipeline(tmp_path, formatter_response="Changed content.", pdf_text="unused")
    second_summary = second_pipeline.run(limit=None, raw_path=None, raw_paths=None, force=False)

    assert first_summary.cleaned_count == 1
    assert second_summary.cleaned_count == 0
    assert second_summary.skipped_existing_count == 1
    assert (config.get_url_cleaned_dir() / "html" / "article.md").read_text(encoding="utf-8") == "Useful content."


def test_clean_content_pipeline_force_reprocesses_existing_cleaned_raw_path(tmp_path: Path) -> None:
    """Allow bounded reprocessing for one already-cleaned raw file."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_html.parent.mkdir(parents=True)
    raw_html.write_text("<html><body>Useful content.</body></html>", encoding="utf-8")
    cleaned_html = config.get_url_cleaned_dir() / "html" / "article.md"
    cleaned_html.parent.mkdir(parents=True)
    cleaned_html.write_text("old cleaned content", encoding="utf-8")
    pipeline = make_pipeline(tmp_path, formatter_response="new cleaned content", pdf_text="unused")

    summary = pipeline.run(limit=None, raw_path=raw_html, raw_paths=None, force=True)

    assert summary.cleaned_count == 1
    assert cleaned_html.read_text(encoding="utf-8") == "new cleaned content"


def test_clean_content_pipeline_can_limit_selected_pending_items(tmp_path: Path) -> None:
    """Allow bounded clean-content runs for operational smoke checks."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_pdf = config.get_url_raw_dir() / "pdf" / "report.pdf"
    raw_html.parent.mkdir(parents=True)
    raw_pdf.parent.mkdir(parents=True)
    raw_html.write_text("<html><body>Useful content.</body></html>", encoding="utf-8")
    raw_pdf.write_bytes(b"%PDF fixture")
    pipeline = make_pipeline(tmp_path, formatter_response="Useful content.", pdf_text="PDF useful content")

    summary = pipeline.run(limit=1, raw_path=None, raw_paths=None, force=False)

    assert summary.total_pending_count == 1
    assert summary.cleaned_count == 1
    cleaned_files = sorted(config.get_url_cleaned_dir().glob("*/*.md"))
    assert len(cleaned_files) == 1


def test_select_pending_items_can_select_one_raw_path(tmp_path: Path) -> None:
    """Select a specific pending raw path for a bounded production smoke."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_pdf = config.get_url_raw_dir() / "pdf" / "report.pdf"
    raw_html.parent.mkdir(parents=True)
    raw_pdf.parent.mkdir(parents=True)
    raw_html.write_text("<html><body>Useful content.</body></html>", encoding="utf-8")
    raw_pdf.write_bytes(b"%PDF fixture")
    pending_items = RawContentScanner(config).scan(include_existing_cleaned=False).pending_items

    selected_items = select_pending_items(pending_items, limit=None, raw_path=raw_pdf, raw_paths=None)

    assert len(selected_items) == 1
    assert selected_items[0].raw_path == raw_pdf


def test_select_pending_items_can_select_multiple_raw_paths(tmp_path: Path) -> None:
    """Select a bounded estimated batch by exact raw paths."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_pdf = config.get_url_raw_dir() / "pdf" / "report.pdf"
    raw_html.parent.mkdir(parents=True)
    raw_pdf.parent.mkdir(parents=True)
    raw_html.write_text("<html><body>Useful content.</body></html>", encoding="utf-8")
    raw_pdf.write_bytes(b"%PDF fixture")
    pending_items = RawContentScanner(config).scan(include_existing_cleaned=False).pending_items

    selected_items = select_pending_items(pending_items, limit=None, raw_path=None, raw_paths=(raw_pdf, raw_html))

    assert {item.raw_path for item in selected_items} == {raw_html, raw_pdf}


def test_select_pending_items_rejects_non_pending_raw_path(tmp_path: Path) -> None:
    """Fail clearly when an explicitly selected raw file is not pending."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_html.parent.mkdir(parents=True)
    raw_html.write_text("<html><body>Useful content.</body></html>", encoding="utf-8")
    pending_items = RawContentScanner(config).scan(include_existing_cleaned=False).pending_items

    with pytest.raises(ValueError, match="not pending"):
        select_pending_items(pending_items, limit=None, raw_path=config.get_url_raw_dir() / "html" / "missing.html", raw_paths=None)


def test_select_pending_items_rejects_non_pending_raw_paths(tmp_path: Path) -> None:
    """Fail clearly when any explicitly selected raw path is not pending."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_html.parent.mkdir(parents=True)
    raw_html.write_text("<html><body>Useful content.</body></html>", encoding="utf-8")
    pending_items = RawContentScanner(config).scan(include_existing_cleaned=False).pending_items

    with pytest.raises(ValueError, match="Selected raw paths"):
        select_pending_items(pending_items, limit=None, raw_path=None, raw_paths=(config.get_url_raw_dir() / "html" / "missing.html",))
