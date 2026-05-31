"""Integration tests for URL raw-to-cleaned Markdown processing."""

from pathlib import Path

import tiktoken

from src.url_ingestion.clean_content_pipeline import UrlCleanContentPipeline
from src.url_ingestion.formatting import FormattingAgent
from src.url_ingestion.raw_processing import (
    HtmlRawProcessor,
    HtmlTextExtractor,
    PdfRawProcessor,
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
    )
    return UrlCleanContentPipeline(RawContentScanner(config), processor_factory)


def test_clean_content_pipeline_writes_cleaned_markdown_for_raw_html_and_pdf(tmp_path: Path) -> None:
    """Process raw HTML and PDF fixtures into cleaned Markdown."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_pdf = config.get_url_raw_dir() / "pdf" / "report.pdf"
    raw_html.parent.mkdir(parents=True)
    raw_pdf.parent.mkdir(parents=True)
    raw_html.write_text("<html><body><h1>Article</h1><p>Useful content.</p></body></html>", encoding="utf-8")
    raw_pdf.write_bytes(b"%PDF fixture")
    pipeline = make_pipeline(tmp_path, formatter_response="# Cleaned\n\nUseful content.", pdf_text="PDF useful content")

    summary = pipeline.run()

    assert summary.cleaned_count == 2
    assert summary.failure_count == 0
    assert (config.get_url_cleaned_dir() / "html" / "article.md").read_text(encoding="utf-8") == "# Cleaned\n\nUseful content."
    assert (config.get_url_cleaned_dir() / "pdf" / "report.md").read_text(encoding="utf-8") == "# Cleaned\n\nUseful content."


def test_clean_content_pipeline_second_run_skips_existing_cleaned_files(tmp_path: Path) -> None:
    """Skip already-cleaned raw files on a second run."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_html.parent.mkdir(parents=True)
    raw_html.write_text("<html><body>Useful content.</body></html>", encoding="utf-8")
    first_pipeline = make_pipeline(tmp_path, formatter_response="Useful content.", pdf_text="unused")
    first_summary = first_pipeline.run()

    second_pipeline = make_pipeline(tmp_path, formatter_response="Changed content.", pdf_text="unused")
    second_summary = second_pipeline.run()

    assert first_summary.cleaned_count == 1
    assert second_summary.cleaned_count == 0
    assert second_summary.skipped_existing_count == 1
    assert (config.get_url_cleaned_dir() / "html" / "article.md").read_text(encoding="utf-8") == "Useful content."
