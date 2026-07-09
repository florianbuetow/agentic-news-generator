"""Integration tests for URL raw-to-cleaned Markdown processing."""

import json
from datetime import date
from pathlib import Path

import pytest
import tiktoken

from src.config import LLMConfig
from src.url_ingestion.clean_content_pipeline import (
    CleaningErrorLog,
    UncleanableRegistry,
    UnextractableRegistry,
    UrlCleanContentPipeline,
    select_pending_items,
)
from src.url_ingestion.formatting import FormattingAgent, LlmClient, OutputWindowExceededError, PromptEstimator
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


def fixed_today() -> date:
    """Return a fixed processing-error log date for deterministic tests."""
    return date(2026, 5, 31)


class WindowAwareTruncatingClient:
    """Fake client that reports output-window truncation until the configured window is large enough."""

    def __init__(self, *, required_output_window: int, cleaned: str) -> None:
        """Record the output window required to finish and the content returned once it fits."""
        self._required_output_window = required_output_window
        self._cleaned = cleaned

    def complete(self, prompt: str, llm: LLMConfig) -> str:
        """Raise as if truncated while the output window is too small, otherwise return cleaned content."""
        if llm.max_tokens < self._required_output_window:
            raise OutputWindowExceededError("simulated finish_reason=length", max_output_tokens=llm.max_tokens)
        return self._cleaned


def make_pipeline(
    tmp_path: Path,
    *,
    formatter_response: str | None = None,
    llm_client: LlmClient | None = None,
    pdf_text: str,
    max_output_tokens: int = 100,
) -> UrlCleanContentPipeline:
    """Create a clean-content pipeline with fake extraction and formatting."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    selected_client: LlmClient
    if llm_client is not None:
        selected_client = llm_client
    elif formatter_response is not None:
        selected_client = FakeLlmClient(formatter_response)
    else:
        raise ValueError("make_pipeline requires formatter_response or llm_client")
    formatting_agent = FormattingAgent(
        llm=make_llm_config(max_tokens=max_output_tokens),
        context_window=1000,
        estimator=PromptEstimator(prompt_template="{source_text}", encoder=tiktoken.get_encoding("o200k_base")),
        skip_threshold_pct=80,
        llm_client=selected_client,
    )
    processor_factory = RawProcessorFactory(
        HtmlRawProcessor(HtmlTextExtractor(), formatting_agent),
        PdfRawProcessor(FakePdfExtractor(pdf_text), formatting_agent),
        PlainTextRawProcessor(formatting_agent),
    )
    registry = UncleanableRegistry(config.get_url_cleaned_dir() / "uncleanable.json")
    unextractable_registry = UnextractableRegistry(config.get_url_cleaned_dir() / "unextractable.json")
    return UrlCleanContentPipeline(
        RawContentScanner(config),
        processor_factory,
        CleaningErrorLog(config, fixed_today),
        registry,
        unextractable_registry,
        max_output_tokens,
    )


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


def test_clean_content_pipeline_writes_failures_to_durable_error_log(tmp_path: Path) -> None:
    """Persist clean-content failures to a dated processing-error log under the cleaned dir for human review."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_html.parent.mkdir(parents=True)
    raw_html.write_text("<html><body><h1>Article</h1><p>Useful content.</p></body></html>", encoding="utf-8")
    # Inline HTML in the formatted output fails cleaned-Markdown validation, producing a processing failure.
    pipeline = make_pipeline(tmp_path, formatter_response="<div>inline html is not allowed</div>", pdf_text="unused")

    summary = pipeline.run(limit=None, raw_path=None, raw_paths=None, force=False)

    assert summary.failure_count == 1
    assert summary.cleaned_count == 0
    error_lines = (config.get_url_cleaned_dir() / "errors" / "2026-05-31.txt").read_text(encoding="utf-8").splitlines()
    assert len(error_lines) == 1
    assert error_lines[0].startswith(f"{raw_html}\t")
    assert not (config.get_url_cleaned_dir() / "html" / "article.md").exists()


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


def test_clean_content_pipeline_records_skips_and_recleans_uncleanable_documents(tmp_path: Path) -> None:
    """Delete truncated output, record the failing window, skip re-runs, then reclean once the output window grows."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "article.html"
    raw_html.parent.mkdir(parents=True)
    raw_html.write_text("<html><body><h1>Article</h1><p>Body.</p></body></html>", encoding="utf-8")
    cleaned_path = config.get_url_cleaned_dir() / "html" / "article.md"
    status_path = config.get_url_cleaned_dir() / "uncleanable.json"
    client = WindowAwareTruncatingClient(required_output_window=1000, cleaned="# Clean\n\nBody.")

    first_summary = make_pipeline(tmp_path, llm_client=client, pdf_text="unused", max_output_tokens=100).run(
        limit=None, raw_path=None, raw_paths=None, force=False
    )

    assert first_summary.uncleanable_count == 1
    assert first_summary.cleaned_count == 0
    assert first_summary.failure_count == 1
    assert not cleaned_path.exists()
    assert json.loads(status_path.read_text(encoding="utf-8"))["article.html"]["failed_output_window_tokens"] == 100

    second_summary = make_pipeline(tmp_path, llm_client=client, pdf_text="unused", max_output_tokens=100).run(
        limit=None, raw_path=None, raw_paths=None, force=False
    )

    assert second_summary.skipped_uncleanable_count == 1
    assert second_summary.uncleanable_count == 0
    assert second_summary.cleaned_count == 0

    third_summary = make_pipeline(tmp_path, llm_client=client, pdf_text="unused", max_output_tokens=100000).run(
        limit=None, raw_path=None, raw_paths=None, force=False
    )

    assert third_summary.cleaned_count == 1
    assert third_summary.skipped_uncleanable_count == 0
    assert cleaned_path.read_text(encoding="utf-8") == "# Clean\n\nBody."


def test_clean_content_pipeline_records_oversized_documents_in_the_same_ledger(tmp_path: Path) -> None:
    """A document too large for the output window lands in uncleanable.json and is skipped next run, with no LLM call."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "big.html"
    raw_html.parent.mkdir(parents=True)
    big_text = " ".join(f"word{index}" for index in range(300))
    raw_html.write_text(f"<html><body>{big_text}</body></html>", encoding="utf-8")
    cleaned_path = config.get_url_cleaned_dir() / "html" / "big.md"
    status_path = config.get_url_cleaned_dir() / "uncleanable.json"
    client = FakeLlmClient("unused")

    first_summary = make_pipeline(tmp_path, llm_client=client, pdf_text="unused", max_output_tokens=100).run(
        limit=None, raw_path=None, raw_paths=None, force=False
    )

    assert first_summary.uncleanable_count == 1
    assert first_summary.cleaned_count == 0
    assert client.prompts == []
    assert not cleaned_path.exists()
    assert json.loads(status_path.read_text(encoding="utf-8"))["big.html"]["failed_output_window_tokens"] == 100

    second_summary = make_pipeline(tmp_path, llm_client=client, pdf_text="unused", max_output_tokens=100).run(
        limit=None, raw_path=None, raw_paths=None, force=False
    )

    assert second_summary.skipped_uncleanable_count == 1
    assert second_summary.uncleanable_count == 0
    assert client.prompts == []


def test_clean_content_pipeline_records_and_skips_unextractable_documents(tmp_path: Path) -> None:
    """Record empty-text extraction failures and skip the same raw bytes on later runs without LLM calls."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "empty.html"
    raw_html.parent.mkdir(parents=True)
    raw_html.write_text("<html><body></body></html>", encoding="utf-8")
    status_path = config.get_url_cleaned_dir() / "unextractable.json"
    client = FakeLlmClient("unused")

    first_summary = make_pipeline(tmp_path, llm_client=client, pdf_text="unused").run(
        limit=None, raw_path=None, raw_paths=None, force=False
    )

    assert first_summary.unextractable_count == 1
    assert first_summary.cleaned_count == 0
    assert first_summary.failure_count == 1
    assert client.prompts == []
    assert json.loads(status_path.read_text(encoding="utf-8"))["empty.html"]["raw_bytes"] == raw_html.stat().st_size

    second_summary = make_pipeline(tmp_path, llm_client=client, pdf_text="unused").run(
        limit=None, raw_path=None, raw_paths=None, force=False
    )

    assert second_summary.skipped_unextractable_count == 1
    assert second_summary.unextractable_count == 0
    assert client.prompts == []


def test_clean_content_pipeline_reattempts_unextractable_after_raw_file_changes(tmp_path: Path) -> None:
    """Re-attempt an unextractable raw file after its byte size changes."""
    config = write_config(tmp_path / "config.yaml", tmp_path / "data")
    raw_html = config.get_url_raw_dir() / "html" / "empty.html"
    raw_html.parent.mkdir(parents=True)
    raw_html.write_text("<html><body></body></html>", encoding="utf-8")
    first_client = FakeLlmClient("unused")
    first_summary = make_pipeline(tmp_path, llm_client=first_client, pdf_text="unused").run(
        limit=None, raw_path=None, raw_paths=None, force=False
    )
    original_raw_bytes = raw_html.stat().st_size

    raw_html.write_text("<html><body><h1>Now</h1><p>Real.</p></body></html>", encoding="utf-8")
    second_summary = make_pipeline(tmp_path, formatter_response="# Now\n\nReal.", pdf_text="unused").run(
        limit=None, raw_path=None, raw_paths=None, force=False
    )

    assert first_summary.unextractable_count == 1
    assert raw_html.stat().st_size != original_raw_bytes
    assert second_summary.skipped_unextractable_count == 0
    assert second_summary.cleaned_count == 1


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
