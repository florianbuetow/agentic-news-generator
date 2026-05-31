#!/usr/bin/env python3
"""Process raw URL content into cleaned Markdown documents."""

import logging
import sys
from pathlib import Path

import tiktoken

from src.config import Config
from src.url_ingestion.clean_content_pipeline import UrlCleanContentPipeline
from src.url_ingestion.formatting import FormattingAgent, LiteLlmClient
from src.url_ingestion.raw_processing import (
    HtmlRawProcessor,
    HtmlTextExtractor,
    PdfRawProcessor,
    PypdfTextExtractor,
    RawContentScanner,
    RawProcessorFactory,
)
from src.util.log_util import configure_root_logger


def main() -> int:
    """Load config and process all configured raw URL content."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"

    try:
        config = Config(config_path)
        configure_root_logger(config.get_data_logs_dir())
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        clean_config = config.get_url_clean_content_config()
        prompt_path = project_root / clean_config.prompt_template
        if not prompt_path.is_file():
            print(f"Error: URL formatting prompt not found: {prompt_path}", file=sys.stderr)
            return 1

        formatting_agent = FormattingAgent(
            llm=clean_config.llm,
            prompt_template=prompt_path.read_text(encoding="utf-8"),
            encoder=tiktoken.get_encoding(config.get_encoding_name()),
            skip_threshold_pct=clean_config.skip_documents_above_context_window_pct,
            llm_client=LiteLlmClient(),
        )
        html_processor = HtmlRawProcessor(HtmlTextExtractor(), formatting_agent)
        pdf_processor = PdfRawProcessor(PypdfTextExtractor(), formatting_agent)
        processor_factory = RawProcessorFactory(html_processor, pdf_processor)
        summary = UrlCleanContentPipeline(RawContentScanner(config), processor_factory).run()
    except (FileNotFoundError, KeyError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"total_pending_count: {summary.total_pending_count}")
    print(f"cleaned_count: {summary.cleaned_count}")
    print(f"skipped_existing_count: {summary.skipped_existing_count}")
    print(f"oversized_count: {summary.oversized_count}")
    print(f"failure_count: {summary.failure_count}")
    if summary.failures:
        print("\n--- Failure Summary ---", file=sys.stderr)
        for failure in summary.failures:
            print(f"{failure.raw_path}: {failure.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
