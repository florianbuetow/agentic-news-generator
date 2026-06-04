#!/usr/bin/env python3
"""Process raw URL content into cleaned Markdown documents."""

import logging
import sys
from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from src.config import Config
from src.url_ingestion.clean_content_pipeline import UrlCleanContentPipeline, select_pending_items
from src.url_ingestion.formatting import FormattingAgent, FormattingWorkEstimate, LiteLlmClient
from src.url_ingestion.raw_processing import (
    HtmlRawProcessor,
    HtmlTextExtractor,
    PdfRawProcessor,
    PlainTextRawProcessor,
    PypdfTextExtractor,
    RawContentItem,
    RawContentScanner,
    RawContentType,
    RawProcessorFactory,
    RawScanResult,
)
from src.util.log_util import configure_root_logger


def main(argv: list[str] | None = None) -> int:
    """Load config and process all configured raw URL content."""
    parser = ArgumentParser(description="Process raw URL content into cleaned Markdown documents.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect pending raw files, extract text, and estimate LLM work without calling the LLM or writing cleaned files.",
    )
    parser.add_argument("--limit", type=positive_int, help="Process at most this many pending raw files.")
    parser.add_argument("--raw-path", type=Path, help="Process only this pending raw file path.")
    parser.add_argument(
        "--max-estimated-prompt-tokens",
        type=positive_int,
        help="Process only files estimated at or below this total prompt-token count.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Include an already-cleaned --raw-path and overwrite its cleaned Markdown output.",
    )
    args = parser.parse_args(argv)
    if args.force and args.raw_path is None:
        parser.error("--force requires --raw-path")

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"

    try:
        print(f"Loading config: {config_path}", flush=True)
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
        print(f"Raw URL content directory: {config.get_url_raw_dir()}", flush=True)
        print(f"Cleaned Markdown directory: {config.get_url_cleaned_dir()}", flush=True)
        print(f"Formatting model: {clean_config.llm.model}", flush=True)
        print(f"Prompt template: {prompt_path}", flush=True)

        formatting_agent = FormattingAgent(
            llm=clean_config.llm,
            prompt_template=prompt_path.read_text(encoding="utf-8"),
            encoder=tiktoken.get_encoding(config.get_encoding_name()),
            skip_threshold_pct=clean_config.skip_documents_above_context_window_pct,
            llm_client=LiteLlmClient(),
            progress_callback=lambda message: print(f"  {message}", flush=True),
        )
        if args.dry_run:
            return run_dry_run(
                config,
                formatting_agent,
                limit=args.limit,
                raw_path=args.raw_path,
                force=args.force,
                max_estimated_prompt_tokens=args.max_estimated_prompt_tokens,
            )

        html_processor = HtmlRawProcessor(HtmlTextExtractor(), formatting_agent)
        pdf_processor = PdfRawProcessor(PypdfTextExtractor(), formatting_agent)
        plain_text_processor = PlainTextRawProcessor(formatting_agent)
        processor_factory = RawProcessorFactory(html_processor, pdf_processor, plain_text_processor)
        selected_raw_paths = estimate_selected_raw_paths(
            config,
            formatting_agent,
            limit=args.limit,
            raw_path=args.raw_path,
            force=args.force,
            max_estimated_prompt_tokens=args.max_estimated_prompt_tokens,
        )
        print("Processing raw URL content...", flush=True)
        summary = UrlCleanContentPipeline(RawContentScanner(config), processor_factory).run(
            limit=args.limit if selected_raw_paths is None else None,
            raw_path=args.raw_path if selected_raw_paths is None else None,
            raw_paths=selected_raw_paths,
            force=args.force,
        )
    except (FileNotFoundError, KeyError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nSummary:")
    print(f"raw_files_pending: {summary.raw_pending_count}")
    print(f"raw_files_selected: {summary.total_pending_count}")
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


def run_dry_run(
    config: Config,
    formatting_agent: FormattingAgent,
    *,
    limit: int | None = None,
    raw_path: Path | None = None,
    force: bool = False,
    max_estimated_prompt_tokens: int | None = None,
) -> int:
    """Inspect pending raw content without calling the LLM or writing cleaned files."""
    scan_result, estimated_items, failures = estimate_pending_items(
        config,
        formatting_agent,
        limit=limit,
        raw_path=raw_path,
        force=force,
        max_estimated_prompt_tokens=max_estimated_prompt_tokens,
    )

    print("Dry run: no LLM calls and no cleaned files will be written.", flush=True)
    print(f"raw_files_pending: {len(scan_result.pending_items)}", flush=True)
    print(f"raw_files_selected: {len(estimated_items)}", flush=True)
    print(f"skipped_existing_cleaned_count: {scan_result.skipped_existing_count}", flush=True)

    raw_pending_count = len(scan_result.pending_items)
    selected_count = len(estimated_items)
    total_prompt_tokens = 0
    for index, estimated_item in enumerate(estimated_items, start=1):
        item = estimated_item.item
        print(f"[{index}/{selected_count}] {item.content_type}: {item.raw_path}", flush=True)
        total_prompt_tokens += estimated_item.estimate.prompt_tokens
        print(f"  raw_bytes: {item.raw_path.stat().st_size}", flush=True)
        print(f"  extracted_chars: {estimated_item.extracted_chars}", flush=True)
        if estimated_item.extraction_type is not None:
            print(f"  extraction_type: {estimated_item.extraction_type}", flush=True)
        print(f"  estimated_prompt_tokens: {estimated_item.estimate.prompt_tokens}", flush=True)
        print(f"  would_clean: {item.cleaned_path}", flush=True)

    print("\nSummary:")
    print(f"raw_files_pending: {raw_pending_count}")
    print(f"raw_files_selected: {selected_count}")
    print(f"skipped_existing_count: {scan_result.skipped_existing_count}")
    print(f"estimated_prompt_tokens: {total_prompt_tokens}")
    print(f"failure_count: {len(failures)}")
    if failures:
        print("\n--- Failure Summary ---", file=sys.stderr)
        for raw_path, reason in failures:
            print(f"{raw_path}: {reason}", file=sys.stderr)
        return 1
    return 0


@dataclass(frozen=True)
class EstimatedRawContentItem:
    """One pending raw item with extraction and LLM work estimates."""

    item: RawContentItem
    extracted_chars: int
    extraction_type: str | None
    estimate: FormattingWorkEstimate


def estimate_selected_raw_paths(
    config: Config,
    formatting_agent: FormattingAgent,
    *,
    limit: int | None,
    raw_path: Path | None,
    force: bool,
    max_estimated_prompt_tokens: int | None,
) -> tuple[Path, ...] | None:
    """Return explicitly selected raw paths when estimate bounds are requested."""
    if max_estimated_prompt_tokens is None:
        return None
    _, estimated_items, failures = estimate_pending_items(
        config,
        formatting_agent,
        limit=limit,
        raw_path=raw_path,
        force=force,
        max_estimated_prompt_tokens=max_estimated_prompt_tokens,
    )
    if failures:
        failure_text = "; ".join(f"{path}: {reason}" for path, reason in failures)
        raise ValueError(f"Cannot process estimated batch with extraction failures: {failure_text}")
    return tuple(estimated_item.item.raw_path for estimated_item in estimated_items)


def estimate_pending_items(
    config: Config,
    formatting_agent: FormattingAgent,
    *,
    limit: int | None,
    raw_path: Path | None,
    force: bool,
    max_estimated_prompt_tokens: int | None,
) -> tuple[RawScanResult, tuple[EstimatedRawContentItem, ...], tuple[tuple[Path, str], ...]]:
    """Extract and estimate pending raw items, applying optional estimate filters."""
    scan_result = RawContentScanner(config).scan(include_existing_cleaned=force)
    pending_items = select_pending_items(scan_result.pending_items, limit=None, raw_path=raw_path, raw_paths=None)
    html_extractor = HtmlTextExtractor()
    pdf_extractor = PypdfTextExtractor()
    estimated_items: list[EstimatedRawContentItem] = []
    failures: list[tuple[Path, str]] = []
    for item in pending_items:
        try:
            extracted_text, extraction_type = extract_text_for_dry_run(item, html_extractor, pdf_extractor)
            estimate = formatting_agent.estimate_work(extracted_text)
        except Exception as exc:
            failures.append((item.raw_path, str(exc)))
            continue
        if max_estimated_prompt_tokens is not None and estimate.prompt_tokens > max_estimated_prompt_tokens:
            continue
        estimated_items.append(
            EstimatedRawContentItem(
                item=item,
                extracted_chars=len(extracted_text),
                extraction_type=extraction_type,
                estimate=estimate,
            )
        )

    selected_items = tuple(estimated_items[:limit] if limit is not None else estimated_items)
    return scan_result, selected_items, tuple(failures)


def extract_text_for_dry_run(
    item: RawContentItem,
    html_extractor: HtmlTextExtractor,
    pdf_extractor: PypdfTextExtractor,
) -> tuple[str, str | None]:
    """Extract text for dry-run readiness reporting."""
    if item.content_type == "html":
        extraction_result = html_extractor.extract_with_result(item.raw_path.read_text(encoding="utf-8"))
        extracted_text = extraction_result.text
        extraction_type = f"html:{extraction_result.page_type}"
    elif item.content_type == "pdf":
        extracted_text = pdf_extractor.extract_text(item.raw_path)
        extraction_type = None
    else:
        extracted_text = item.raw_path.read_text(encoding="utf-8").strip()
        extraction_type = None
    if not extracted_text.strip():
        raise ValueError(f"{raw_content_type_label(item.content_type)} extraction produced empty text for {item.raw_path}")
    return extracted_text, extraction_type


def raw_content_type_label(content_type: RawContentType) -> str:
    """Return a display label for a raw content type."""
    return content_type.upper()


def positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""
    parsed_value = int(value)
    if parsed_value < 1:
        raise ArgumentTypeError("value must be greater than 0")
    return parsed_value


if __name__ == "__main__":
    sys.exit(main())
