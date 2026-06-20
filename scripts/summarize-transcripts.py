#!/usr/bin/env python3
"""Summarize cleaned transcripts using an LLM hosted by LM Studio."""

from __future__ import annotations

import itertools
import logging
import re
import statistics
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm
import tiktoken
from litellm.exceptions import BadRequestError

from src.config import Config, LLMConfig, SummarizeTranscriptsConfig
from src.summarize.lm_studio_client import get_loaded_context_length
from src.summarize.transcripts import collect_pending_files, strip_think_tags
from src.util.fs_util import FSUtil
from src.util.log_util import configure_root_logger, get_logger

logger = get_logger(__name__)


class ContextSizeExceededError(RuntimeError):
    """The LLM server rejected the request because it exceeded the active context."""


def setup_environment() -> tuple[SummarizeTranscriptsConfig, Path, Path, str, str] | int:
    """Load configuration and initialize environment.

    Returns:
        Tuple of (summarize_cfg, cleaned_dir, summaries_dir, prompt_template, encoding_name)
        or error code if setup fails.
    """
    config_path = Config.repo_config_path()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    config = Config(config_path)
    configure_root_logger(config.get_data_logs_dir())

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    litellm.suppress_debug_info = True

    try:
        summarize_cfg = config.get_summarize_transcripts_config()
    except KeyError as e:
        logger.error(f"Error: {e}")
        return 1

    cleaned_dir = config.get_data_downloads_transcripts_cleaned_dir()
    summaries_dir = config.get_data_downloads_transcripts_summaries_dir()

    if not cleaned_dir.exists():
        logger.error(f"Cleaned transcripts directory not found: {cleaned_dir}")
        return 1

    prompt_path = Path(__file__).parent.parent / "prompts" / "summarize-transcript.md"
    if not prompt_path.exists():
        logger.error(f"Prompt template not found: {prompt_path}")
        return 1

    prompt_template = FSUtil.read_text_file(prompt_path)
    encoding_name = config.get_encoding_name()

    configured_context_window = summarize_cfg.llm.context_window
    context_label = "auto" if configured_context_window == "auto" else f"{configured_context_window:,} tokens"

    logger.info(f"Input: {cleaned_dir}")
    logger.info(f"Output: {summaries_dir}")
    logger.info(
        "Model: %s (ctx=%s, skip>%s%%, workers=%s)",
        summarize_cfg.llm.model,
        context_label,
        summarize_cfg.skip_transcripts_above_context_window_pct,
        summarize_cfg.parallelism,
    )

    return summarize_cfg, cleaned_dir, summaries_dir, prompt_template, encoding_name


def call_llm(prompt: str, llm: LLMConfig, max_tokens: int) -> str:
    """Call the LLM and return the response text."""
    messages = [{"role": "user", "content": prompt}]

    try:
        response = litellm.completion(
            model=llm.model,
            messages=messages,
            api_base=llm.api_base,
            api_key=llm.api_key,
            max_tokens=max_tokens,
            temperature=llm.temperature,
        )
    except BadRequestError as e:
        error_msg = str(e)
        if "No models loaded" in error_msg:
            raise RuntimeError(
                f"No models loaded in LM Studio. Expected model: {llm.model}. Load it with: lms load {llm.model.split('/')[-1]}"
            ) from e
        if "Context size has been exceeded" in error_msg:
            raise ContextSizeExceededError("Context size has been exceeded") from e
        raise

    response_text = response.choices[0].message.content
    if response_text is None or response_text.strip() == "":
        raise ValueError("LLM returned empty response")

    return strip_think_tags(response_text.strip())


def resolve_effective_context_window(llm: LLMConfig) -> int:
    """Resolve the context window used for transcript-size gating."""
    configured_context_window = llm.context_window
    try:
        loaded_context_window = get_loaded_context_length(llm.api_base, llm.model)
    except RuntimeError as exc:
        if configured_context_window == "auto":
            raise RuntimeError(f"context_window=auto but LM Studio ctx is unavailable: {exc}") from exc
        logger.warning(
            "LM Studio ctx unavailable; using config ctx=%s (%s)",
            f"{configured_context_window:,}",
            exc,
        )
        return configured_context_window

    if configured_context_window == "auto":
        return loaded_context_window

    if configured_context_window > loaded_context_window:
        raise RuntimeError(
            f"Configured context_window ({configured_context_window:,}) exceeds loaded LM Studio context length ({loaded_context_window:,})"
        )

    if loaded_context_window > configured_context_window:
        logger.info(
            "Using loaded ctx=%s (config=%s)",
            f"{loaded_context_window:,}",
            f"{configured_context_window:,}",
        )

    return loaded_context_window


def resolve_parallel_context_window(effective_context_window: int, parallelism: int) -> int:
    """Return the conservative per-worker context window for local skip checks."""
    return max(1, effective_context_window // parallelism)


def process_single_file(
    txt_file: Path,
    output_file: Path,
    prompt_template: str,
    llm: LLMConfig,
    encoder: tiktoken.Encoding,
    effective_context_window: int,
    skip_threshold_pct: int,
) -> tuple[str, str | None]:
    """Process a single transcript file.

    Returns (status, note). Raises on persistent LLM failure.
    """
    if output_file.exists():
        return "skip", None

    transcript = FSUtil.read_text_file(txt_file)
    if not transcript.strip():
        return "skip (empty)", None

    prompt = prompt_template.replace("{transcript}", transcript)
    prompt_tokens = len(encoder.encode(prompt, disallowed_special=()))
    token_limit = int(effective_context_window * skip_threshold_pct / 100)
    if prompt_tokens > token_limit:
        return (
            "skip (oversized)",
            f"prompt={prompt_tokens:,} > limit={token_limit:,}",
        )

    available_output_tokens = effective_context_window - prompt_tokens
    if available_output_tokens <= 0:
        return (
            "skip (oversized)",
            f"prompt={prompt_tokens:,} > ctx={effective_context_window:,}",
        )
    request_max_tokens = min(llm.max_tokens, available_output_tokens)

    result: tuple[str, str | None] | None = None
    for attempt in range(1, llm.max_retries + 1):
        try:
            summary = call_llm(prompt, llm, request_max_tokens)
            FSUtil.write_text_file(output_file, summary, create_parents=True)
            result = "ok", None
            break
        except ContextSizeExceededError as e:
            raise ContextSizeExceededError(f"context exceeded; prompt={prompt_tokens:,}, ctx={effective_context_window:,}") from e
        except Exception:
            if attempt == llm.max_retries:
                raise
            time.sleep(llm.retry_delay)

    if result is None:
        raise RuntimeError("unreachable")
    return result


def format_eta(seconds_remaining: float) -> str:
    """Format remaining duration as hours and minutes."""
    total_minutes = max(0, int(seconds_remaining // 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours}h{minutes}m"


def estimate_eta_seconds(recent_completion_times: Sequence[float], remaining: int) -> float | None:
    """Estimate seconds remaining from recent completion timestamps (monotonic).

    Uses the median interval between consecutive completions so that occasional
    stalls (a hung request, a giant outlier file) do not inflate the estimate.
    Returns None when there are too few completions to estimate.
    """
    times = list(recent_completion_times)
    if len(times) < 4:  # need a few intervals for a meaningful median
        return None
    intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    return statistics.median(intervals) * remaining


def format_processing_eta(recent_completion_times: Sequence[float], remaining: int) -> str:
    """Format ETA for a file that is about to start processing."""
    eta_seconds = estimate_eta_seconds(recent_completion_times, remaining)
    return "calculating" if eta_seconds is None else format_eta(eta_seconds)


def format_worker_progress_label(worker_id: str, worker_count: int) -> str:
    """Format the worker label used by pre-work progress logs."""
    match = re.fullmatch(r"worker-(\d+)", worker_id)
    if match is None:
        raise ValueError(f"Invalid worker id: {worker_id}")

    worker_number = int(match.group(1))
    return f"WORKER-{worker_number}/{worker_count}"


def log_processing_start(
    item_number: int,
    total: int,
    rel_path: str,
    recent_completion_times: Sequence[float],
    remaining: int,
    worker_id: str,
    worker_count: int,
) -> None:
    """Log the item being processed before slow work starts."""
    eta_label = format_processing_eta(recent_completion_times, remaining)
    worker_prefix = f"{format_worker_progress_label(worker_id, worker_count)} " if worker_count > 1 else ""
    logger.info(f"{worker_prefix}Processing [{item_number}/{total} {item_number / total * 100:.1f}%] [ETA {eta_label}] ... {rel_path}")


def record_process_result(
    rel_path: str,
    result: tuple[str, str, str | None],
    oversized_files: list[str],
    failures: list[tuple[str, str]],
) -> tuple[int, int]:
    """Record one completed worker result. Return (success_delta, skipped_delta)."""
    _, status, note = result

    if status == "context exceeded":
        detail = note or ""
        failures.append((rel_path, detail))
        return 0, 0

    if status == "failed":
        detail = note or ""
        failures.append((rel_path, detail))
        return 0, 0

    if status == "ok":
        return 1, 0

    if status == "skip (oversized)":
        oversized_files.append(rel_path)

    return 0, 1


def run_process_single_file(
    txt_file: Path,
    output_file: Path,
    prompt_template: str,
    llm: LLMConfig,
    encoder: tiktoken.Encoding,
    effective_context_window: int,
    skip_threshold_pct: int,
    worker_id: str,
    worker_count: int,
) -> tuple[str, str, str | None]:
    """Run one pending item and convert exceptions to worker results."""
    display_worker_id = worker_id if worker_count > 1 else ""
    try:
        status, note = process_single_file(
            txt_file,
            output_file,
            prompt_template,
            llm,
            encoder,
            effective_context_window,
            skip_threshold_pct,
        )
    except ContextSizeExceededError as exc:
        return display_worker_id, "context exceeded", str(exc)
    except Exception as exc:
        return display_worker_id, "failed", str(exc)
    return display_worker_id, status, note


def process_pending(
    pending: list[tuple[Path, Path]],
    prompt_template: str,
    llm: LLMConfig,
    encoder: tiktoken.Encoding,
    effective_context_window: int,
    skip_threshold_pct: int,
    parallelism: int,
) -> int:
    """Process all pending files. Return 1 if any file fails after retries."""
    success_count = 0
    skipped_count = 0
    oversized_files: list[str] = []
    failures: list[tuple[str, str]] = []
    total = len(pending)

    worker_numbers = itertools.count(1)
    worker_number_lock = threading.Lock()
    worker_local = threading.local()
    completed_count = 0
    recent_completion_times: deque[float] = deque(maxlen=50)
    progress_lock = threading.Lock()

    def initialize_worker() -> None:
        with worker_number_lock:
            worker_local.worker_id = f"worker-{next(worker_numbers):02d}"

    def current_worker_id() -> str:
        return str(worker_local.worker_id)

    def run_pending_item(item_number: int, txt_file: Path, output_file: Path) -> tuple[str, str, str | None]:
        nonlocal completed_count
        worker_id = current_worker_id()
        rel_path = f"{txt_file.parent.name}/{txt_file.name}"
        with progress_lock:
            remaining = total - completed_count
            recent_times_snapshot = list(recent_completion_times)

        log_processing_start(
            item_number,
            total,
            rel_path,
            recent_times_snapshot,
            remaining,
            worker_id,
            parallelism,
        )

        result = run_process_single_file(
            txt_file,
            output_file,
            prompt_template,
            llm,
            encoder,
            effective_context_window,
            skip_threshold_pct,
            worker_id,
            parallelism,
        )
        with progress_lock:
            completed_count += 1
            recent_completion_times.append(time.monotonic())
        return result

    with ThreadPoolExecutor(
        max_workers=parallelism,
        initializer=initialize_worker,
        thread_name_prefix="summarizer",
    ) as executor:
        future_to_path = {
            executor.submit(run_pending_item, item_number, txt_file, output_file): f"{txt_file.parent.name}/{txt_file.name}"
            for item_number, (txt_file, output_file) in enumerate(pending, start=1)
        }
        for future in as_completed(future_to_path):
            rel_path = future_to_path[future]
            result = future.result()

            success_delta, skipped_delta = record_process_result(
                rel_path,
                result,
                oversized_files,
                failures,
            )
            success_count += success_delta
            skipped_count += skipped_delta

    logger.info(f"Summary: pending={total}, summarized={success_count}, skipped={skipped_count}, failed={len(failures)}")

    if oversized_files:
        logger.warning(f"Oversized skipped: {len(oversized_files)} file(s); limit={skip_threshold_pct}% of worker context")

    if failures:
        logger.error(f"Failures: {len(failures)} file(s)")
        return 1

    return 0


def main() -> int:
    """Main entry point."""
    channel_filter = ""
    if len(sys.argv) == 3 and sys.argv[1] == "--channel":
        channel_filter = sys.argv[2].strip()
    elif len(sys.argv) != 1:
        logger.error("Usage: summarize-transcripts.py [--channel CHANNEL]")
        return 1

    setup_result = setup_environment()
    if isinstance(setup_result, int):
        return setup_result
    summarize_cfg, cleaned_dir, summaries_dir, prompt_template, encoding_name = setup_result

    encoder = tiktoken.get_encoding(encoding_name)
    llm = summarize_cfg.llm

    pending, total, already_done, empty_files = collect_pending_files(cleaned_dir, summaries_dir, channel_filter)

    if total == 0:
        logger.error(f"No .txt files found in: {cleaned_dir}")
        return 1

    logger.info(f"Queue: {len(pending):,} pending / {total:,} total (done {already_done:,}, empty {empty_files:,})")

    if not pending:
        logger.info("Nothing to do — all files already processed.")
        return 0

    try:
        effective_context_window = resolve_effective_context_window(llm)
    except RuntimeError as exc:
        logger.error(f"Error: {exc}")
        return 1

    logger.info(f"Context: {effective_context_window:,} tokens")
    parallel_context_window = resolve_parallel_context_window(effective_context_window, summarize_cfg.parallelism)
    if summarize_cfg.parallelism > 1:
        logger.info(f"Worker context: {parallel_context_window:,} tokens each")
    logger.info(f"Output cap: {llm.max_tokens:,} tokens")

    return process_pending(
        pending,
        prompt_template,
        llm,
        encoder,
        parallel_context_window,
        summarize_cfg.skip_transcripts_above_context_window_pct,
        summarize_cfg.parallelism,
    )


if __name__ == "__main__":
    sys.exit(main())
