#!/usr/bin/env python3
"""Summarize cleaned transcripts using an LLM hosted by LM Studio."""

from __future__ import annotations

import itertools
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

import litellm
import tiktoken
from litellm.exceptions import BadRequestError

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config, LLMConfig, SummarizeTranscriptsConfig
from src.util.fs_util import FSUtil
from src.util.log_util import configure_root_logger, get_logger

logger = get_logger(__name__)

LM_STUDIO_MODELS_PATH = "/api/v0/models"


class ContextSizeExceededError(RuntimeError):
    """The LLM server rejected the request because it exceeded the active context."""


def setup_environment() -> tuple[SummarizeTranscriptsConfig, Path, Path, str, str] | int:
    """Load configuration and initialize environment.

    Returns:
        Tuple of (summarize_cfg, cleaned_dir, summaries_dir, prompt_template, encoding_name)
        or error code if setup fails.
    """
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
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


def strip_think_tags(text: str) -> str:
    """Strip <think> reasoning tags from model responses."""
    think_match = re.search(r"</think>\s*(.*)$", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return text


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


def get_lm_studio_models_url(api_base: str | None) -> str:
    """Return the native LM Studio models endpoint for an OpenAI-compatible API base."""
    if api_base is None:
        raise ValueError("api_base is required to query LM Studio model metadata")

    parsed = urlparse(api_base)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid LM Studio api_base: {api_base}")

    return urlunparse((parsed.scheme, parsed.netloc, LM_STUDIO_MODELS_PATH, "", "", ""))


def get_model_id_candidates(model: str) -> list[str]:
    """Return model IDs to try against LM Studio metadata."""
    candidates = [model]
    if model.startswith("openai/"):
        candidates.append(model.removeprefix("openai/"))
    return candidates


def fetch_lm_studio_models(models_url: str) -> list[object]:
    """Fetch LM Studio model metadata."""
    try:
        with urlopen(models_url, timeout=10) as response:
            payload = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
    except HTTPError as exc:
        raise RuntimeError(f"LM Studio metadata request failed with HTTP {exc.code}: {models_url}") from exc
    except URLError as exc:
        raise RuntimeError(f"LM Studio metadata endpoint is unreachable: {models_url} ({exc.reason})") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"LM Studio metadata request timed out: {models_url}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LM Studio metadata response was not valid JSON: {models_url}") from exc

    data_raw = payload.get("data")
    if not isinstance(data_raw, list):
        raise RuntimeError("LM Studio metadata response did not contain a data list")

    return cast(list[object], data_raw)


def extract_loaded_context_length(model_info: dict[str, Any], model: str) -> int:
    """Extract the loaded context length from one LM Studio model record."""
    if model_info.get("state") != "loaded":
        raise RuntimeError(f"Configured model is not loaded in LM Studio: {model}")
    loaded_context_length = model_info.get("loaded_context_length")
    if not isinstance(loaded_context_length, int) or loaded_context_length <= 0:
        raise RuntimeError(f"Loaded LM Studio model does not report loaded_context_length: {model}")
    return loaded_context_length


def get_loaded_context_length(llm: LLMConfig) -> int:
    """Fetch the loaded context length for the configured LM Studio model."""
    data = fetch_lm_studio_models(get_lm_studio_models_url(llm.api_base))
    candidates = set(get_model_id_candidates(llm.model))
    loaded_context_length: int | None = None
    for model_info_raw in data:
        if not isinstance(model_info_raw, dict):
            continue
        model_info = cast(dict[str, Any], model_info_raw)
        if model_info.get("id") not in candidates:
            continue
        loaded_context_length = extract_loaded_context_length(model_info, llm.model)
        break

    if loaded_context_length is not None:
        return loaded_context_length

    raise RuntimeError(f"Configured model was not found in LM Studio metadata: {llm.model}")


def resolve_effective_context_window(llm: LLMConfig) -> int:
    """Resolve the context window used for transcript-size gating."""
    configured_context_window = llm.context_window
    try:
        loaded_context_window = get_loaded_context_length(llm)
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


def format_processing_eta(start_time: float, completed_count: int, total: int) -> str:
    """Format ETA for a file that is about to start processing."""
    if completed_count == 0:
        return "calculating"

    elapsed = time.monotonic() - start_time
    avg = elapsed / completed_count
    eta_seconds = avg * (total - completed_count)
    return format_eta(eta_seconds)


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
    start_time: float,
    completed_count: int,
    worker_id: str,
    worker_count: int,
) -> None:
    """Log the item being processed before slow work starts."""
    eta_label = format_processing_eta(start_time, completed_count, total)
    worker_prefix = f"{format_worker_progress_label(worker_id, worker_count)} " if worker_count > 1 else ""
    logger.info(f"{worker_prefix}Processing [{item_number}/{total} {item_number / total * 100:.1f}%] [ETA {eta_label}] ... {rel_path}")


def collect_pending_files(
    cleaned_dir: Path,
    summaries_dir: Path,
    channel_filter: str,
) -> tuple[list[tuple[Path, Path]], int, int, int]:
    """Scan for transcript files and partition into pending/done/empty."""
    txt_files = FSUtil.find_files_by_extension(cleaned_dir, ".txt", recursive=True)
    txt_files = [f for f in txt_files if not f.name.startswith("._")]
    if channel_filter:
        txt_files = [f for f in txt_files if f.parent.name == channel_filter]

    pending: list[tuple[Path, Path]] = []
    already_done = 0
    empty_files = 0

    for txt_file in txt_files:
        channel_name = txt_file.parent.name
        output_file = summaries_dir / channel_name / (txt_file.stem + ".md")

        if output_file.exists():
            already_done += 1
            continue

        content = FSUtil.read_text_file(txt_file)
        if not content.strip():
            empty_files += 1
            continue

        pending.append((txt_file, output_file))

    pending_per_channel: dict[str, int] = {}
    for txt_file, _ in pending:
        pending_per_channel[txt_file.parent.name] = pending_per_channel.get(txt_file.parent.name, 0) + 1

    pending.sort(key=lambda pair: (pending_per_channel[pair[0].parent.name], pair[0].parent.name, pair[0].name))

    return pending, len(txt_files), already_done, empty_files


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
    start_time = time.monotonic()

    worker_numbers = itertools.count(1)
    worker_number_lock = threading.Lock()
    worker_local = threading.local()
    eta_completed_count = 0
    eta_completed_count_lock = threading.Lock()

    def initialize_worker() -> None:
        with worker_number_lock:
            worker_local.worker_id = f"worker-{next(worker_numbers):02d}"

    def current_worker_id() -> str:
        return str(worker_local.worker_id)

    def run_pending_item(item_number: int, txt_file: Path, output_file: Path) -> tuple[str, str, str | None]:
        nonlocal eta_completed_count
        worker_id = current_worker_id()
        rel_path = f"{txt_file.parent.name}/{txt_file.name}"
        with eta_completed_count_lock:
            completed_count_snapshot = eta_completed_count

        log_processing_start(
            item_number,
            total,
            rel_path,
            start_time,
            completed_count_snapshot,
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
        with eta_completed_count_lock:
            eta_completed_count += 1
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
