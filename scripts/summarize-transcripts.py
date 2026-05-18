#!/usr/bin/env python3
"""Summarize cleaned transcripts using an LLM hosted by LM Studio."""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import litellm
import tiktoken
from litellm.exceptions import BadRequestError

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config, LLMConfig, SummarizeTranscriptsConfig
from src.util.fs_util import FSUtil
from src.util.log_util import configure_root_logger, get_logger

logger = get_logger(__name__)


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
    configure_root_logger(config.getDataLogsDir())

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        summarize_cfg = config.get_summarize_transcripts_config()
    except KeyError as e:
        logger.error(f"Error: {e}")
        return 1

    cleaned_dir = config.getDataDownloadsTranscriptsCleanedDir()
    summaries_dir = config.getDataDownloadsTranscriptsSummariesDir()

    if not cleaned_dir.exists():
        logger.error(f"Cleaned transcripts directory not found: {cleaned_dir}")
        return 1

    prompt_path = Path(__file__).parent.parent / "prompts" / "summarize-transcript.md"
    if not prompt_path.exists():
        logger.error(f"Prompt template not found: {prompt_path}")
        return 1

    prompt_template = FSUtil.read_text_file(prompt_path)
    encoding_name = config.getEncodingName()

    logger.info(f"Cleaned transcripts directory: {cleaned_dir}")
    logger.info(f"Summaries output directory: {summaries_dir}")
    logger.info(f"Model: {summarize_cfg.llm.model}")
    logger.info(f"Context window: {summarize_cfg.llm.context_window:,} tokens")

    return summarize_cfg, cleaned_dir, summaries_dir, prompt_template, encoding_name


def strip_think_tags(text: str) -> str:
    """Strip <think> reasoning tags from model responses."""
    think_match = re.search(r"</think>\s*(.*)$", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return text


def call_llm(prompt: str, llm: LLMConfig) -> str:
    """Call the LLM and return the response text."""
    messages = [{"role": "user", "content": prompt}]

    try:
        response = litellm.completion(
            model=llm.model,
            messages=messages,
            api_base=llm.api_base,
            api_key=llm.api_key,
            max_tokens=llm.max_tokens,
            temperature=llm.temperature,
        )
    except BadRequestError as e:
        error_msg = str(e)
        if "No models loaded" in error_msg:
            raise RuntimeError(
                f"No models loaded in LM Studio. Expected model: {llm.model}. Load it with: lms load {llm.model.split('/')[-1]}"
            ) from e
        raise

    response_text = response.choices[0].message.content
    if response_text is None or response_text.strip() == "":
        raise ValueError("LLM returned empty response")

    return strip_think_tags(response_text.strip())


def process_single_file(
    txt_file: Path,
    output_file: Path,
    prompt_template: str,
    llm: LLMConfig,
    encoder: tiktoken.Encoding,
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

    input_tokens = len(encoder.encode(prompt, disallowed_special=()))
    threshold = int(llm.context_window * llm.context_window_threshold / 100)
    if input_tokens > threshold:
        return (
            "skip (oversized)",
            f"Input tokens ({input_tokens:,}) exceed {llm.context_window_threshold}% of context window ({threshold:,})",
        )

    for attempt in range(1, llm.max_retries + 1):
        try:
            summary = call_llm(prompt, llm)
            FSUtil.write_text_file(output_file, summary, create_parents=True)
            return "ok", None
        except Exception as e:
            logger.warning(f"  Attempt {attempt}/{llm.max_retries} failed: {e}")
            if attempt == llm.max_retries:
                raise
            time.sleep(llm.retry_delay)

    raise RuntimeError("unreachable")


def format_eta(seconds_remaining: float) -> str:
    """Format remaining duration as hours and minutes."""
    total_minutes = max(0, int(seconds_remaining // 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours}h{minutes}m"


def collect_pending_files(
    cleaned_dir: Path,
    summaries_dir: Path,
) -> tuple[list[tuple[Path, Path]], int, int, int]:
    """Scan for transcript files and partition into pending/done/empty."""
    txt_files = FSUtil.find_files_by_extension(cleaned_dir, ".txt", recursive=True)
    txt_files = [f for f in txt_files if not f.name.startswith("._")]

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


def process_pending(
    pending: list[tuple[Path, Path]],
    prompt_template: str,
    llm: LLMConfig,
    encoder: tiktoken.Encoding,
) -> None:
    """Process all pending files. Raises on persistent LLM failure."""
    success_count = 0
    skipped_count = 0
    start_time = time.monotonic()

    for idx, (txt_file, output_file) in enumerate(pending, start=1):
        completed = idx - 1
        elapsed = time.monotonic() - start_time
        avg = (elapsed / completed) if completed > 0 else 0.0
        eta_seconds = avg * (len(pending) - completed)

        rel_path = f"{txt_file.parent.name}/{txt_file.name}"
        logger.info(f"[{idx}/{len(pending)}] {idx / len(pending) * 100:5.1f}% ETA {format_eta(eta_seconds)}  {rel_path}")

        status, note = process_single_file(txt_file, output_file, prompt_template, llm, encoder)
        if status != "ok":
            logger.info(f"  -> {status}")
            skipped_count += 1
        else:
            success_count += 1

        if note:
            logger.warning(f"  Note: {note}")

    logger.info("---")
    logger.info("=" * 50)
    logger.info("Summary")
    logger.info("=" * 50)
    logger.info(f"Total pending: {len(pending)}")
    logger.info(f"Summarized: {success_count}")
    logger.info(f"Skipped: {skipped_count}")


def main() -> int:
    """Main entry point."""
    setup_result = setup_environment()
    if isinstance(setup_result, int):
        return setup_result
    summarize_cfg, cleaned_dir, summaries_dir, prompt_template, encoding_name = setup_result

    encoder = tiktoken.get_encoding(encoding_name)
    llm = summarize_cfg.llm

    pending, total, already_done, empty_files = collect_pending_files(cleaned_dir, summaries_dir)

    if total == 0:
        logger.error(f"No .txt files found in: {cleaned_dir}")
        return 1

    logger.info(f"Found {total} transcript(s), {already_done} already summarized, {empty_files} empty, {len(pending)} pending")
    logger.info("---")

    if not pending:
        logger.info("Nothing to do — all files already processed.")
        return 0

    process_pending(pending, prompt_template, llm, encoder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
