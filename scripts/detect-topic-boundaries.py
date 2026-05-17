#!/usr/bin/env python3
"""Detect topic boundaries in cleaned TXT transcripts using an LLM."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import litellm
import tiktoken
from litellm.exceptions import BadRequestError
from pydantic import RootModel, ValidationError, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config, LLMConfig
from src.util.fs_util import FSUtil
from src.util.log_util import configure_root_logger, get_logger

logger = get_logger(__name__)


class TopicBoundaryResponse(RootModel[list[int]]):
    """Strict response shape for topic boundaries."""

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("Boundary list must not be empty")
        if value[0] != 1:
            raise ValueError("First boundary must be 1")
        for idx in range(1, len(value)):
            if value[idx] <= value[idx - 1]:
                raise ValueError("Boundary numbers must be strictly increasing")
        return value


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Detect topic boundaries in cleaned transcript TXT files")
    parser.add_argument(
        "--file",
        type=Path,
        help="Optional specific TXT transcript file to process (must be inside configured input_dir)",
    )
    return parser.parse_args()


def strip_llm_wrapping(text: str) -> str:
    """Strip think tags and markdown fences from model responses."""
    cleaned = text.strip()

    think_match = re.search(r"</think>\s*(.*)$", cleaned, re.DOTALL)
    if think_match:
        cleaned = think_match.group(1).strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def split_sentences(transcript: str) -> list[str]:
    """Split transcript text into numbered sentence candidates."""
    merged = " ".join(line.strip() for line in transcript.splitlines() if line.strip())
    if not merged:
        raise ValueError("Transcript has no non-empty content")

    rough = re.split(r"(?<=[.!?])\s+", merged)
    sentences = [s.strip() for s in rough if s.strip()]
    if not sentences:
        raise ValueError("Failed to derive sentences from transcript")
    return sentences


def build_numbered_sentences(sentences: list[str]) -> str:
    """Format sentences as `N: sentence` lines for the prompt."""
    return "\n".join(f"{index}: {sentence}" for index, sentence in enumerate(sentences, start=1))


def build_prompt(prompt_template: str, txt_file: Path, numbered_sentences: str) -> str:
    """Render prompt template with file path and sentence list."""
    return (
        prompt_template.replace("{{INPUT_TXT_FILE}}", str(txt_file)).replace("{{NUMBERED_SENTENCES}}", numbered_sentences)
    )


def call_llm(prompt: str, llm: LLMConfig) -> str:
    """Call LLM with plain text prompt and return raw response text."""
    try:
        response = litellm.completion(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
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
    if response_text is None or not response_text.strip():
        raise ValueError("LLM returned empty response")
    return response_text.strip()


def parse_and_validate_boundaries(raw_response: str, sentence_count: int) -> list[int]:
    """Parse JSON array and validate type + bounds."""
    cleaned = strip_llm_wrapping(raw_response)

    try:
        decoded = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e.msg} at line {e.lineno} col {e.colno}") from e

    try:
        validated = TopicBoundaryResponse.model_validate(decoded)
    except ValidationError as e:
        raise ValueError(f"LLM output failed schema validation: {e}") from e

    boundaries = validated.root
    for boundary in boundaries:
        if boundary > sentence_count:
            raise ValueError(f"Boundary {boundary} exceeds sentence count {sentence_count}")

    return boundaries


def output_path_for(txt_file: Path, input_dir: Path, output_dir: Path) -> Path:
    """Build output path mirroring source channel/file structure."""
    relative = txt_file.relative_to(input_dir)
    return output_dir / relative.parent / f"{txt_file.stem}_topic_boundaries.json"


def collect_target_files(input_dir: Path, target_file: Path | None) -> list[Path]:
    """Collect all input files or a single requested file."""
    if target_file is not None:
        resolved = target_file.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Specified file not found: {resolved}")
        if resolved.suffix.lower() != ".txt":
            raise ValueError(f"Specified file is not a .txt file: {resolved}")
        try:
            resolved.relative_to(input_dir.resolve())
        except ValueError as e:
            raise ValueError(f"Specified file must be inside input directory {input_dir}: {resolved}") from e
        if resolved.name.startswith("._"):
            raise ValueError(f"AppleDouble files are not supported: {resolved}")
        return [resolved]

    txt_files = sorted(p for p in input_dir.rglob("*.txt") if not p.name.startswith("._"))
    return txt_files


def process_single_file(
    txt_file: Path,
    input_dir: Path,
    output_dir: Path,
    prompt_template: str,
    llm: LLMConfig,
    encoder: tiktoken.Encoding,
) -> None:
    """Process a single transcript file and write topic boundaries."""
    output_file = output_path_for(txt_file, input_dir, output_dir)
    if output_file.exists():
        logger.info(f"Skipping existing output: {output_file}")
        return

    transcript = FSUtil.read_text_file(txt_file)
    sentences = split_sentences(transcript)
    numbered_sentences = build_numbered_sentences(sentences)
    prompt = build_prompt(prompt_template, txt_file, numbered_sentences)

    input_tokens = len(encoder.encode(prompt, disallowed_special=()))
    threshold = int(llm.context_window * llm.context_window_threshold / 100)
    if input_tokens > threshold:
        raise ValueError(
            f"Input tokens ({input_tokens:,}) exceed {llm.context_window_threshold}% of context window ({threshold:,})"
        )

    last_error: str | None = None
    for attempt in range(1, llm.max_retries + 1):
        try:
            raw_response = call_llm(prompt, llm)
            boundaries = parse_and_validate_boundaries(raw_response, len(sentences))
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(boundaries, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(
                f"Detected {len(boundaries)} boundary point(s) for {txt_file.parent.name}/{txt_file.name} -> {output_file.name}"
            )
            return
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt}/{llm.max_retries} failed for {txt_file.name}: {last_error}")
            if attempt < llm.max_retries:
                time.sleep(llm.retry_delay)

    raise RuntimeError(f"Boundary detection failed for {txt_file}: {last_error}")


def main() -> int:
    """Run topic boundary detection for configured transcripts."""
    args = parse_args()

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    config = Config(config_path)
    configure_root_logger(config.getDataLogsDir())

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    boundary_cfg = config.get_topic_boundaries_config()
    llm = boundary_cfg.llm
    input_dir = config.getDataDir() / boundary_cfg.input_dir
    output_dir = config.getDataDir() / boundary_cfg.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    prompt_path = Path(__file__).parent.parent / boundary_cfg.prompt_template
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    prompt_template = FSUtil.read_text_file(prompt_path)
    encoder = tiktoken.get_encoding(config.getEncodingName())

    target_file = args.file.resolve() if args.file is not None else None
    txt_files = collect_target_files(input_dir, target_file)
    if not txt_files:
        raise RuntimeError(f"No .txt files found in input directory: {input_dir}")

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {llm.model}")
    logger.info(f"Files to process: {len(txt_files)}")

    start_time = time.monotonic()
    for index, txt_file in enumerate(txt_files, start=1):
        logger.info(f"[{index}/{len(txt_files)}] Processing {txt_file.parent.name}/{txt_file.name}")
        process_single_file(txt_file, input_dir, output_dir, prompt_template, llm, encoder)

    elapsed = time.monotonic() - start_time
    logger.info(f"Completed boundary detection for {len(txt_files)} file(s) in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
