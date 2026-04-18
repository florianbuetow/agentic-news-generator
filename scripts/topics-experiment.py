#!/usr/bin/env python3
"""Experimental LLM topic segmentation of de-hallucinated SRT files.

Walks config-defined channels, picks up to ``MAX_FILES_PER_CHANNEL`` .srt files
(alphabetical) under each channel's cleaned-transcripts directory, selects the
largest LM Studio model that fits the longest of those files, then calls the
LLM to label topic boundaries on each one. Resume works via skip-if-exists.

Usage:
    uv run python scripts/topics-experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import tiktoken

from src.config import ChannelConfig, Config, TopicsExperimentConfig
from src.topic_detection.experiment.lmstudio_models import (
    LMStudioModel,
    fetch_lmstudio_models,
    select_best_model,
)
from src.topic_detection.experiment.segmenter import (
    SegmenterOptions,
    build_output_document,
    segment_srt_entries,
)
from src.util.channel_name import sanitize_channel_name
from src.util.srt_util import SRTUtil

MAX_FILES_PER_CHANNEL = 50


@dataclass(frozen=True)
class SRTWorkItem:
    """A single SRT file scheduled for processing."""

    channel: ChannelConfig
    channel_dir: Path
    srt_path: Path


@dataclass(frozen=True)
class CollectResult:
    """Result of walking the cleaned-transcripts root.

    ``max_bytes_tokens`` is the token count of the largest-by-bytesize SRT file,
    computed in a single pass via incremental tokenization whenever a new max
    is seen during the walk.
    """

    work_items: list[SRTWorkItem]
    max_bytes_path: Path | None
    max_bytes_size: int
    max_bytes_tokens: int


def collect_srt_files(*, config: Config, encoder: tiktoken.Encoding) -> CollectResult:
    """Collect up to ``MAX_FILES_PER_CHANNEL`` de-hallucinated SRTs per channel.

    Picks the alphabetically-first non-AppleDouble ``.srt`` files under each
    channel directory (capped at ``MAX_FILES_PER_CHANNEL``). Tokenizes each
    picked file on new byte-size max to find the largest candidate across the
    collected set (used to size the LLM context window).
    """
    cleaned_root = config.getDataDownloadsTranscriptsCleanedDir()
    if not cleaned_root.exists():
        raise FileNotFoundError(f"Cleaned transcripts directory not found: {cleaned_root}")

    work_items: list[SRTWorkItem] = []
    max_bytes_path: Path | None = None
    max_bytes_size = -1
    max_bytes_tokens = 0

    for channel in config.get_channels():
        channel_dir = cleaned_root / sanitize_channel_name(channel.name)
        if not channel_dir.exists():
            continue

        srt_files = sorted(p for p in channel_dir.rglob("*.srt") if not p.name.startswith("._"))
        for srt_path in srt_files[:MAX_FILES_PER_CHANNEL]:
            work_items.append(SRTWorkItem(channel=channel, channel_dir=channel_dir, srt_path=srt_path))

            size = srt_path.stat().st_size
            if size > max_bytes_size:
                text = srt_path.read_text(encoding="utf-8", errors="replace")
                token_count = len(encoder.encode(text, disallowed_special=()))
                max_bytes_size = size
                max_bytes_tokens = token_count
                max_bytes_path = srt_path

    return CollectResult(
        work_items=work_items,
        max_bytes_path=max_bytes_path,
        max_bytes_size=max_bytes_size if max_bytes_size >= 0 else 0,
        max_bytes_tokens=max_bytes_tokens,
    )


def compute_required_tokens(*, base_tokens: int, cfg: TopicsExperimentConfig) -> int:
    """Apply safety factor + prompt overhead + output budget to input tokens."""
    inflated = int(base_tokens * cfg.token_safety_factor)
    return inflated + cfg.prompt_overhead_tokens + cfg.max_output_tokens


def pick_model(*, cfg: TopicsExperimentConfig, required_tokens: int) -> LMStudioModel:
    """Query LM Studio for available models and pick the best fit."""
    models = fetch_lmstudio_models(
        models_endpoint=cfg.models_endpoint,
        timeout=cfg.models_http_timeout,
    )
    return select_best_model(
        models=models,
        required_tokens=required_tokens,
        prefer_loaded=cfg.prefer_loaded,
    )


def output_path_for(*, item: SRTWorkItem, output_root: Path) -> Path:
    """Deterministic output path: <output_root>/<channel>/<srt_stem>.json."""
    channel_slug = sanitize_channel_name(item.channel.name)
    return output_root / channel_slug / f"{item.srt_path.stem}.json"


def process_srt_file(
    *,
    item: SRTWorkItem,
    model: LMStudioModel,
    options: SegmenterOptions,
    output_root: Path,
    encoder: tiktoken.Encoding,
) -> str:
    """Parse SRT, call LLM, write JSON output. Returns a status string."""
    out_path = output_path_for(item=item, output_root=output_root)
    if out_path.exists():
        return "skip (exists)"

    entries = SRTUtil.parse_srt_file(item.srt_path)
    if not entries:
        return "skip (empty)"

    srt_text = item.srt_path.read_text(encoding="utf-8", errors="replace")
    input_tokens = len(encoder.encode(srt_text, disallowed_special=()))

    run_result = segment_srt_entries(entries=entries, options=options)

    doc = build_output_document(
        srt_path_str=str(item.srt_path),
        model_id=model.model_id,
        entries=entries,
        input_tokens=input_tokens,
        run=run_result,
        generated_at_iso=datetime.now(UTC).isoformat(),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)

    return f"ok ({len(run_result.topics.topics)} topics, {run_result.attempts_made} attempt(s))"


def count_file_tokens(*, path: Path, encoder: tiktoken.Encoding) -> int:
    """Return the token count for the full text content of a file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(encoder.encode(text, disallowed_special=()))


def format_eta(seconds_remaining: float) -> str:
    r"""Format a remaining duration as ``\d+h\d+m`` (hours + minutes, floored)."""
    total_minutes = max(0, int(seconds_remaining // 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours}h{minutes}m"


def format_progress(*, index: int, total: int, eta_seconds: float) -> str:
    r"""Build the progress prefix: ``[n/total] xx.x% ETA \d+h\d+m``."""
    percent = (index / total * 100.0) if total > 0 else 0.0
    return f"[{index}/{total}] {percent:5.1f}% ETA {format_eta(eta_seconds)}"


def main() -> int:
    """Run the experimental topic segmentation: up to N SRTs per channel."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config(config_path)
    experiment_cfg = config.get_topics_experiment_config()

    encoder = tiktoken.get_encoding(config.getEncodingName())

    print("Walking cleaned-transcripts root and measuring largest SRT...")
    collect = collect_srt_files(config=config, encoder=encoder)
    if not collect.work_items:
        print("No SRT files found to process.")
        return 0

    if collect.max_bytes_path is None:
        raise RuntimeError("No maximum-sized SRT file was identified (unexpected).")

    total_found = len(collect.work_items)
    print(f"Found {total_found} SRT file(s) across {len(config.get_channels())} channel(s).")
    print(f"Largest SRT: {collect.max_bytes_path.name} ({collect.max_bytes_size:,} bytes, {collect.max_bytes_tokens:,} tokens)")

    required_tokens_info = compute_required_tokens(base_tokens=collect.max_bytes_tokens, cfg=experiment_cfg)
    print(
        f"Required context window for largest file: {required_tokens_info:,} tokens "
        f"(safety {experiment_cfg.token_safety_factor}x + prompt {experiment_cfg.prompt_overhead_tokens} "
        f"+ output {experiment_cfg.max_output_tokens}) — files exceeding 90% of selected ctx are skipped"
    )

    print("Querying LM Studio for available models...")
    model = pick_model(cfg=experiment_cfg, required_tokens=1)
    print(
        f"Selected model: {model.model_id} (type={model.model_type}, state={model.state}, "
        f"ctx={model.effective_context_length:,}, arch_max={model.max_context_length:,})"
    )

    options = SegmenterOptions(
        model=f"openai/{model.model_id}",
        api_base=experiment_cfg.api_base,
        api_key=experiment_cfg.api_key,
        temperature=experiment_cfg.temperature,
        max_output_tokens=experiment_cfg.max_output_tokens,
        max_retries=experiment_cfg.max_retries,
        retry_delay=experiment_cfg.retry_delay,
    )

    output_root = config.getDataOutputDir() / experiment_cfg.output_subdir
    print(f"Output root: {output_root}")

    print("Pre-scanning pending files (detecting already-processed and empty SRTs)...")
    pending: list[SRTWorkItem] = []
    already_done = 0
    empty_srts = 0
    for item in collect.work_items:
        if output_path_for(item=item, output_root=output_root).exists():
            already_done += 1
            continue
        entries = SRTUtil.parse_srt_file(item.srt_path)
        if not entries:
            empty_srts += 1
            continue
        pending.append(item)

    print(
        f"Status: {already_done}/{total_found} already processed, "
        f"{empty_srts} empty (both skipped, excluded from progress/ETA), "
        f"{len(pending)} pending"
    )
    print()

    total = len(pending)
    if total == 0:
        print("Nothing to do — all files already processed.")
        return 0

    start_time = time.monotonic()

    for idx, item in enumerate(pending, start=1):
        completed = idx - 1
        elapsed = time.monotonic() - start_time
        avg = (elapsed / completed) if completed > 0 else 0.0
        eta_seconds = avg * (total - completed)

        prefix = format_progress(index=idx, total=total, eta_seconds=eta_seconds)
        rel = item.srt_path.relative_to(item.channel_dir)
        print(f"{prefix}  {item.channel.name}/{rel}")

        file_tokens = count_file_tokens(path=item.srt_path, encoder=encoder)
        ctx_threshold = int(model.effective_context_length * 0.9)
        if file_tokens > ctx_threshold:
            print(
                f"          -> skip (oversized: {file_tokens:,} tokens > 90% of ctx {model.effective_context_length:,} = {ctx_threshold:,})"
            )
            continue

        status = process_srt_file(
            item=item,
            model=model,
            options=options,
            output_root=output_root,
            encoder=encoder,
        )
        print(f"          -> {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
