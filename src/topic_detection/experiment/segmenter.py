"""LLM topic segmentation of SRT transcripts via litellm."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import litellm
from litellm.exceptions import BadRequestError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.util.srt_util import SRTEntry


class TopicSpan(BaseModel):
    """One contiguous topic span defined by inclusive utterance indices."""

    start_index: int = Field(..., ge=1, description="1-based utterance index where topic begins (inclusive)")
    end_index: int = Field(..., ge=1, description="1-based utterance index where topic ends (inclusive)")
    title: str = Field(..., min_length=1, description="Short topic title")

    model_config = ConfigDict(frozen=True, extra="forbid")


class LLMSegmentationResult(BaseModel):
    """Parsed LLM response for a single SRT file."""

    topics: list[TopicSpan] = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True)
class SegmenterOptions:
    """Runtime parameters for the LLM segmenter."""

    model: str
    api_base: str
    api_key: str
    temperature: float
    max_output_tokens: int
    max_retries: int
    retry_delay: float


@dataclass(frozen=True)
class SegmentationRunResult:
    """Outcome of a segmentation call, including retry history."""

    topics: LLMSegmentationResult
    attempts_made: int
    prior_errors: list[str]


_SYSTEM_PROMPT = """You label topic boundaries in a video transcript.

You receive a numbered list of utterances [1], [2], [3], ... [N].
Group consecutive utterances into contiguous semantic topics.

Rules:
- Every utterance from 1 to N belongs to exactly one topic.
- Topics must cover 1..N with no gaps and no overlaps.
- Topic sizes should VARY and reflect real semantic shifts. Do NOT partition uniformly by count.
- Detect topic boundaries using discourse markers ("but", "now", "let's", "what about", "so"),
  explicit topic announcements, subject changes, and example-vs-argument transitions.
- Each topic title: 3-8 words, specific to the content.

Respond with ONLY a JSON object in this exact shape (no markdown fences, no prose, no comments):

{
  "topics": [
    {"start_index": 1, "end_index": 4, "title": "Opening hook and framing"},
    {"start_index": 5, "end_index": 12, "title": "First supporting example"},
    {"start_index": 13, "end_index": 20, "title": "Main argument"}
  ]
}

The top-level MUST be an object with a single key "topics" whose value is an array.
Do NOT return a bare array. Do NOT wrap the response in markdown fences."""


def _format_timestamp(td: timedelta) -> str:
    """Render timedelta as HH:MM:SS,mmm (SRT style)."""
    total_ms = int(td.total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_prompt(*, entries: list[SRTEntry]) -> str:
    """Build the user prompt for topic segmentation."""
    if not entries:
        raise ValueError("Cannot build prompt for empty entries list")

    lines = ["Utterances:"]
    for i, entry in enumerate(entries, start=1):
        content = entry.content.replace("\n", " ").strip()
        lines.append(f"[{i}] {content}")
    lines.append("")
    lines.append(f"Group these {len(entries)} utterances into contiguous topics.")
    return "\n".join(lines)


def _strip_llm_wrapping(text: str) -> str:
    """Strip <think> tags, markdown code fences, and YAML-style `---` separators."""
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
    cleaned = cleaned.strip()

    if cleaned.startswith("---"):
        newline_pos = cleaned.find("\n")
        if newline_pos != -1:
            cleaned = cleaned[newline_pos + 1 :].strip()
    if cleaned.endswith("---"):
        cleaned = cleaned[:-3].rstrip()

    return cleaned


_TRANSIENT_NETWORK_MARKERS = (
    "Tailscale",
    "WebSocket",
    "keepalive",
    "peer_keepalive_timeout",
    "Connection aborted",
    "Connection reset",
    "Read timed out",
    "ECONNRESET",
)


def _is_transient_network_error(error_msg: str) -> bool:
    """Return True if ``error_msg`` looks like a retryable network/transport flake."""
    return any(marker in error_msg for marker in _TRANSIENT_NETWORK_MARKERS)


def _parse_response(response_text: str) -> LLMSegmentationResult:
    cleaned = _strip_llm_wrapping(response_text)
    if not cleaned:
        raise ValueError(f"Empty LLM response after cleaning. Raw (first 500): {response_text[:500]}")

    try:
        data = json.loads(cleaned, strict=False)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM response is not valid JSON: {e}. Response (first 500): {response_text[:500]}") from e

    try:
        return LLMSegmentationResult.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"LLM response does not match schema: {e}. Data: {data}") from e


def _validate_coverage(*, result: LLMSegmentationResult, n_utterances: int) -> None:
    """Verify topics cover [1..n_utterances] contiguously without gaps or overlaps."""
    if not result.topics:
        raise ValueError("LLM returned no topics")

    ordered = sorted(result.topics, key=lambda t: t.start_index)
    expected_start = 1
    for topic in ordered:
        if topic.start_index != expected_start:
            raise ValueError(
                f"Topic coverage gap/overlap: expected start_index={expected_start}, "
                f"got topic {topic.start_index}-{topic.end_index} '{topic.title}'"
            )
        if topic.end_index < topic.start_index:
            raise ValueError(f"Topic has end_index<{topic.start_index}: {topic}")
        expected_start = topic.end_index + 1

    if expected_start - 1 != n_utterances:
        raise ValueError(f"Topics do not cover all {n_utterances} utterances; last covered index={expected_start - 1}")


def segment_srt_entries(
    *,
    entries: list[SRTEntry],
    options: SegmenterOptions,
) -> SegmentationRunResult:
    """Call the LLM to segment a list of SRT entries into topics.

    Returns the parsed topics plus retry metadata (attempts used, prior errors).
    Raises ValueError after exhausting ``options.max_retries``.
    """
    if not entries:
        raise ValueError("segment_srt_entries: entries must not be empty")

    user_prompt = build_prompt(entries=entries)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prior_errors: list[str] = []
    last_error: Exception | None = None
    for attempt in range(1, options.max_retries + 1):
        print(f"      LLM segment attempt {attempt}/{options.max_retries}...")
        try:
            response = litellm.completion(
                model=options.model,
                messages=messages,
                api_base=options.api_base,
                api_key=options.api_key,
                max_tokens=options.max_output_tokens,
                temperature=options.temperature,
            )
            response_text = response.choices[0].message.content
            if response_text is None or not response_text.strip():
                last_error = ValueError("LLM returned empty response")
            else:
                try:
                    parsed = _parse_response(response_text)
                    _validate_coverage(result=parsed, n_utterances=len(entries))
                    if attempt > 1:
                        print(f"      LLM segment attempt {attempt}/{options.max_retries}: success after {attempt - 1} retry(ies)")
                    return SegmentationRunResult(topics=parsed, attempts_made=attempt, prior_errors=prior_errors)
                except ValueError as e:
                    last_error = e
        except BadRequestError as e:
            error_msg = str(e)
            if "No models loaded" in error_msg:
                raise BadRequestError(
                    message=(
                        f"No models loaded in LM Studio. Expected model: {options.model}. "
                        f"Load it with: lms load {options.model.split('/')[-1]}"
                    ),
                    model=options.model,
                    llm_provider="openai",
                ) from e
            if _is_transient_network_error(error_msg):
                last_error = e
            else:
                raise

        prior_errors.append(str(last_error))
        print(f"      LLM segment attempt {attempt}/{options.max_retries} FAILED: {last_error}")
        if attempt < options.max_retries:
            time.sleep(options.retry_delay)

    raise ValueError(f"LLM segmentation failed after {options.max_retries} attempts. Errors: {prior_errors}")


def build_output_document(
    *,
    srt_path_str: str,
    model_id: str,
    entries: list[SRTEntry],
    input_tokens: int,
    run: SegmentationRunResult,
    generated_at_iso: str,
) -> dict[str, Any]:
    """Assemble the JSON document written per processed SRT file."""
    ordered_topics = sorted(run.topics.topics, key=lambda t: t.start_index)

    boundaries = [0] * len(entries)
    for topic in ordered_topics:
        boundaries[topic.start_index - 1] = 1

    segments: list[dict[str, Any]] = []
    for seg_id, topic in enumerate(ordered_topics, start=1):
        span = entries[topic.start_index - 1 : topic.end_index]
        text = " ".join(e.content.replace("\n", " ").strip() for e in span)
        segments.append(
            {
                "segment_id": seg_id,
                "title": topic.title,
                "start_index": topic.start_index,
                "end_index": topic.end_index,
                "start_timestamp": _format_timestamp(entries[topic.start_index - 1].start),
                "end_timestamp": _format_timestamp(entries[topic.end_index - 1].end),
                "text": text,
            }
        )

    return {
        "source_file": srt_path_str,
        "generated_at": generated_at_iso,
        "model": model_id,
        "total_entries": len(entries),
        "input_tokens": input_tokens,
        "attempts_made": run.attempts_made,
        "prior_errors": list(run.prior_errors),
        "boundaries": boundaries,
        "segments": segments,
    }
