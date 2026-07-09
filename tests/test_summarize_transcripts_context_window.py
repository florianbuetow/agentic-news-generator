"""Tests for summarize-transcripts LM Studio context-window resolution."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Literal

import pytest

from src.config import LLMConfig


def load_summarize_transcripts_module() -> Any:
    script_path = Path(__file__).parent.parent / "scripts" / "summarize-transcripts.py"
    spec = importlib.util.spec_from_file_location("summarize_transcripts", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODEL: str = "openai/qwen/qwen3.6-35b-a3b:2"


def make_llm(*, context_window: int | Literal["auto"] = "auto") -> LLMConfig:
    return LLMConfig(
        models=["openai/qwen/qwen3.6-35b-a3b", _MODEL],
        api_base="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        context_window=context_window,
        max_tokens=32768,
        temperature=0.3,
        context_window_threshold=90,
        max_retries=1,
        retry_delay=1.0,
    )


def test_parallel_context_window_divides_loaded_context_by_parallelism() -> None:
    module = load_summarize_transcripts_module()

    assert module.resolve_parallel_context_window(32678, 4) == 8169
    assert module.resolve_parallel_context_window(32678, 1) == 32678


def test_process_single_file_uses_effective_context_window_for_skip_decision(tmp_path: Path) -> None:
    module = load_summarize_transcripts_module()
    txt_file = tmp_path / "transcript.txt"
    output_file = tmp_path / "summary.md"
    txt_file.write_text("tiny transcript", encoding="utf-8")

    class FakeEncoder:
        def encode(self, transcript: str, disallowed_special: tuple[object, ...]) -> list[int]:
            assert transcript == "tiny transcript"
            assert disallowed_special == ()
            return [0, 1, 2, 3, 4, 5]

    status, note = module.process_single_file(
        txt_file,
        output_file,
        "{transcript}",
        make_llm(context_window=1000),
        _MODEL,
        FakeEncoder(),
        10,
        50,
    )

    assert status == "skip (oversized)"
    assert note is not None
    assert note == "prompt=6 > limit=5"


def test_process_single_file_counts_prompt_tokens_for_skip_decision(tmp_path: Path) -> None:
    module = load_summarize_transcripts_module()
    txt_file = tmp_path / "transcript.txt"
    output_file = tmp_path / "summary.md"
    txt_file.write_text("tiny transcript", encoding="utf-8")

    class FakeEncoder:
        def encode(self, text: str, disallowed_special: tuple[object, ...]) -> list[int]:
            if text == "tiny transcript":
                return [0, 1]
            return list(range(9))

    status, note = module.process_single_file(
        txt_file,
        output_file,
        "prompt wrapper {transcript}",
        make_llm(context_window=1000),
        _MODEL,
        FakeEncoder(),
        10,
        80,
    )

    assert status == "skip (oversized)"
    assert note is not None
    assert note == "prompt=9 > limit=8"


def test_process_single_file_caps_max_tokens_to_remaining_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_summarize_transcripts_module()
    txt_file = tmp_path / "transcript.txt"
    output_file = tmp_path / "summary.md"
    txt_file.write_text("tiny transcript", encoding="utf-8")
    requested_max_tokens: list[int] = []

    class FakeEncoder:
        def encode(self, text: str, disallowed_special: tuple[object, ...]) -> list[int]:
            if text == "tiny transcript":
                return [0, 1]
            return list(range(12))

    def complete(prompt: str, llm: LLMConfig, model: str, max_tokens: int) -> str:
        requested_max_tokens.append(max_tokens)
        return "summary"

    monkeypatch.setattr(module, "call_llm", complete)

    status, note = module.process_single_file(
        txt_file,
        output_file,
        "prompt wrapper {transcript}",
        make_llm(context_window=1000),
        _MODEL,
        FakeEncoder(),
        20,
        80,
    )

    assert status == "ok"
    assert note is None
    assert requested_max_tokens == [8]
    assert output_file.read_text(encoding="utf-8") == "summary"


def test_process_single_file_does_not_swallow_context_size_exceeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_summarize_transcripts_module()
    txt_file = tmp_path / "transcript.txt"
    output_file = tmp_path / "summary.md"
    txt_file.write_text("short enough locally", encoding="utf-8")
    calls = 0

    class FakeEncoder:
        def encode(self, transcript: str, disallowed_special: tuple[object, ...]) -> list[int]:
            return [0, 1, 2]

    def fail_with_context_error(prompt: str, llm: LLMConfig, model: str, max_tokens: int) -> str:
        nonlocal calls
        calls += 1
        raise module.ContextSizeExceededError("Context size has been exceeded")

    monkeypatch.setattr(module, "call_llm", fail_with_context_error)

    with pytest.raises(module.ContextSizeExceededError) as exc_info:
        module.process_single_file(
            txt_file,
            output_file,
            "{transcript}",
            make_llm(context_window=1000),
            _MODEL,
            FakeEncoder(),
            100,
            80,
        )

    assert "prompt=3, ctx=100" in str(exc_info.value)
    assert calls == 1
    assert not output_file.exists()


def test_processing_start_log_includes_worker_prefix_when_parallel(caplog: pytest.LogCaptureFixture) -> None:
    module = load_summarize_transcripts_module()

    caplog.set_level(logging.INFO)

    module.log_processing_start(
        2,
        2855,
        "Greg_Isenberg/Tutorial on how to write tweets with man who gets 1B+ impressions per year [ovB8nX_hUe8].txt",
        1,
        60.0,
        "worker-01",
        4,
    )

    assert "WORKER-1/4 Processing [2/2855 0.1%] [ETA " in caplog.text
    assert "... Greg_Isenberg/Tutorial on how to write tweets with man who gets 1B+ impressions per year [ovB8nX_hUe8].txt" in caplog.text


def test_processing_start_log_omits_worker_prefix_when_single_worker(caplog: pytest.LogCaptureFixture) -> None:
    module = load_summarize_transcripts_module()

    caplog.set_level(logging.INFO)

    module.log_processing_start(
        1,
        3,
        "channel/first.txt",
        0,
        0.0,
        "worker-01",
        1,
    )

    assert "Processing [1/3 33.3%] [ETA calculating] ... channel/first.txt" in caplog.text
    assert "WORKER-1/1" not in caplog.text


def test_record_process_result_only_updates_counts_and_collections(caplog: pytest.LogCaptureFixture) -> None:
    module = load_summarize_transcripts_module()
    oversized_files: list[str] = []
    failures: list[tuple[str, str]] = []

    caplog.set_level(logging.INFO)

    assert module.record_process_result(
        "channel/ok.txt",
        ("", "ok", None),
        oversized_files,
        failures,
    ) == (1, 0)
    assert module.record_process_result(
        "channel/large.txt",
        ("", "skip (oversized)", "prompt=9 > limit=8"),
        oversized_files,
        failures,
    ) == (0, 1)
    assert module.record_process_result(
        "channel/fail.txt",
        ("", "context exceeded", "prompt=9, ctx=10"),
        oversized_files,
        failures,
    ) == (0, 0)

    assert oversized_files == ["channel/large.txt"]
    assert failures == [("channel/fail.txt", "prompt=9, ctx=10")]
    assert caplog.records == []


def test_final_summary_details_failures_but_counts_oversized_skips(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failures must be listed with file and reason; oversized skips stay an aggregate count."""
    module = load_summarize_transcripts_module()

    caplog.set_level(logging.INFO)

    def fake_process_single_file(
        txt_file: Path,
        output_file: Path,
        prompt_template: str,
        llm: object,
        model: str,
        encoder: object,
        effective_context_window: int,
        skip_threshold_pct: int,
    ) -> tuple[str, str | None]:
        if txt_file.name == "large.txt":
            return "skip (oversized)", "prompt=9 > limit=8"
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "process_single_file", fake_process_single_file)

    rc = module.process_pending(
        [
            (Path("channel/large.txt"), Path("out/large.md")),
            (Path("channel/fail.txt"), Path("out/fail.md")),
        ],
        "",
        None,
        _MODEL,
        None,
        10,
        80,
        1,
    )

    assert rc == 1

    messages = [record.message for record in caplog.records]

    # Aggregate counts and the oversized skip stay as counts.
    assert "Summary: pending=2, summarized=0, skipped=1, failed=1" in messages
    assert "Oversized skipped: 1 file(s); limit=80% of worker context" in messages

    # Failures are listed individually with the file path and the reason.
    assert "--- Failure Summary ---" in messages
    assert "❌ channel/fail.txt: boom" in messages

    # The oversized skip is summarized as a count, not relisted by filename in the end-of-run summary.
    summary_section = messages[messages.index("Summary: pending=2, summarized=0, skipped=1, failed=1") :]
    assert not any("large.txt" in message for message in summary_section)
