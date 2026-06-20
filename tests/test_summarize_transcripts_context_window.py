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


def make_llm(*, context_window: int | Literal["auto"] = "auto") -> LLMConfig:
    return LLMConfig(
        model="openai/qwen/qwen3.6-35b-a3b",
        api_base="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        context_window=context_window,
        max_tokens=32768,
        temperature=0.3,
        context_window_threshold=90,
        max_retries=1,
        retry_delay=1.0,
    )


def test_auto_context_window_uses_loaded_context(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_summarize_transcripts_module()

    def loaded(api_base: str | None, model: str) -> int:
        assert api_base == "http://127.0.0.1:1234/v1"
        assert model == "openai/qwen/qwen3.6-35b-a3b"
        return 8618

    monkeypatch.setattr(module, "get_loaded_context_length", loaded)

    assert module.resolve_effective_context_window(make_llm(context_window="auto")) == 8618


def test_numeric_context_window_uses_larger_loaded_context(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_summarize_transcripts_module()

    def loaded(api_base: str | None, model: str) -> int:
        return 8618

    monkeypatch.setattr(module, "get_loaded_context_length", loaded)

    assert module.resolve_effective_context_window(make_llm(context_window=4096)) == 8618


def test_numeric_context_window_rejects_loaded_context_that_is_smaller(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_summarize_transcripts_module()

    def loaded(api_base: str | None, model: str) -> int:
        return 4096

    monkeypatch.setattr(module, "get_loaded_context_length", loaded)

    with pytest.raises(RuntimeError, match="exceeds loaded LM Studio context length"):
        module.resolve_effective_context_window(make_llm(context_window=8618))


def test_numeric_context_window_warns_and_uses_config_when_metadata_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = load_summarize_transcripts_module()

    def fail(api_base: str | None, model: str) -> int:
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(module, "get_loaded_context_length", fail)

    assert module.resolve_effective_context_window(make_llm(context_window=4096)) == 4096
    assert "LM Studio ctx unavailable; using config ctx=4,096" in caplog.text


def test_auto_context_window_rejects_missing_loaded_context(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_summarize_transcripts_module()

    def fail(api_base: str | None, model: str) -> int:
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(module, "get_loaded_context_length", fail)

    with pytest.raises(RuntimeError, match="context_window=auto"):
        module.resolve_effective_context_window(make_llm(context_window="auto"))


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

    def complete(prompt: str, llm: LLMConfig, max_tokens: int) -> str:
        requested_max_tokens.append(max_tokens)
        return "summary"

    monkeypatch.setattr(module, "call_llm", complete)

    status, note = module.process_single_file(
        txt_file,
        output_file,
        "prompt wrapper {transcript}",
        make_llm(context_window=1000),
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

    def fail_with_context_error(prompt: str, llm: LLMConfig, max_tokens: int) -> str:
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
        module.time.monotonic() - 60,
        1,
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
        module.time.monotonic(),
        0,
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


def test_final_summary_counts_do_not_repeat_filenames(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_summarize_transcripts_module()

    caplog.set_level(logging.INFO)

    def fake_process_single_file(
        txt_file: Path,
        output_file: Path,
        prompt_template: str,
        llm: object,
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
        None,
        10,
        80,
        1,
    )

    assert rc == 1

    summary_messages = [
        record.message for record in caplog.records if record.message.startswith(("Summary:", "Oversized skipped:", "Failures:"))
    ]
    assert summary_messages == [
        "Summary: pending=2, summarized=0, skipped=1, failed=1",
        "Oversized skipped: 1 file(s); limit=80% of worker context",
        "Failures: 1 file(s)",
    ]
