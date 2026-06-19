"""Tests for summarize-transcripts LM Studio context-window resolution."""

from __future__ import annotations

import importlib.util
import json
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


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_get_loaded_context_length_reads_lm_studio_loaded_context() -> None:
    module = load_summarize_transcripts_module()

    def fake_urlopen(url: str, timeout: int) -> FakeResponse:
        assert url == "http://127.0.0.1:1234/api/v0/models"
        assert timeout == 10
        return FakeResponse(
            {
                "data": [
                    {
                        "id": "qwen/qwen3.6-35b-a3b",
                        "state": "loaded",
                        "max_context_length": 262144,
                        "loaded_context_length": 8618,
                    }
                ]
            }
        )

    module.urlopen = fake_urlopen

    assert module.get_loaded_context_length(make_llm()) == 8618


def test_auto_context_window_uses_loaded_context(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_summarize_transcripts_module()

    def loaded(_: LLMConfig) -> int:
        return 8618

    monkeypatch.setattr(module, "get_loaded_context_length", loaded)

    assert module.resolve_effective_context_window(make_llm(context_window="auto")) == 8618


def test_numeric_context_window_uses_larger_loaded_context(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_summarize_transcripts_module()

    def loaded(_: LLMConfig) -> int:
        return 8618

    monkeypatch.setattr(module, "get_loaded_context_length", loaded)

    assert module.resolve_effective_context_window(make_llm(context_window=4096)) == 8618


def test_numeric_context_window_rejects_loaded_context_that_is_smaller(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_summarize_transcripts_module()

    def loaded(_: LLMConfig) -> int:
        return 4096

    monkeypatch.setattr(module, "get_loaded_context_length", loaded)

    with pytest.raises(RuntimeError, match="exceeds loaded LM Studio context length"):
        module.resolve_effective_context_window(make_llm(context_window=8618))


def test_numeric_context_window_warns_and_uses_config_when_metadata_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = load_summarize_transcripts_module()

    def fail(_: LLMConfig) -> int:
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(module, "get_loaded_context_length", fail)

    assert module.resolve_effective_context_window(make_llm(context_window=4096)) == 4096
    assert "Could not verify loaded LM Studio context length" in caplog.text


def test_auto_context_window_rejects_missing_loaded_context(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_summarize_transcripts_module()

    def fail(_: LLMConfig) -> int:
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(module, "get_loaded_context_length", fail)

    with pytest.raises(RuntimeError, match="context_window is auto"):
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
    assert "context window 10" in note


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
    assert "prompt contains 9 tokens" in note


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


def test_process_single_file_does_not_retry_context_size_exceeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    status, note = module.process_single_file(
        txt_file,
        output_file,
        "{transcript}",
        make_llm(context_window=1000),
        FakeEncoder(),
        100,
        80,
    )

    assert status == "skip (context exceeded)"
    assert note == "Context size has been exceeded"
    assert calls == 1
    assert not output_file.exists()
