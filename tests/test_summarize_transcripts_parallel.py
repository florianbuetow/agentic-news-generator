"""Tests for bounded dispatch in summarize-transcripts.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import tiktoken

from src.config import LLMConfig


def load_summarize_module() -> Any:
    """Load the hyphenated summarize-transcripts.py as an importable module."""
    script_path = Path(__file__).parent.parent / "scripts" / "summarize-transcripts.py"
    spec = importlib.util.spec_from_file_location("summarize_transcripts", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pending_files(tmp_path: Path, count: int) -> list[tuple[Path, Path]]:
    return [(tmp_path / "channel" / f"{idx}.txt", tmp_path / "out" / f"{idx}.md") for idx in range(count)]


def test_process_pending_runs_with_parallelism_one(tmp_path: Path, monkeypatch: Any) -> None:
    module = load_summarize_module()
    processed: list[str] = []

    def fake_process_single_file(
        txt_file: Path,
        output_file: Path,
        prompt_template: str,
        llm: object,
        encoder: object,
        effective_context_window: int,
        skip_threshold_pct: int,
    ) -> tuple[str, None]:
        assert effective_context_window == 8192
        assert skip_threshold_pct == 80
        processed.append(txt_file.name)
        return "ok", None

    monkeypatch.setattr(module, "process_single_file", fake_process_single_file)

    rc = module.process_pending(pending_files(tmp_path, 3), "", None, None, 8192, 80, 1)

    assert rc == 0
    assert processed == ["0.txt", "1.txt", "2.txt"]


def test_process_pending_parallelism_one_collects_failures(tmp_path: Path, monkeypatch: Any) -> None:
    module = load_summarize_module()
    processed: list[str] = []

    def fake_process_single_file(
        txt_file: Path,
        output_file: Path,
        prompt_template: str,
        llm: object,
        encoder: object,
        effective_context_window: int,
        skip_threshold_pct: int,
    ) -> tuple[str, None]:
        processed.append(txt_file.name)
        if txt_file.name == "1.txt":
            raise RuntimeError("boom")
        return "ok", None

    monkeypatch.setattr(module, "process_single_file", fake_process_single_file)

    rc = module.process_pending(pending_files(tmp_path, 3), "", None, None, 8192, 80, 1)

    assert rc == 1
    assert processed == ["0.txt", "1.txt", "2.txt"]


def _make_llm_config() -> LLMConfig:
    return LLMConfig(
        model="test/model",
        api_base=None,
        api_key="test",
        context_window=8192,
        max_tokens=512,
        temperature=0.0,
        context_window_threshold=80,
        max_retries=1,
        retry_delay=0.01,
    )


def _fake_litellm_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


_THINK_ONLY_RESPONSE = "<think>internal reasoning only, no answer</think>"


def _think_only_completion(**_: object) -> MagicMock:
    return _fake_litellm_response(_THINK_ONLY_RESPONSE)


def test_call_llm_raises_when_think_tags_produce_empty_response(monkeypatch: Any) -> None:
    """call_llm must raise ValueError when the LLM returns only a <think> block."""
    module = load_summarize_module()
    monkeypatch.setattr(module.litellm, "completion", _think_only_completion)

    with pytest.raises(ValueError, match="empty"):
        module.call_llm("test prompt", _make_llm_config(), 512)


def test_process_single_file_does_not_write_empty_file_when_llm_returns_only_think_tags(tmp_path: Path, monkeypatch: Any) -> None:
    """process_single_file must not create a 0-byte summary when the LLM returns only think tags."""
    module = load_summarize_module()

    txt_file = tmp_path / "channel" / "video.txt"
    txt_file.parent.mkdir(parents=True)
    txt_file.write_text("Full transcript text here.", encoding="utf-8")
    output_file = tmp_path / "out" / "video.md"

    monkeypatch.setattr(module.litellm, "completion", _think_only_completion)

    encoder = tiktoken.get_encoding("cl100k_base")

    with pytest.raises(ValueError, match="empty"):
        module.process_single_file(txt_file, output_file, "{transcript}", _make_llm_config(), encoder, 8192, 80)

    assert not output_file.exists()
