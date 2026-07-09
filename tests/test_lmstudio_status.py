"""Tests for exact per-model loaded-state rendering in scripts/lmstudio_status.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from src.llm.lm_studio import ModelRecord


def _load_lmstudio_status_module() -> Any:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "lmstudio_status.py"
    spec = importlib.util.spec_from_file_location("lmstudio_status_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/lmstudio_status.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lmstudio_status = _load_lmstudio_status_module()


def _registry(states: dict[str, str]) -> Any:
    """Build a registry whose loaded records report a context length."""
    return lmstudio_status.LMStudioRegistry(
        api_base="http://127.0.0.1:1234/v1",
        records={
            model_id: ModelRecord(
                lm_studio_id=model_id,
                state=state,
                loaded_context_length=32768 if state == "loaded" else None,
                max_context_length=262144,
            )
            for model_id, state in states.items()
        },
    )


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, registry: Any) -> None:
    def fake_fetch(_cls: Any, _api_base: str | None) -> Any:
        return registry

    monkeypatch.setattr(lmstudio_status.LMStudioRegistry, "fetch", classmethod(fake_fetch))


def test_check_base_reports_each_configured_model_with_its_own_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact matching: the not-loaded base id and the loaded ':2' instance are distinct rows."""
    _patch_fetch(monkeypatch, _registry({"qwen/qwen3.6-35b-a3b": "not-loaded", "qwen/qwen3.6-35b-a3b:2": "loaded"}))

    entry = lmstudio_status.RequiredModels(
        section="summarize_transcripts.llm",
        models=["openai/qwen/qwen3.6-35b-a3b", "openai/qwen/qwen3.6-35b-a3b:2"],
        api_base="http://127.0.0.1:1234/v1",
    )

    ok, rows = lmstudio_status._check_base("http://127.0.0.1:1234/v1", [entry])

    assert ok is True
    assert rows == [
        ("summarize_transcripts.llm", "openai/qwen/qwen3.6-35b-a3b", "qwen/qwen3.6-35b-a3b", False, "not-loaded"),
        ("summarize_transcripts.llm", "openai/qwen/qwen3.6-35b-a3b:2", "qwen/qwen3.6-35b-a3b:2", True, "loaded"),
    ]


def test_check_base_not_ok_when_no_configured_model_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A section with nothing loaded is not OK, but does not raise."""
    _patch_fetch(monkeypatch, _registry({"qwen/qwen3.6-35b-a3b": "not-loaded"}))

    entry = lmstudio_status.RequiredModels(
        section="summarize_transcripts.llm",
        models=["openai/qwen/qwen3.6-35b-a3b"],
        api_base="http://127.0.0.1:1234/v1",
    )

    ok, rows = lmstudio_status._check_base("http://127.0.0.1:1234/v1", [entry])

    assert ok is False
    assert rows == [("summarize_transcripts.llm", "openai/qwen/qwen3.6-35b-a3b", "qwen/qwen3.6-35b-a3b", False, "not-loaded")]


def test_check_base_marks_undownloaded_model_with_no_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model absent from the server has no instance and no state."""
    _patch_fetch(monkeypatch, _registry({"something-else": "loaded"}))

    entry = lmstudio_status.RequiredModels(
        section="summarize_transcripts.llm",
        models=["openai/qwen/qwen3.6-35b-a3b"],
        api_base="http://127.0.0.1:1234/v1",
    )

    ok, rows = lmstudio_status._check_base("http://127.0.0.1:1234/v1", [entry])

    assert ok is False
    assert rows == [("summarize_transcripts.llm", "openai/qwen/qwen3.6-35b-a3b", "-", False, None)]


def test_check_base_reports_unreachable_server_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable server is a FAIL row, never an exception."""

    def raising_fetch(_cls: Any, _api_base: str) -> Any:
        raise RuntimeError("LM Studio metadata endpoint is unreachable")

    monkeypatch.setattr(lmstudio_status.LMStudioRegistry, "fetch", classmethod(raising_fetch))

    entry = lmstudio_status.RequiredModels(
        section="summarize_transcripts.llm",
        models=["openai/qwen/qwen3.6-35b-a3b"],
        api_base="http://127.0.0.1:1234/v1",
    )

    ok, rows = lmstudio_status._check_base("http://127.0.0.1:1234/v1", [entry])

    assert ok is False
    assert rows == []


def test_collect_required_models_reads_models_lists() -> None:
    """All four LLM sections yield their ordered models list."""
    config = {
        "summarize_transcripts": {"llm": {"models": ["a"], "api_base": "http://127.0.0.1:1234/v1"}},
        "url_clean_content": {"llm": {"models": ["b"], "api_base": "http://127.0.0.1:1234/v1"}},
        "agentic_unit_test_reviews": {"llm": {"models": ["c"], "base_url": "http://localhost:1234/v1"}},
        "agentic_shell_script_reviews": {"llm": {"models": ["d"], "base_url": "http://localhost:1234/v1"}},
    }

    required = lmstudio_status.collect_required_models(config)

    assert [entry.models for entry in required] == [["a"], ["b"], ["c"], ["d"]]
    assert [entry.api_base for entry in required] == [
        "http://127.0.0.1:1234/v1",
        "http://127.0.0.1:1234/v1",
        "http://localhost:1234/v1",
        "http://localhost:1234/v1",
    ]


def test_main_exits_zero_when_nothing_is_loaded(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """just status must always exit 0, even with no model loaded."""
    _patch_fetch(monkeypatch, _registry({"something-else": "not-loaded"}))

    def fake_safe_load(_stream: Any) -> dict[str, Any]:
        return {}

    def fake_collect(_config: dict[str, Any]) -> list[Any]:
        return [
            lmstudio_status.RequiredModels(
                section="summarize_transcripts.llm",
                models=["openai/qwen/qwen3.6-35b-a3b"],
                api_base="http://127.0.0.1:1234/v1",
            )
        ]

    monkeypatch.setattr(lmstudio_status, "CONFIG_PATH", Path(__file__))
    monkeypatch.setattr(lmstudio_status.yaml, "safe_load", fake_safe_load)
    monkeypatch.setattr(lmstudio_status, "collect_required_models", fake_collect)

    assert lmstudio_status.main() == 0
    assert "Some checks failed" in capsys.readouterr().out


def test_state_cell_distinguishes_loaded_loading_and_missing() -> None:
    assert "loaded" in lmstudio_status._state_cell("loaded")
    assert lmstudio_status._GREEN in lmstudio_status._state_cell("loaded")
    assert "loading" in lmstudio_status._state_cell("loading")
    assert lmstudio_status._YELLOW in lmstudio_status._state_cell("loading")
    assert "not-loaded" in lmstudio_status._state_cell("not-loaded")
    assert lmstudio_status._state_cell(None) == f"{lmstudio_status._RED}-{lmstudio_status._RESET}"
