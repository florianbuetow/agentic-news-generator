"""Tests for per-instance loaded/loading state rendering in scripts/lmstudio_status.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


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


def test_matching_instances_surfaces_colon_suffixed_duplicates() -> None:
    states = {
        "qwen/qwen3.6-35b-a3b": "loaded",
        "qwen/qwen3.6-35b-a3b:2": "loading",
        "google/gemma-4-e2b-it": "not-loaded",
    }
    # Configured with the openai/ prefix; both the base and the ':2' duplicate must match.
    assert lmstudio_status._matching_instances("openai/qwen/qwen3.6-35b-a3b", states) == [
        "qwen/qwen3.6-35b-a3b",
        "qwen/qwen3.6-35b-a3b:2",
    ]


def test_matching_instances_empty_when_model_not_downloaded() -> None:
    states = {"google/gemma-4-e2b-it": "not-loaded"}
    assert lmstudio_status._matching_instances("openai/qwen/qwen3.6-35b-a3b", states) == []


def test_check_base_emits_one_row_per_instance_with_real_state(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = lmstudio_status.ModelIndex(
        states={
            "qwen/qwen3.6-35b-a3b": "loaded",
            "qwen/qwen3.6-35b-a3b:2": "loading",
        }
    )

    def fake_index(_api_base: str) -> Any:
        return fake

    monkeypatch.setattr(lmstudio_status, "get_model_index", fake_index)
    entry = lmstudio_status.RequiredModel(
        section="summarize_transcripts.llm",
        model="openai/qwen/qwen3.6-35b-a3b",
        api_base="http://127.0.0.1:1234/v1",
    )

    ok, rows = lmstudio_status._check_base("http://127.0.0.1:1234/v1", [entry])

    assert ok is True
    # The duplicate ':2' instance is its own row, and a loading instance is not
    # silently reported as loaded.
    assert rows == [
        ("summarize_transcripts.llm", "openai/qwen/qwen3.6-35b-a3b", "qwen/qwen3.6-35b-a3b", True, "loaded"),
        ("summarize_transcripts.llm", "openai/qwen/qwen3.6-35b-a3b", "qwen/qwen3.6-35b-a3b:2", True, "loading"),
    ]


def test_check_base_marks_missing_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = lmstudio_status.ModelIndex(states={"something-else": "loaded"})

    def fake_index(_api_base: str) -> Any:
        return fake

    monkeypatch.setattr(lmstudio_status, "get_model_index", fake_index)
    entry = lmstudio_status.RequiredModel(
        section="summarize_transcripts.llm",
        model="openai/qwen/qwen3.6-35b-a3b",
        api_base="http://127.0.0.1:1234/v1",
    )

    ok, rows = lmstudio_status._check_base("http://127.0.0.1:1234/v1", [entry])

    assert ok is False
    assert rows == [("summarize_transcripts.llm", "openai/qwen/qwen3.6-35b-a3b", "-", False, None)]


def test_state_cell_distinguishes_loaded_loading_and_missing() -> None:
    assert "loaded" in lmstudio_status._state_cell("loaded")
    assert lmstudio_status._GREEN in lmstudio_status._state_cell("loaded")
    assert "loading" in lmstudio_status._state_cell("loading")
    assert lmstudio_status._YELLOW in lmstudio_status._state_cell("loading")
    assert "not-loaded" in lmstudio_status._state_cell("not-loaded")
    assert lmstudio_status._state_cell(None) == f"{lmstudio_status._RED}-{lmstudio_status._RESET}"
