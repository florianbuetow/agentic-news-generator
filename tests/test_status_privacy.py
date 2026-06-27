"""Tests for the privacy-mode channel label helper in scripts/status.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load_status_module() -> Any:
    repo_root = Path(__file__).resolve().parent.parent
    status_path = repo_root / "scripts" / "status.py"
    spec = importlib.util.spec_from_file_location("status_script", status_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/status.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


status = _load_status_module()


def test_display_label_returns_channel_name_when_privacy_disabled() -> None:
    assert status._channel_display_label("Nate B Jones", None) == "Nate B Jones"


def test_display_label_returns_uppercase_category_in_privacy_mode() -> None:
    mapping = {"Nate B Jones": "ai-news", "Anthropic": "ai-engineering_research"}
    assert status._channel_display_label("Nate B Jones", mapping) == "AI-NEWS"
    assert status._channel_display_label("Anthropic", mapping) == "AI-ENGINEERING_RESEARCH"


def test_display_label_uses_unknown_when_channel_missing_from_mapping() -> None:
    assert status._channel_display_label("Untracked Channel", {}) == "UNKNOWN"
