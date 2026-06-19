"""Tests for bounded dispatch in summarize-transcripts.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


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
