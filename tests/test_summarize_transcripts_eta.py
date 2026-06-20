"""Tests for the progress-rate ETA estimator in summarize-transcripts.py."""

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


def test_estimate_eta_uses_completed_fraction_and_elapsed_time() -> None:
    module = load_summarize_module()

    assert module.estimate_eta_seconds(completed=25, total=100, elapsed_seconds=3600.0) == 10800.0


def test_estimate_eta_returns_zero_when_complete() -> None:
    module = load_summarize_module()

    assert module.estimate_eta_seconds(completed=100, total=100, elapsed_seconds=3600.0) == 0.0


def test_estimate_eta_returns_none_until_progress_exists() -> None:
    module = load_summarize_module()

    assert module.estimate_eta_seconds(completed=0, total=100, elapsed_seconds=3600.0) is None
    assert module.estimate_eta_seconds(completed=25, total=100, elapsed_seconds=0.0) is None


def test_estimate_eta_rejects_invalid_counts() -> None:
    module = load_summarize_module()

    for completed, total, elapsed, message in [
        (0, 0, 1.0, "total must be positive"),
        (-1, 100, 1.0, "completed must not be negative"),
        (101, 100, 1.0, "completed must not exceed total"),
        (1, 100, -1.0, "elapsed_seconds must not be negative"),
    ]:
        try:
            module.estimate_eta_seconds(completed=completed, total=total, elapsed_seconds=elapsed)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected ValueError containing {message!r}")


def test_format_processing_eta_calculating_until_progress_exists() -> None:
    module = load_summarize_module()

    assert module.format_processing_eta(completed=0, total=10, elapsed_seconds=60.0) == "calculating"


def test_format_processing_eta_formats_remaining() -> None:
    module = load_summarize_module()

    assert module.format_processing_eta(completed=25, total=100, elapsed_seconds=3600.0) == "3h0m"
