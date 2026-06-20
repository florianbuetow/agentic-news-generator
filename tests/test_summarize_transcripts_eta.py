"""Tests for the stall-robust ETA estimator in summarize-transcripts.py."""

from __future__ import annotations

import importlib.util
import itertools
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


def timestamps_from_intervals(intervals: list[float]) -> list[float]:
    """Build cumulative monotonic timestamps that reproduce the given intervals."""
    return [0.0, *itertools.accumulate(intervals)]


def test_estimate_eta_rejects_stall_outlier() -> None:
    """A single huge interval (a stall) must not inflate the estimate.

    Intervals are thirty normal 45s files plus one 3000s stall. The median is 45,
    so ETA == 45 * remaining. The old lifetime-mean formula would use
    mean == 4350/31 ≈ 140.3, a ~3.1x inflation — that was the bug.
    """
    module = load_summarize_module()
    times = timestamps_from_intervals([45.0] * 30 + [3000.0])

    assert module.estimate_eta_seconds(times, 100) == 45.0 * 100


def test_estimate_eta_median_ignores_lone_spike() -> None:
    module = load_summarize_module()
    times = timestamps_from_intervals([10.0, 10.0, 10.0, 10.0, 10.0, 9999.0])

    assert module.estimate_eta_seconds(times, 7) == 10.0 * 7


def test_estimate_eta_returns_none_with_too_few_samples() -> None:
    module = load_summarize_module()

    assert module.estimate_eta_seconds([1.0, 2.0, 3.0], 10) is None
    assert module.estimate_eta_seconds([], 10) is None


def test_estimate_eta_with_minimum_samples() -> None:
    module = load_summarize_module()
    times = timestamps_from_intervals([20.0, 20.0, 20.0])

    assert module.estimate_eta_seconds(times, 5) == 20.0 * 5


def test_format_processing_eta_calculating_until_enough_samples() -> None:
    module = load_summarize_module()

    assert module.format_processing_eta([1.0, 2.0], 10) == "calculating"


def test_format_processing_eta_formats_remaining() -> None:
    module = load_summarize_module()
    # Median interval 60s, 130 remaining -> 7800s -> 2h10m.
    times = timestamps_from_intervals([60.0, 60.0, 60.0])

    assert module.format_processing_eta(times, 130) == "2h10m"
