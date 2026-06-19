"""Tests for the pending transcript token histogram script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_histogram_module() -> ModuleType:
    """Load the histogram script as a module."""
    script_path = Path(__file__).parent.parent / "scripts" / "summarize-transcripts-token-histogram.py"
    spec = importlib.util.spec_from_file_location("summarize_transcripts_token_histogram", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bucket_index_boundaries() -> None:
    """Bucket assignment keeps the requested 8K-wide ranges."""
    module = load_histogram_module()

    assert module.bucket_index(0, 256_000, 8_000) == 0
    assert module.bucket_index(7_999, 256_000, 8_000) == 0
    assert module.bucket_index(8_000, 256_000, 8_000) == 1
    assert module.bucket_index(255_999, 256_000, 8_000) == 31
    assert module.bucket_index(256_000, 256_000, 8_000) == 31
    assert module.bucket_index(256_001, 256_000, 8_000) is None


def test_build_histogram_counts_overflow() -> None:
    """Histogram uses 32 buckets and tracks values beyond the rendered x-axis."""
    module = load_histogram_module()

    result = module.build_histogram([0, 7_999, 8_000, 255_999, 256_000, 256_001], 256_000, 8_000)

    assert len(result.bucket_counts) == 32
    assert result.bucket_counts[0] == 2
    assert result.bucket_counts[1] == 1
    assert result.bucket_counts[31] == 2
    assert result.overflow_count == 1
    assert result.total_files == 6
    assert result.max_observed_tokens == 256_001


def test_choose_bucket_width_uses_smallest_power_of_two_width() -> None:
    """Adaptive buckets use the smallest power-of-two token width that covers the largest file."""
    module = load_histogram_module()

    assert module.choose_bucket_width(0, 32) == 1_024
    assert module.choose_bucket_width(32_768, 32) == 1_024
    assert module.choose_bucket_width(32_769, 32) == 2_048
    assert module.choose_bucket_width(65_537, 32) == 4_096


def test_resolve_histogram_range_defaults_to_32_adaptive_buckets() -> None:
    """Default range resolution covers the largest observed file with 32 buckets."""
    module = load_histogram_module()

    bucket_width, max_tokens = module.resolve_histogram_range(19_363, 32, None, None)

    assert bucket_width == 1_024
    assert max_tokens == 32_768


def test_render_histogram_includes_axes_and_counts() -> None:
    """Rendered output includes percent labels, count labels, token labels, and blocks."""
    module = load_histogram_module()
    result = module.build_histogram([0, 8_000, 16_000, 16_001], 256_000, 8_000)

    rendered = module.render_histogram(result, 256_000, 8_000, 20, 2)

    assert "100%" in rendered
    assert "75%" in rendered
    assert "50%" in rendered
    assert "25%" in rendered
    assert "0%" in rendered
    assert "0K" in rendered
    assert "16K" in rendered
    assert "256K" in rendered
    assert "files" in rendered
    assert "█" in rendered


def test_render_histogram_uses_half_block_for_top_partial_row() -> None:
    """A bucket that reaches less than half of its current row uses a half-height top block."""
    module = load_histogram_module()
    result = module.build_histogram([0, 0, 0, 0, 1, 2, 3, 4, 5], 10, 1)

    rendered = module.render_histogram(result, 10, 1, 10, 1)

    assert "▄" in rendered
