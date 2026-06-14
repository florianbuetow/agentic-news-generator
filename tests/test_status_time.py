"""Tests for transcript-time helpers in scripts/status.py."""

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


def test_timestamp_to_seconds_parses_supported_formats() -> None:
    assert status._timestamp_to_seconds("00:00:59,999") == 59
    assert status._timestamp_to_seconds("01:02:03.500") == 3723
    assert status._timestamp_to_seconds("999:59:59,000") == 3599999


def test_timestamp_to_seconds_rejects_invalid_values() -> None:
    assert status._timestamp_to_seconds("00:60:00,000") is None
    assert status._timestamp_to_seconds("invalid") is None


def test_format_seconds_as_dhm() -> None:
    assert status._format_seconds_as_dhm(0) == "0d0h0m"
    assert status._format_seconds_as_dhm(90061) == "1d1h1m"
    assert status._format_seconds_as_dhm(172799) == "1d23h59m"


def test_extract_last_timestamp_seconds_from_srt(tmp_path: Path) -> None:
    srt_file = tmp_path / "sample.srt"
    srt_file.write_text(
        """1\n00:00:05,000 --> 00:00:07,250\nhello\n\n2\n00:00:10,000 --> 00:00:15,999\nworld\n""",
        encoding="utf-8",
    )

    assert status._extract_last_timestamp_seconds_from_srt(srt_file) == 15


def test_completion_denominator_adds_raw_queue_to_transcripts() -> None:
    # Transcribed items plus the larger of the two raw-queue stages (videos vs audio).
    assert status._completion_denominator(97, 17, 16) == 97 + 17
    # Audio can exceed videos when a video was already removed after extraction.
    assert status._completion_denominator(0, 2, 5) == 5
    # No raw queue on disk: denominator is just the transcript count.
    assert status._completion_denominator(708, 0, 0) == 708


def test_sum_channel_cleaned_srt_seconds_skips_missing_timestamps(tmp_path: Path) -> None:
    channel_dir = tmp_path / "channel"
    channel_dir.mkdir(parents=True)

    (channel_dir / "a.srt").write_text(
        """1\n00:00:00,000 --> 00:01:00,000\na\n""",
        encoding="utf-8",
    )
    (channel_dir / "b.srt").write_text(
        """1\n00:00:10,000 --> 00:02:00,000\nb\n""",
        encoding="utf-8",
    )
    (channel_dir / "bad.srt").write_text("no timestamps here\n", encoding="utf-8")

    assert status._sum_channel_cleaned_srt_seconds(channel_dir) == (60 + 120)
