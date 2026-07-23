"""Unit tests for :func:`src.util.media_stem.resolve_media_stem`.

All fixtures are synthetic files under ``tmp_path``; no real data directory or
config is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.util.media_stem import resolve_media_stem

_VIDEO_ID = "dQw4w9WgXcQ"
# Title with an em dash, umlaut, and emoji, mirroring real channel filenames.
_TITLE = "Some Title — Ünïcode 🧲"
_STEM = f"{_TITLE} [{_VIDEO_ID}]"


def _touch(directory: Path, name: str) -> Path:
    """Create ``directory/name`` (with parents) as a small non-empty file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("x", encoding="utf-8")
    return path


def test_resolves_stem_from_single_dir(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _touch(audio, f"{_STEM}.wav")
    assert resolve_media_stem([audio], _VIDEO_ID) == _STEM


def test_returns_none_when_no_artifact(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    assert resolve_media_stem([audio], _VIDEO_ID) is None


def test_falls_back_to_archived_video_when_audio_absent(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()  # exists but holds no matching WAV (archived: WAV removed)
    archive = tmp_path / "archive_videos"
    _touch(archive, f"{_STEM}.mp4")
    assert resolve_media_stem([audio, archive], _VIDEO_ID) == _STEM


def test_falls_back_to_transcript_when_only_transcript_exists(tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts"
    _touch(transcripts, f"{_STEM}.srt")
    assert resolve_media_stem([transcripts], _VIDEO_ID) == _STEM


def test_first_directory_wins_over_later_ones(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _touch(audio, f"{_STEM}.wav")
    archive = tmp_path / "archive_videos"
    _touch(archive, f"A Different Title [{_VIDEO_ID}].mp4")
    assert resolve_media_stem([audio, archive], _VIDEO_ID) == _STEM


def test_strips_format_code_suffix(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _touch(audio, f"{_STEM}.f251-11.wav")
    assert resolve_media_stem([audio], _VIDEO_ID) == _STEM


def test_strips_double_extension(tmp_path: Path) -> None:
    # A leftover next to a video may carry a compound extension; the stem must
    # still be cut at the video-id marker.
    videos = tmp_path / "videos"
    _touch(videos, f"{_STEM}.mp4.part")
    assert resolve_media_stem([videos], _VIDEO_ID) == _STEM


def test_ignores_appledouble_sidecars(tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts"
    _touch(transcripts, f"._{_STEM}.srt")  # only a macOS sidecar exists
    assert resolve_media_stem([transcripts], _VIDEO_ID) is None


def test_skips_missing_directories(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    audio = tmp_path / "audio"
    _touch(audio, f"{_STEM}.wav")
    assert resolve_media_stem([missing, audio], _VIDEO_ID) == _STEM


def test_ignores_other_videos_in_same_dir(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    _touch(audio, "Unrelated Video [abcdefghijk].wav")
    assert resolve_media_stem([audio], _VIDEO_ID) is None


def test_invalid_video_id_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid YouTube video ID"):
        resolve_media_stem([tmp_path], "too-short")
