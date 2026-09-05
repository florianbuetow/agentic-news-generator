"""Tests for the audio classification map runner.

All tests run on synthetic WAV files under tmp_path with injected classifier
functions; no ONNX model and no real data directory is touched.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from src.audio_classification.runner import classify_waveform, run_audio_classification
from src.config import AudioClassificationConfig, Config


def build_tool_config() -> AudioClassificationConfig:
    """Build the threshold config used across the runner tests."""
    return AudioClassificationConfig(
        silence_floor_dbfs=-50.0,
        music_threshold=0.5,
        music_override_speech_max=0.2,
        speech_onset_probability=0.5,
        speech_offset_probability=0.35,
        min_segment_seconds=0.3,
    )


class _Config:
    """Duck-typed stand-in for Config, backed by tmp_path directories."""

    def __init__(self, data_dir: Path) -> None:
        self.audio_dir = data_dir / "audio"
        self.map_dir = data_dir / "audio-classification"
        self.audio_dir.mkdir(parents=True)
        self.map_dir.mkdir(parents=True)

    def get_audio_classification_config(self) -> AudioClassificationConfig:
        return build_tool_config()

    def get_data_downloads_audio_dir(self) -> Path:
        return self.audio_dir

    def get_data_downloads_audio_classification_dir(self) -> Path:
        return self.map_dir


def write_wav(path: Path, amplitude: float, seconds: float) -> None:
    """Write a 16 kHz mono PCM-16 WAV with a constant absolute amplitude."""
    n_samples = int(16000 * seconds)
    samples = np.full(n_samples, int(amplitude * 32767.0), dtype=np.int16)
    samples[::2] = -samples[::2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(samples.tobytes())


def all_speech_fn(wav: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Fake speech classifier: every chunk is speech."""
    return np.ones(len(wav) // 512, dtype=np.float32)


def no_music_fn(wav: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Fake music classifier: no music, full speech posterior."""
    return np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)


class TestClassifyWaveform:
    def test_loud_speech_covers_the_whole_duration(self) -> None:
        """A loud waveform with a firing speech detector is all speech."""
        wav = np.full(8000, 0.5, dtype=np.float32)
        wav[::2] = -0.5
        intervals = classify_waveform(wav, build_tool_config(), all_speech_fn, no_music_fn)
        assert intervals == {"speech": [(0.0, 0.5)], "music": [], "other": []}

    def test_silent_audio_is_other_despite_speech_detector(self) -> None:
        """Digital silence stays OTHER even when the speech detector fires."""
        wav = np.zeros(8000, dtype=np.float32)
        intervals = classify_waveform(wav, build_tool_config(), all_speech_fn, no_music_fn)
        assert intervals == {"speech": [], "music": [], "other": [(0.0, 0.5)]}

    def test_empty_waveform_gives_three_empty_classes(self) -> None:
        """Zero samples: all classes present and empty."""
        wav = np.zeros(0, dtype=np.float32)
        intervals = classify_waveform(wav, build_tool_config(), all_speech_fn, no_music_fn)
        assert intervals == {"speech": [], "music": [], "other": []}


class TestRunAudioClassification:
    def test_writes_one_map_per_wav_and_reports_success(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Every pending WAV gets a JSON map; exit code is 0."""
        config = _Config(tmp_path)
        write_wav(config.audio_dir / "channel" / "Video [aaaaaaaaaaa].wav", amplitude=0.5, seconds=0.5)
        exit_code = run_audio_classification(cast(Config, config), all_speech_fn, no_music_fn)
        assert exit_code == 0
        map_path = config.map_dir / "channel" / "Video [aaaaaaaaaaa].json"
        assert json.loads(map_path.read_text(encoding="utf-8")) == {
            "speech": [[0.0, 0.5]],
            "music": [],
            "other": [],
        }
        output = capsys.readouterr().out
        assert "Processing: channel/Video [aaaaaaaaaaa].wav" in output
        assert "done" in output

    def test_existing_map_is_skipped_with_one_line(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """An existing JSON map is neither recomputed nor overwritten."""
        config = _Config(tmp_path)
        write_wav(config.audio_dir / "channel" / "Video [aaaaaaaaaaa].wav", amplitude=0.5, seconds=0.5)
        map_path = config.map_dir / "channel" / "Video [aaaaaaaaaaa].json"
        map_path.parent.mkdir(parents=True)
        map_path.write_text("marker", encoding="utf-8")
        exit_code = run_audio_classification(cast(Config, config), all_speech_fn, no_music_fn)
        assert exit_code == 0
        assert map_path.read_text(encoding="utf-8") == "marker"
        output = capsys.readouterr().out
        assert output.count("Skipping: channel/Video [aaaaaaaaaaa].wav") == 1
        assert "Processing:" not in output

    def test_channels_with_fewest_pending_files_go_first(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A channel with one pending file is processed before one with two."""
        config = _Config(tmp_path)
        write_wav(config.audio_dir / "alpha" / "One [aaaaaaaaaaa].wav", amplitude=0.5, seconds=0.2)
        write_wav(config.audio_dir / "alpha" / "Two [bbbbbbbbbbb].wav", amplitude=0.5, seconds=0.2)
        write_wav(config.audio_dir / "beta" / "Three [ccccccccccc].wav", amplitude=0.5, seconds=0.2)
        exit_code = run_audio_classification(cast(Config, config), all_speech_fn, no_music_fn)
        assert exit_code == 0
        output = capsys.readouterr().out
        assert output.index("Processing: beta/") < output.index("Processing: alpha/")

    def test_one_broken_file_does_not_stop_the_batch(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A corrupt WAV is reported in the failure summary; the rest completes."""
        config = _Config(tmp_path)
        broken = config.audio_dir / "channel" / "Broken [bbbbbbbbbbb].wav"
        broken.parent.mkdir(parents=True)
        broken.write_bytes(b"not a wav file")
        write_wav(config.audio_dir / "channel" / "Good [aaaaaaaaaaa].wav", amplitude=0.5, seconds=0.5)
        exit_code = run_audio_classification(cast(Config, config), all_speech_fn, no_music_fn)
        assert exit_code == 1
        assert (config.map_dir / "channel" / "Good [aaaaaaaaaaa].json").is_file()
        assert not (config.map_dir / "channel" / "Broken [bbbbbbbbbbb].json").exists()
        output = capsys.readouterr().out
        assert "--- Failure Summary ---" in output
        assert "Broken [bbbbbbbbbbb].wav" in output

    def test_no_audio_files_reports_nothing_to_do(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """An empty audio directory succeeds with a clear summary."""
        config = _Config(tmp_path)
        exit_code = run_audio_classification(cast(Config, config), all_speech_fn, no_music_fn)
        assert exit_code == 0
        assert "0 processed" in capsys.readouterr().out
