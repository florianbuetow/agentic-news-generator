"""Tests for the ONNX plumbing of the audio classification map.

All tests use fake in-memory sessions; no model file is loaded and no network
access happens.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from src.audio_classification.audio_io import read_wav_mono_16k
from src.audio_classification.inference import (
    SILERO_CHUNK_SAMPLES,
    SILERO_CONTEXT_SAMPLES,
    YAMNET_MODEL_FILENAME,
    YAMNET_PASS_OFFSET_SAMPLES,
    ensure_models,
    silero_speech_probabilities,
    yamnet_music_and_speech,
)


def assert_close(actual: npt.NDArray[np.float32], expected: npt.NDArray[np.float32] | list[float], tolerance: float = 1e-6) -> None:
    """Assert two float sequences match element-wise within tolerance."""
    actual_values = cast(list[float], np.asarray(actual, dtype=np.float64).tolist())
    expected_values = cast(list[float], np.asarray(expected, dtype=np.float64).tolist())
    assert len(actual_values) == len(expected_values)
    for actual_value, expected_value in zip(actual_values, expected_values, strict=True):
        assert abs(actual_value - expected_value) < tolerance


class FakeSileroSession:
    """Stateful fake: the state carries the last sample of the previous chunk.

    Requires the official Silero input layout: 64 context samples followed by
    the 512-sample chunk.
    """

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[npt.NDArray[np.float32]]:
        chunk = input_feed["input"]
        assert chunk.shape[1] == SILERO_CONTEXT_SAMPLES + SILERO_CHUNK_SAMPLES
        state = input_feed["state"]
        prob = 0.5 * chunk.mean(axis=1) + 0.5 * state[0, :, 0]
        new_state = np.zeros_like(state)
        new_state[:, :, :] = chunk[:, -1][np.newaxis, :, np.newaxis]
        return [prob.astype(np.float32).reshape(-1, 1), new_state.astype(np.float32)]


def sequential_silero_reference(wav: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Chunk-by-chunk reference implementation of the fake session semantics."""
    n_chunks = len(wav) // SILERO_CHUNK_SAMPLES
    probs = np.zeros(n_chunks, dtype=np.float32)
    state = 0.0
    context = np.zeros(SILERO_CONTEXT_SAMPLES, dtype=np.float32)
    for i in range(n_chunks):
        chunk = wav[i * SILERO_CHUNK_SAMPLES : (i + 1) * SILERO_CHUNK_SAMPLES]
        model_input = np.concatenate([context, chunk])
        probs[i] = 0.5 * model_input.mean() + 0.5 * state
        state = float(model_input[-1])
        context = chunk[-SILERO_CONTEXT_SAMPLES:]
    return probs


class TestSileroSpeechProbabilities:
    def test_batched_run_matches_the_sequential_reference(self) -> None:
        """Warm lead-in keeps stream seams identical to a sequential run."""
        wav = (np.arange(40 * SILERO_CHUNK_SAMPLES, dtype=np.float32) % 997.0) / 997.0
        probs = silero_speech_probabilities(FakeSileroSession(), wav, batch_size=4, warmup_chunks=2)
        assert_close(probs, sequential_silero_reference(wav))

    def test_returns_one_probability_per_full_chunk(self) -> None:
        """A trailing partial chunk produces no probability."""
        wav = np.zeros(5 * SILERO_CHUNK_SAMPLES + 100, dtype=np.float32)
        probs = silero_speech_probabilities(FakeSileroSession(), wav, batch_size=4, warmup_chunks=2)
        assert len(probs) == 5

    def test_fewer_chunks_than_batch_size_still_works(self) -> None:
        """Three chunks with batch size 16 match the sequential reference."""
        wav = np.linspace(-1.0, 1.0, 3 * SILERO_CHUNK_SAMPLES, dtype=np.float32)
        probs = silero_speech_probabilities(FakeSileroSession(), wav, batch_size=16, warmup_chunks=8)
        assert_close(probs, sequential_silero_reference(wav))

    def test_empty_waveform_gives_empty_probabilities(self) -> None:
        """No full chunk: empty result, no session call."""
        wav = np.zeros(SILERO_CHUNK_SAMPLES - 1, dtype=np.float32)
        probs = silero_speech_probabilities(FakeSileroSession(), wav, batch_size=4, warmup_chunks=2)
        assert len(probs) == 0

    def test_warmup_chunks_must_be_positive(self) -> None:
        """A warm lead-in is required for correct stream seams."""
        wav = np.zeros(4 * SILERO_CHUNK_SAMPLES, dtype=np.float32)
        with pytest.raises(ValueError):
            silero_speech_probabilities(FakeSileroSession(), wav, batch_size=4, warmup_chunks=0)


YAMNET_FAKE_WINDOW_SAMPLES = 15600
YAMNET_FAKE_HOP_SAMPLES = 7680


class FakeYamnetSession:
    """Fake that encodes the input offset and frame index into the scores."""

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[npt.NDArray[np.float32]]:
        waveform = input_feed["waveform"]
        n_frames = max(0, 1 + (len(waveform) - YAMNET_FAKE_WINDOW_SAMPLES) // YAMNET_FAKE_HOP_SAMPLES)
        scores = np.zeros((n_frames, 521), dtype=np.float32)
        for frame in range(n_frames):
            scores[frame, 132] = float(waveform[0]) + float(frame)
            scores[frame, 0] = 1000.0 + float(waveform[0]) + float(frame)
        embeddings = np.zeros((n_frames, 1024), dtype=np.float32)
        log_mel = np.zeros((0, 64), dtype=np.float32)
        return [scores, embeddings, log_mel]


class TestYamnetMusicAndSpeech:
    def test_passes_interleave_into_a_fine_grid(self) -> None:
        """Grid index 5*i+k holds frame i of pass k."""
        wav = np.arange(4 * YAMNET_FAKE_WINDOW_SAMPLES, dtype=np.float32)
        music, _speech = yamnet_music_and_speech(FakeYamnetSession(), wav, passes=5)
        frames_per_pass = len(music) // 5
        assert frames_per_pass >= 2
        for k in range(5):
            for i in range(frames_per_pass):
                assert abs(float(music[5 * i + k]) - (k * YAMNET_PASS_OFFSET_SAMPLES + i)) < 1e-6

    def test_speech_posterior_comes_from_the_speech_classes(self) -> None:
        """The speech track carries the encoded speech-class values."""
        wav = np.arange(3 * YAMNET_FAKE_WINDOW_SAMPLES, dtype=np.float32)
        music, speech = yamnet_music_and_speech(FakeYamnetSession(), wav, passes=5)
        assert_close(speech, music + 1000.0)

    def test_all_passes_are_trimmed_to_the_shortest_pass(self) -> None:
        """The interleaved length is a multiple of the pass count."""
        wav = np.arange(2 * YAMNET_FAKE_WINDOW_SAMPLES + 3000, dtype=np.float32)
        music, _ = yamnet_music_and_speech(FakeYamnetSession(), wav, passes=5)
        assert len(music) % 5 == 0
        assert len(music) > 0

    def test_audio_shorter_than_one_window_gives_a_silent_pseudo_frame(self) -> None:
        """Very short audio yields one zero frame so pooling stays defined."""
        wav = np.ones(1000, dtype=np.float32)
        music, speech = yamnet_music_and_speech(FakeYamnetSession(), wav, passes=5)
        assert music.tolist() == [0.0]
        assert speech.tolist() == [0.0]


class TestEnsureModels:
    def test_existing_model_files_are_not_downloaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-empty model files skip the download."""
        (tmp_path / "silero_vad.onnx").write_bytes(b"model")
        (tmp_path / "yamnet.onnx").write_bytes(b"model")

        def fail_run(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("curl must not run when models exist")

        monkeypatch.setattr(subprocess, "run", fail_run)
        ensure_models(tmp_path, announce=lambda line: None)

    def test_missing_model_is_downloaded_with_curl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing file triggers one curl download to a temp path."""
        (tmp_path / "silero_vad.onnx").write_bytes(b"model")
        commands: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            Path(command[command.index("--output") + 1]).write_bytes(b"downloaded")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        announced: list[str] = []
        ensure_models(tmp_path, announce=announced.append)
        assert len(commands) == 1
        assert commands[0][0] == "curl"
        assert (tmp_path / YAMNET_MODEL_FILENAME).read_bytes() == b"downloaded"
        assert any(YAMNET_MODEL_FILENAME in line for line in announced)

    def test_failed_download_raises_and_leaves_no_model_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A curl failure raises RuntimeError; no target file appears."""
        (tmp_path / "silero_vad.onnx").write_bytes(b"model")

        def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, 22, stdout="", stderr="HTTP 404")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuntimeError):
            ensure_models(tmp_path, announce=lambda line: None)
        assert not (tmp_path / YAMNET_MODEL_FILENAME).exists()


def write_wav(path: Path, rate: int, channels: int, sample_width: int, samples: bytes) -> None:
    """Write a small WAV file with the given raw sample bytes."""
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(rate)
        writer.writeframes(samples)


class TestReadWavMono16k:
    def test_reads_pcm16_mono_as_normalized_float32(self, tmp_path: Path) -> None:
        """Sample value 16384 becomes 0.5."""
        path = tmp_path / "ok.wav"
        write_wav(path, rate=16000, channels=1, sample_width=2, samples=np.array([16384, -16384], dtype=np.int16).tobytes())
        wav = read_wav_mono_16k(path)
        assert wav.dtype == np.float32
        assert_close(wav, [0.5, -0.5])

    def test_rejects_wrong_sample_rate(self, tmp_path: Path) -> None:
        """44.1 kHz input is out of contract."""
        path = tmp_path / "rate.wav"
        write_wav(path, rate=44100, channels=1, sample_width=2, samples=np.zeros(4, dtype=np.int16).tobytes())
        with pytest.raises(ValueError):
            read_wav_mono_16k(path)

    def test_rejects_stereo(self, tmp_path: Path) -> None:
        """Two channels are out of contract."""
        path = tmp_path / "stereo.wav"
        write_wav(path, rate=16000, channels=2, sample_width=2, samples=np.zeros(8, dtype=np.int16).tobytes())
        with pytest.raises(ValueError):
            read_wav_mono_16k(path)

    def test_rejects_non_16_bit_samples(self, tmp_path: Path) -> None:
        """8-bit samples are out of contract."""
        path = tmp_path / "depth.wav"
        write_wav(path, rate=16000, channels=1, sample_width=1, samples=bytes(8))
        with pytest.raises(ValueError):
            read_wav_mono_16k(path)
