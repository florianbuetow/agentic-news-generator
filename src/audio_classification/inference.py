"""ONNX inference plumbing for the audio classification map.

Two one-vs-rest classifiers run over each waveform:

* Silero VAD v6 (speech vs. rest) consumes 512-sample (32 ms) chunks. The
  waveform is split into contiguous streams that run as one batch; every
  stream after the first replays the previous stream's tail chunks as a
  discarded warm lead-in so the recurrent state is warm at each seam.
* YAMNet (music vs. rest) natively hops 0.48 s. Several passes over the
  waveform, each shifted by 96 ms, interleave into a fine music grid; the
  same scores also provide a YAMNet speech posterior for arbitration.

Model files download once via curl into the configured models directory.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import onnxruntime

SAMPLE_RATE: int = 16000
SILERO_CHUNK_SAMPLES: int = 512
SILERO_CONTEXT_SAMPLES: int = 64
SILERO_BATCH_SIZE: int = 16
SILERO_WARMUP_CHUNKS: int = 32
SILERO_STATE_DIMENSION: int = 128
SILERO_MODEL_FILENAME: str = "silero_vad.onnx"
SILERO_MODEL_URL: str = "https://github.com/snakers4/silero-vad/raw/v6.0/src/silero_vad/data/silero_vad.onnx"
YAMNET_MODEL_FILENAME: str = "yamnet.onnx"
YAMNET_MODEL_URL: str = "https://huggingface.co/andrelgomes/yamnet-onnx/resolve/main/yamnet.onnx"
YAMNET_PASSES: int = 5
YAMNET_PASS_OFFSET_SAMPLES: int = 1536
YAMNET_FRAME_PERIOD_SECONDS: float = 0.096
YAMNET_FIRST_CENTER_SECONDS: float = 0.48
YAMNET_MIN_INPUT_SAMPLES: int = 15600
YAMNET_CLASS_COUNT: int = 521
YAMNET_MUSIC_CLASS_START: int = 132
YAMNET_MUSIC_CLASS_END: int = 277
YAMNET_SPEECH_CLASS_INDICES: tuple[int, ...] = (0, 1, 2, 3, 5)
YAMNET_INTRA_OP_THREADS: int = 8
MODEL_DOWNLOADS: tuple[tuple[str, str], ...] = (
    (SILERO_MODEL_FILENAME, SILERO_MODEL_URL),
    (YAMNET_MODEL_FILENAME, YAMNET_MODEL_URL),
)


class OnnxRunnable(Protocol):
    """Structural type for the one onnxruntime session method this module uses."""

    def run(self, output_names: Sequence[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        """Run the model on the given named inputs and return its outputs."""
        ...


def silero_speech_probabilities(
    session: OnnxRunnable,
    wav: npt.NDArray[np.float32],
    batch_size: int,
    warmup_chunks: int,
) -> npt.NDArray[np.float32]:
    """Run Silero VAD over the waveform and return one probability per 32 ms chunk.

    Args:
        session: Silero VAD ONNX session.
        wav: 16 kHz mono waveform in [-1.0, 1.0].
        batch_size: Number of parallel streams the waveform is split into.
        warmup_chunks: Discarded lead-in chunks replayed at each stream seam.

    Returns:
        One speech probability per full 512-sample chunk; a trailing partial
        chunk produces no probability.

    Raises:
        ValueError: If batch_size or warmup_chunks is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if warmup_chunks <= 0:
        raise ValueError(f"warmup_chunks must be positive, got {warmup_chunks}")
    n_chunks = len(wav) // SILERO_CHUNK_SAMPLES
    if n_chunks == 0:
        return np.zeros(0, dtype=np.float32)
    chunks = wav[: n_chunks * SILERO_CHUNK_SAMPLES].reshape(n_chunks, SILERO_CHUNK_SAMPLES)
    chunks_per_stream = math.ceil(n_chunks / min(batch_size, n_chunks))
    n_streams = math.ceil(n_chunks / chunks_per_stream)
    total_steps = warmup_chunks + chunks_per_stream
    batched = np.zeros((n_streams, total_steps, SILERO_CHUNK_SAMPLES), dtype=np.float32)
    own_lengths: list[int] = []
    for stream in range(n_streams):
        own_start = stream * chunks_per_stream
        own_end = min(own_start + chunks_per_stream, n_chunks)
        own_lengths.append(own_end - own_start)
        warmup_available = min(warmup_chunks, own_start)
        warmup_start = warmup_chunks - warmup_available
        batched[stream, warmup_start:warmup_chunks] = chunks[own_start - warmup_available : own_start]
        batched[stream, warmup_chunks : warmup_chunks + own_end - own_start] = chunks[own_start:own_end]
    state = np.zeros((2, n_streams, SILERO_STATE_DIMENSION), dtype=np.float32)
    context = np.zeros((n_streams, SILERO_CONTEXT_SAMPLES), dtype=np.float32)
    sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)
    step_probs = np.zeros((n_streams, total_steps), dtype=np.float32)
    for step in range(total_steps):
        model_input = np.concatenate([context, batched[:, step, :]], axis=1)
        outputs = session.run(None, {"input": model_input, "state": state, "sr": sample_rate})
        step_probs[:, step] = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        state = np.asarray(outputs[1], dtype=np.float32)
        context = batched[:, step, SILERO_CHUNK_SAMPLES - SILERO_CONTEXT_SAMPLES :]
    probs = np.zeros(n_chunks, dtype=np.float32)
    for stream in range(n_streams):
        own_start = stream * chunks_per_stream
        probs[own_start : own_start + own_lengths[stream]] = step_probs[stream, warmup_chunks : warmup_chunks + own_lengths[stream]]
    return probs


def yamnet_music_and_speech(
    session: OnnxRunnable,
    wav: npt.NDArray[np.float32],
    passes: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Run YAMNet in shifted passes and return interleaved music and speech tracks.

    Pass k runs on the waveform shifted by k * 96 ms; the per-pass 0.48 s
    frame sequences interleave into one sequence with a 96 ms period whose
    first frame center sits at 0.48 s.

    Args:
        session: YAMNet ONNX session.
        wav: 16 kHz mono waveform in [-1.0, 1.0].
        passes: Number of shifted passes to interleave.

    Returns:
        Tuple of (music probabilities, speech posterior), both on the
        interleaved grid. Audio shorter than one YAMNet window yields one
        zero-valued pseudo frame per track.

    Raises:
        ValueError: If passes is not positive.
    """
    if passes <= 0:
        raise ValueError(f"passes must be positive, got {passes}")
    music_per_pass: list[npt.NDArray[np.float32]] = []
    speech_per_pass: list[npt.NDArray[np.float32]] = []
    speech_indices = np.array(YAMNET_SPEECH_CLASS_INDICES, dtype=np.int64)
    for k in range(passes):
        shifted = wav[k * YAMNET_PASS_OFFSET_SAMPLES :]
        if len(shifted) < YAMNET_MIN_INPUT_SAMPLES:
            break
        outputs = session.run(None, {"waveform": shifted})
        scores = np.asarray(outputs[0], dtype=np.float32)
        if len(scores) == 0:
            break
        music_per_pass.append(scores[:, YAMNET_MUSIC_CLASS_START:YAMNET_MUSIC_CLASS_END].max(axis=1))
        speech_per_pass.append(scores[:, speech_indices].max(axis=1))
    if not music_per_pass:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    min_frames = min(len(track) for track in music_per_pass)
    if min_frames == 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    music = np.stack([track[:min_frames] for track in music_per_pass], axis=1).reshape(-1)
    speech = np.stack([track[:min_frames] for track in speech_per_pass], axis=1).reshape(-1)
    return music, speech


def ensure_models(models_dir: Path, announce: Callable[[str], None]) -> None:
    """Download any missing model file into models_dir via curl.

    Args:
        models_dir: Directory that holds the ONNX model files.
        announce: Callback for one-line progress messages.

    Raises:
        RuntimeError: If a download fails or produces an empty file.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_DOWNLOADS:
        target = models_dir / filename
        if target.is_file() and target.stat().st_size > 0:
            continue
        announce(f"Downloading {filename} from {url}...")
        temp_target = models_dir / f"{filename}.download"
        result = subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error", "--output", str(temp_target), url],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            temp_target.unlink(missing_ok=True)
            raise RuntimeError(f"Download of {filename} failed with exit code {result.returncode}: {result.stderr.strip()}")
        if not temp_target.is_file() or temp_target.stat().st_size == 0:
            temp_target.unlink(missing_ok=True)
            raise RuntimeError(f"Download of {filename} produced no data")
        temp_target.replace(target)
        announce(f"Downloaded {filename} ({target.stat().st_size} bytes)")


def load_silero_session(models_dir: Path) -> onnxruntime.InferenceSession:
    """Create the Silero VAD ONNX session on the CPU execution provider."""
    return onnxruntime.InferenceSession(
        str(models_dir / SILERO_MODEL_FILENAME),
        providers=["CPUExecutionProvider"],
    )


def load_yamnet_session(models_dir: Path) -> onnxruntime.InferenceSession:
    """Create the YAMNet ONNX session on the CPU execution provider."""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = YAMNET_INTRA_OP_THREADS
    return onnxruntime.InferenceSession(
        str(models_dir / YAMNET_MODEL_FILENAME),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
