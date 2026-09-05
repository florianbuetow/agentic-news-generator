"""WAV reading for the audio classification map.

The extract-audio stage produces 16 kHz mono PCM-16 WAV files. This module
reads exactly that contract and rejects everything else instead of resampling.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

REQUIRED_SAMPLE_RATE: int = 16000
REQUIRED_CHANNELS: int = 1
REQUIRED_SAMPLE_WIDTH_BYTES: int = 2
PCM16_FULL_SCALE: float = 32768.0


def read_wav_mono_16k(path: Path) -> npt.NDArray[np.float32]:
    """Read a 16 kHz mono PCM-16 WAV file as a normalized float32 waveform.

    Args:
        path: WAV file to read.

    Returns:
        Waveform with samples in [-1.0, 1.0].

    Raises:
        ValueError: If the file is not 16 kHz, mono, PCM-16.
        wave.Error: If the file is not a valid WAV file.
        OSError: If the file cannot be read.
    """
    with wave.open(str(path), "rb") as reader:
        sample_rate = reader.getframerate()
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        if sample_rate != REQUIRED_SAMPLE_RATE:
            raise ValueError(f"{path}: sample rate must be {REQUIRED_SAMPLE_RATE} Hz, got {sample_rate} Hz")
        if channels != REQUIRED_CHANNELS:
            raise ValueError(f"{path}: audio must be mono, got {channels} channels")
        if sample_width != REQUIRED_SAMPLE_WIDTH_BYTES:
            raise ValueError(f"{path}: samples must be {REQUIRED_SAMPLE_WIDTH_BYTES * 8}-bit PCM, got {sample_width * 8}-bit")
        frames = reader.readframes(reader.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16)
    return (samples.astype(np.float32)) / PCM16_FULL_SCALE
