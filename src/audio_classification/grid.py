"""Pure logic for the 100 ms speech/music/other classification grid.

The pipeline produces two one-vs-rest probability tracks (speech from Silero
VAD, music from YAMNet) plus a YAMNet speech posterior and per-bin RMS levels.
This module pools those tracks onto a shared 100 ms grid, combines them into
one label per bin, smooths implausibly short runs, and turns the labels into
per-class ``[start, end)`` second intervals.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from src.config import AudioClassificationConfig

GRID_SECONDS: float = 0.1
SPEECH: int = 0
MUSIC: int = 1
OTHER: int = 2
CLASS_NAMES: dict[int, str] = {SPEECH: "speech", MUSIC: "music", OTHER: "other"}
BOUNDARY_EPSILON: float = 1e-9
MIN_RMS_LINEAR: float = 1e-10


def max_pool_to_grid(frame_probs: npt.NDArray[np.float32], frame_seconds: float, n_bins: int) -> npt.NDArray[np.float32]:
    """Pool per-frame probabilities onto the 100 ms grid with a maximum.

    Args:
        frame_probs: One probability per fixed-length frame.
        frame_seconds: Duration of one frame in seconds.
        n_bins: Number of 100 ms bins to produce.

    Returns:
        One probability per bin: the maximum over all frames that overlap the
        bin, or 0.0 for bins beyond the last frame.

    Raises:
        ValueError: If frame_seconds is not positive.
    """
    if frame_seconds <= 0.0:
        raise ValueError(f"frame_seconds must be positive, got {frame_seconds}")
    pooled = np.zeros(n_bins, dtype=np.float32)
    for b in range(n_bins):
        start = b * GRID_SECONDS
        end = start + GRID_SECONDS
        first = max(0, math.floor(start / frame_seconds + BOUNDARY_EPSILON))
        last = min(len(frame_probs), math.ceil(end / frame_seconds - BOUNDARY_EPSILON))
        if first < last:
            pooled[b] = frame_probs[first:last].max()
    return pooled


def nearest_frame_to_grid(
    frame_values: npt.NDArray[np.float32],
    frame_period: float,
    first_center: float,
    n_bins: int,
) -> npt.NDArray[np.float32]:
    """Sample per-frame values onto the 100 ms grid by nearest frame center.

    Args:
        frame_values: One value per frame.
        frame_period: Seconds between consecutive frame centers.
        first_center: Second position of the first frame center.
        n_bins: Number of 100 ms bins to produce.

    Returns:
        One value per bin, taken from the frame whose center is nearest to the
        bin center; bins outside the frame range clamp to the edge frames.

    Raises:
        ValueError: If frame_values is empty or frame_period is not positive.
    """
    if len(frame_values) == 0:
        raise ValueError("frame_values must not be empty")
    if frame_period <= 0.0:
        raise ValueError(f"frame_period must be positive, got {frame_period}")
    bin_centers = (np.arange(n_bins, dtype=np.float64) + 0.5) * GRID_SECONDS
    indices = np.rint((bin_centers - first_center) / frame_period).astype(np.int64)
    indices = np.clip(indices, 0, len(frame_values) - 1)
    return np.asarray(frame_values[indices], dtype=np.float32)


def rms_dbfs_per_bin(wav: npt.NDArray[np.float32], sample_rate: int, n_bins: int) -> npt.NDArray[np.float32]:
    """Compute the RMS level in dBFS for each 100 ms bin.

    Args:
        wav: Mono waveform with samples in [-1.0, 1.0].
        sample_rate: Samples per second of the waveform.
        n_bins: Number of 100 ms bins to produce.

    Returns:
        One dBFS level per bin; empty bins report the floor level of the
        minimal representable RMS.

    Raises:
        ValueError: If sample_rate is not positive.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    samples_per_bin = round(sample_rate * GRID_SECONDS)
    levels = np.full(n_bins, 20.0 * math.log10(MIN_RMS_LINEAR), dtype=np.float32)
    for b in range(n_bins):
        segment = wav[b * samples_per_bin : (b + 1) * samples_per_bin]
        if len(segment) == 0:
            continue
        rms = max(float(np.sqrt(np.mean(np.square(segment, dtype=np.float64)))), MIN_RMS_LINEAR)
        levels[b] = 20.0 * math.log10(rms)
    return levels


def classify_bins(
    p_speech: npt.NDArray[np.float32],
    p_music: npt.NDArray[np.float32],
    p_yamnet_speech: npt.NDArray[np.float32],
    rms_dbfs: npt.NDArray[np.float32],
    config: AudioClassificationConfig,
) -> npt.NDArray[np.int8]:
    """Combine the one-vs-rest tracks into one label per 100 ms bin.

    Decision order per bin: silence floor wins, then strong music with a weak
    YAMNet speech posterior overrides the speech detector (singing trips
    Silero), then hysteresis-tracked speech, then music, otherwise other.

    Args:
        p_speech: Silero speech probability per bin.
        p_music: YAMNet music probability per bin.
        p_yamnet_speech: YAMNet speech posterior per bin.
        rms_dbfs: RMS level in dBFS per bin.
        config: Threshold configuration.

    Returns:
        One label per bin: SPEECH, MUSIC, or OTHER.

    Raises:
        ValueError: If the input arrays have different lengths.
    """
    n_bins = len(p_speech)
    if not (len(p_music) == len(p_yamnet_speech) == len(rms_dbfs) == n_bins):
        raise ValueError(
            f"All tracks must have equal length, got p_speech={n_bins}, p_music={len(p_music)}, "
            f"p_yamnet_speech={len(p_yamnet_speech)}, rms_dbfs={len(rms_dbfs)}"
        )
    labels = np.empty(n_bins, dtype=np.int8)
    speech_active = False
    for b in range(n_bins):
        if speech_active:
            speech_active = float(p_speech[b]) >= config.speech_offset_probability
        else:
            speech_active = float(p_speech[b]) >= config.speech_onset_probability
        music_confident = float(p_music[b]) > config.music_threshold
        if float(rms_dbfs[b]) < config.silence_floor_dbfs:
            labels[b] = OTHER
        elif music_confident and float(p_yamnet_speech[b]) < config.music_override_speech_max:
            labels[b] = MUSIC
        elif speech_active:
            labels[b] = SPEECH
        elif music_confident:
            labels[b] = MUSIC
        else:
            labels[b] = OTHER
    return labels


def absorb_short_runs(labels: npt.NDArray[np.int8], min_bins: int) -> npt.NDArray[np.int8]:
    """Merge label runs shorter than min_bins into a neighboring run.

    The shortest run (leftmost on ties) is merged into its longer neighbor
    (previous neighbor on ties) until every remaining run has at least
    min_bins bins or only one run is left.

    Args:
        labels: One label per bin.
        min_bins: Minimal run length in bins that survives smoothing.

    Returns:
        Smoothed labels with the same length as the input.

    Raises:
        ValueError: If min_bins is not positive.
    """
    if min_bins <= 0:
        raise ValueError(f"min_bins must be positive, got {min_bins}")
    runs = _coalesce_runs([[int(label), 1] for label in labels.tolist()])
    while len(runs) > 1:
        short_lengths = [length for _, length in runs if length < min_bins]
        if not short_lengths:
            break
        index = next(i for i, (_, length) in enumerate(runs) if length == min(short_lengths))
        neighbor = _absorbing_neighbor(runs, index)
        runs[neighbor][1] += runs[index][1]
        del runs[index]
        runs = _coalesce_runs(runs)
    smoothed = np.empty(len(labels), dtype=np.int8)
    position = 0
    for label, length in runs:
        smoothed[position : position + length] = label
        position += length
    return smoothed


def _coalesce_runs(runs: list[list[int]]) -> list[list[int]]:
    """Merge adjacent (label, length) runs that share the same label."""
    merged: list[list[int]] = []
    for label, length in runs:
        if merged and merged[-1][0] == label:
            merged[-1][1] += length
        else:
            merged.append([label, length])
    return merged


def _absorbing_neighbor(runs: list[list[int]], index: int) -> int:
    """Pick the neighbor run that absorbs the run at index.

    The longer neighbor wins; ties and edges fall to the previous run.
    """
    if index == 0:
        return 1
    if index == len(runs) - 1:
        return index - 1
    if runs[index + 1][1] > runs[index - 1][1]:
        return index + 1
    return index - 1


def labels_to_intervals(labels: npt.NDArray[np.int8]) -> dict[str, list[tuple[float, float]]]:
    """Convert per-bin labels into per-class second intervals.

    Args:
        labels: One label per 100 ms bin.

    Returns:
        Mapping of class name to a list of [start, end) second pairs, rounded
        to one decimal; all three classes are always present.
    """
    intervals: dict[str, list[tuple[float, float]]] = {name: [] for name in CLASS_NAMES.values()}
    run_start = 0
    for b in range(1, len(labels) + 1):
        if b == len(labels) or labels[b] != labels[run_start]:
            class_name = CLASS_NAMES[int(labels[run_start])]
            intervals[class_name].append((round(run_start * GRID_SECONDS, 1), round(b * GRID_SECONDS, 1)))
            run_start = b
    return intervals
