"""Batch runner for the speech/music/other audio classification map.

Walks every channel directory below the configured audio directory, runs the
two one-vs-rest classifiers over each pending WAV file, and writes one JSON
map per file: ``{"speech": [[start, end], ...], "music": [...], "other":
[...]}`` with second timestamps on a 100 ms grid, relative to the WAV
timeline. Channels with the fewest pending files are processed first, and a
failure in one file never stops the batch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.audio_classification.audio_io import read_wav_mono_16k
from src.audio_classification.grid import (
    GRID_SECONDS,
    absorb_short_runs,
    classify_bins,
    labels_to_intervals,
    max_pool_to_grid,
    nearest_frame_to_grid,
    rms_dbfs_per_bin,
)
from src.audio_classification.inference import (
    SAMPLE_RATE,
    SILERO_BATCH_SIZE,
    SILERO_CHUNK_SAMPLES,
    SILERO_WARMUP_CHUNKS,
    YAMNET_FIRST_CENTER_SECONDS,
    YAMNET_FRAME_PERIOD_SECONDS,
    YAMNET_PASSES,
    ensure_models,
    load_silero_session,
    load_yamnet_session,
    silero_speech_probabilities,
    yamnet_music_and_speech,
)
from src.config import AudioClassificationConfig, Config

type SpeechProbabilityFn = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
type MusicScoreFn = Callable[[npt.NDArray[np.float32]], tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]

MODELS_SUBDIR: str = "audio-classification"


def classify_waveform(
    wav: npt.NDArray[np.float32],
    tool_config: AudioClassificationConfig,
    speech_fn: SpeechProbabilityFn,
    music_fn: MusicScoreFn,
) -> dict[str, list[tuple[float, float]]]:
    """Classify one waveform into per-class second intervals on the 100 ms grid.

    Args:
        wav: 16 kHz mono waveform in [-1.0, 1.0].
        tool_config: Threshold configuration.
        speech_fn: Returns one speech probability per 32 ms chunk.
        music_fn: Returns (music probabilities, speech posterior) on the
            interleaved YAMNet grid.

    Returns:
        Mapping of class name to [start, end) second pairs; the three classes
        partition the padded-to-grid duration.
    """
    n_bins = math.ceil(len(wav) / (SAMPLE_RATE * GRID_SECONDS))
    if n_bins == 0:
        return {"speech": [], "music": [], "other": []}
    p_speech = max_pool_to_grid(speech_fn(wav), SILERO_CHUNK_SAMPLES / SAMPLE_RATE, n_bins)
    music_track, yamnet_speech_track = music_fn(wav)
    p_music = nearest_frame_to_grid(music_track, YAMNET_FRAME_PERIOD_SECONDS, YAMNET_FIRST_CENTER_SECONDS, n_bins)
    p_yamnet_speech = nearest_frame_to_grid(yamnet_speech_track, YAMNET_FRAME_PERIOD_SECONDS, YAMNET_FIRST_CENTER_SECONDS, n_bins)
    rms_levels = rms_dbfs_per_bin(wav, SAMPLE_RATE, n_bins)
    labels = classify_bins(p_speech, p_music, p_yamnet_speech, rms_levels, tool_config)
    min_bins = max(1, round(tool_config.min_segment_seconds / GRID_SECONDS))
    return labels_to_intervals(absorb_short_runs(labels, min_bins))


def find_channel_work(audio_dir: Path, map_dir: Path) -> list[tuple[str, list[Path]]]:
    """List every channel with its WAV files, fewest pending files first.

    Args:
        audio_dir: Directory holding one subdirectory per channel.
        map_dir: Directory holding the JSON maps, mirrored per channel.

    Returns:
        (channel name, sorted WAV paths) tuples, ordered by the number of WAV
        files that have no JSON map yet, ties by channel name.
    """
    channels: list[tuple[str, list[Path]]] = []
    for channel_dir in sorted(path for path in audio_dir.iterdir() if path.is_dir()):
        wav_paths = sorted(channel_dir.glob("*.wav"))
        if wav_paths:
            channels.append((channel_dir.name, wav_paths))

    def pending_count(entry: tuple[str, list[Path]]) -> tuple[int, str]:
        channel_name, wav_paths = entry
        pending = sum(1 for wav_path in wav_paths if not (map_dir / channel_name / f"{wav_path.stem}.json").is_file())
        return (pending, channel_name)

    return sorted(channels, key=pending_count)


def write_map(map_path: Path, intervals: dict[str, list[tuple[float, float]]]) -> None:
    """Write one classification map atomically as JSON.

    Args:
        map_path: Final JSON file path.
        intervals: Per-class second intervals.
    """
    map_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = map_path.parent / f"{map_path.name}.tmp"
    temp_path.write_text(json.dumps(intervals) + "\n", encoding="utf-8")
    temp_path.replace(map_path)


def run_audio_classification(config: Config, speech_fn: SpeechProbabilityFn, music_fn: MusicScoreFn) -> int:
    """Classify every pending WAV file and report a failure summary.

    Args:
        config: Project configuration.
        speech_fn: Returns one speech probability per 32 ms chunk.
        music_fn: Returns (music probabilities, speech posterior) on the
            interleaved YAMNet grid.

    Returns:
        0 if every pending file was classified, 1 if any file failed.
    """
    tool_config = config.get_audio_classification_config()
    audio_dir = config.get_data_downloads_audio_dir()
    map_dir = config.get_data_downloads_audio_classification_dir()
    processed = 0
    skipped = 0
    failures: list[tuple[str, str]] = []
    for channel_name, wav_paths in find_channel_work(audio_dir, map_dir):
        for wav_path in wav_paths:
            display_name = f"{channel_name}/{wav_path.name}"
            map_path = map_dir / channel_name / f"{wav_path.stem}.json"
            if map_path.is_file():
                print(f"Skipping: {display_name}")
                skipped += 1
                continue
            print(f"Processing: {display_name}")
            try:
                print("classifying...")
                wav = read_wav_mono_16k(wav_path)
                intervals = classify_waveform(wav, tool_config, speech_fn, music_fn)
                print("writing map...")
                write_map(map_path, intervals)
                print("done")
                processed += 1
            except Exception as e:
                print(f"❌ Failed: {e}")
                failures.append((display_name, str(e)))
    print(f"\n{processed} processed, {skipped} skipped, {len(failures)} failed")
    if failures:
        print("\n--- Failure Summary ---")
        for display_name, error in failures:
            print(f"❌ {display_name}: {error}")
        return 1
    return 0


def run_with_onnx_models(config: Config) -> int:
    """Load the ONNX models (downloading them if missing) and run the batch.

    Args:
        config: Project configuration.

    Returns:
        0 if every pending file was classified, 1 if any file failed.

    Raises:
        RuntimeError: If a model download fails.
    """
    models_dir = config.get_data_models_dir() / MODELS_SUBDIR
    ensure_models(models_dir, announce=print)
    silero_session = load_silero_session(models_dir)
    yamnet_session = load_yamnet_session(models_dir)

    def speech_fn(wav: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return silero_speech_probabilities(silero_session, wav, SILERO_BATCH_SIZE, SILERO_WARMUP_CHUNKS)

    def music_fn(wav: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return yamnet_music_and_speech(yamnet_session, wav, YAMNET_PASSES)

    return run_audio_classification(config, speech_fn, music_fn)
