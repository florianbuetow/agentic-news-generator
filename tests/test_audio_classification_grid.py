"""Tests for the pure 100 ms grid logic of the audio classification map."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from src.audio_classification.grid import (
    MUSIC,
    OTHER,
    SPEECH,
    absorb_short_runs,
    classify_bins,
    labels_to_intervals,
    max_pool_to_grid,
    nearest_frame_to_grid,
    rms_dbfs_per_bin,
)
from src.config import AudioClassificationConfig


def assert_close(actual: npt.NDArray[np.float32], expected: npt.NDArray[np.float32] | list[float], tolerance: float = 1e-6) -> None:
    """Assert two float sequences match element-wise within tolerance."""
    actual_values = cast(list[float], np.asarray(actual, dtype=np.float64).tolist())
    expected_values = cast(list[float], np.asarray(expected, dtype=np.float64).tolist())
    assert len(actual_values) == len(expected_values)
    for actual_value, expected_value in zip(actual_values, expected_values, strict=True):
        assert abs(actual_value - expected_value) < tolerance


def build_config() -> AudioClassificationConfig:
    """Build a threshold config with the values documented in the design."""
    return AudioClassificationConfig(
        silence_floor_dbfs=-50.0,
        music_threshold=0.5,
        music_override_speech_max=0.2,
        speech_onset_probability=0.5,
        speech_offset_probability=0.35,
        min_segment_seconds=0.3,
    )


class TestMaxPoolToGrid:
    def test_takes_maximum_of_frames_inside_each_bin(self) -> None:
        """Two 50 ms frames per 100 ms bin: each bin keeps the larger value."""
        probs = np.array([0.1, 0.9, 0.2, 0.3], dtype=np.float32)
        pooled = max_pool_to_grid(probs, frame_seconds=0.05, n_bins=2)
        assert_close(pooled, [0.9, 0.3])

    def test_bin_without_frames_is_zero(self) -> None:
        """Bins beyond the last frame get probability 0.0."""
        probs = np.array([0.4, 0.6], dtype=np.float32)
        pooled = max_pool_to_grid(probs, frame_seconds=0.05, n_bins=3)
        assert_close(pooled, [0.6, 0.0, 0.0])

    def test_frame_overlapping_bin_boundary_counts_for_both_bins(self) -> None:
        """A 32 ms frame spanning 96-128 ms belongs to bin 0 and bin 1."""
        probs = np.array([0.0, 0.0, 0.0, 0.8, 0.1, 0.1, 0.1], dtype=np.float32)
        pooled = max_pool_to_grid(probs, frame_seconds=0.032, n_bins=2)
        assert_close(pooled, [0.8, 0.8])


class TestNearestFrameToGrid:
    def test_maps_bin_centers_to_nearest_frame_center(self) -> None:
        """Bin 10 (center 1.05 s) maps to frame 6 for 0.48 s + k * 0.096 s centers."""
        values = np.arange(20, dtype=np.float32)
        mapped = nearest_frame_to_grid(values, frame_period=0.096, first_center=0.48, n_bins=11)
        assert abs(float(mapped[10]) - 6.0) < 1e-6

    def test_clamps_bins_before_first_frame_center(self) -> None:
        """Early bins clamp to the first frame instead of indexing negatively."""
        values = np.array([0.7, 0.2], dtype=np.float32)
        mapped = nearest_frame_to_grid(values, frame_period=0.096, first_center=0.48, n_bins=2)
        assert_close(mapped, [0.7, 0.7])

    def test_clamps_bins_after_last_frame_center(self) -> None:
        """Late bins clamp to the last frame value."""
        values = np.array([0.7, 0.2], dtype=np.float32)
        mapped = nearest_frame_to_grid(values, frame_period=0.096, first_center=0.0, n_bins=30)
        assert abs(float(mapped[29]) - 0.2) < 1e-6


class TestRmsDbfsPerBin:
    def test_full_scale_square_wave_is_zero_dbfs(self) -> None:
        """A +/-1.0 square wave has an RMS of full scale: 0 dBFS."""
        wav = np.ones(3200, dtype=np.float32)
        wav[::2] = -1.0
        levels = rms_dbfs_per_bin(wav, sample_rate=16000, n_bins=2)
        assert_close(levels, [0.0, 0.0], tolerance=1e-3)

    def test_silent_bin_is_far_below_any_silence_floor(self) -> None:
        """Digital silence lands far below -100 dBFS."""
        wav = np.zeros(1600, dtype=np.float32)
        levels = rms_dbfs_per_bin(wav, sample_rate=16000, n_bins=1)
        assert levels[0] < -100.0

    def test_partial_last_bin_uses_only_available_samples(self) -> None:
        """A trailing bin with 40 ms of loud audio is still loud."""
        wav = np.concatenate(
            [
                np.zeros(1600, dtype=np.float32),
                np.full(640, 0.5, dtype=np.float32),
            ]
        )
        levels = rms_dbfs_per_bin(wav, sample_rate=16000, n_bins=2)
        assert levels[0] < -100.0
        assert abs(float(levels[1]) - 20.0 * math.log10(0.5)) < 1e-3


class TestClassifyBins:
    def test_silence_wins_over_speech_and_music(self) -> None:
        """Bins below the silence floor are OTHER even with confident classifiers."""
        labels = classify_bins(
            p_speech=np.array([0.9], dtype=np.float32),
            p_music=np.array([0.9], dtype=np.float32),
            p_yamnet_speech=np.array([0.9], dtype=np.float32),
            rms_dbfs=np.array([-80.0], dtype=np.float32),
            config=build_config(),
        )
        assert labels.tolist() == [OTHER]

    def test_speech_hysteresis_keeps_speech_until_offset_threshold(self) -> None:
        """Speech stays active while probability holds above the offset threshold."""
        labels = classify_bins(
            p_speech=np.array([0.6, 0.4, 0.4, 0.2, 0.4], dtype=np.float32),
            p_music=np.zeros(5, dtype=np.float32),
            p_yamnet_speech=np.ones(5, dtype=np.float32),
            rms_dbfs=np.zeros(5, dtype=np.float32),
            config=build_config(),
        )
        assert labels.tolist() == [SPEECH, SPEECH, SPEECH, OTHER, OTHER]

    def test_strong_music_with_weak_yamnet_speech_overrides_silero(self) -> None:
        """Singing trips Silero; YAMNet arbitration turns the bin into MUSIC."""
        labels = classify_bins(
            p_speech=np.array([0.9], dtype=np.float32),
            p_music=np.array([0.9], dtype=np.float32),
            p_yamnet_speech=np.array([0.1], dtype=np.float32),
            rms_dbfs=np.zeros(1, dtype=np.float32),
            config=build_config(),
        )
        assert labels.tolist() == [MUSIC]

    def test_speech_beats_music_when_yamnet_also_hears_speech(self) -> None:
        """Narration over a music bed stays SPEECH."""
        labels = classify_bins(
            p_speech=np.array([0.9], dtype=np.float32),
            p_music=np.array([0.9], dtype=np.float32),
            p_yamnet_speech=np.array([0.5], dtype=np.float32),
            rms_dbfs=np.zeros(1, dtype=np.float32),
            config=build_config(),
        )
        assert labels.tolist() == [SPEECH]

    def test_music_without_speech_is_music(self) -> None:
        """Music above threshold with no speech becomes MUSIC."""
        labels = classify_bins(
            p_speech=np.array([0.1], dtype=np.float32),
            p_music=np.array([0.6], dtype=np.float32),
            p_yamnet_speech=np.array([0.4], dtype=np.float32),
            rms_dbfs=np.zeros(1, dtype=np.float32),
            config=build_config(),
        )
        assert labels.tolist() == [MUSIC]

    def test_nothing_confident_is_other(self) -> None:
        """No classifier above threshold: OTHER."""
        labels = classify_bins(
            p_speech=np.array([0.2], dtype=np.float32),
            p_music=np.array([0.2], dtype=np.float32),
            p_yamnet_speech=np.array([0.2], dtype=np.float32),
            rms_dbfs=np.zeros(1, dtype=np.float32),
            config=build_config(),
        )
        assert labels.tolist() == [OTHER]


class TestAbsorbShortRuns:
    def test_short_run_between_long_runs_is_absorbed(self) -> None:
        """A single MUSIC bin inside SPEECH becomes SPEECH."""
        labels = np.array([SPEECH] * 5 + [MUSIC] + [SPEECH] * 5, dtype=np.int8)
        smoothed = absorb_short_runs(labels, min_bins=3)
        assert smoothed.tolist() == [SPEECH] * 11

    def test_short_run_joins_the_longer_neighbor(self) -> None:
        """A short MUSIC run between SPEECH(5) and OTHER(7) joins OTHER."""
        labels = np.array([SPEECH] * 5 + [MUSIC] * 2 + [OTHER] * 7, dtype=np.int8)
        smoothed = absorb_short_runs(labels, min_bins=3)
        assert smoothed.tolist() == [SPEECH] * 5 + [OTHER] * 9

    def test_tie_between_neighbors_prefers_the_previous_run(self) -> None:
        """Equal neighbors: the short run joins the previous one."""
        labels = np.array([SPEECH] * 4 + [MUSIC] * 2 + [OTHER] * 4, dtype=np.int8)
        smoothed = absorb_short_runs(labels, min_bins=3)
        assert smoothed.tolist() == [SPEECH] * 6 + [OTHER] * 4

    def test_short_leading_run_joins_the_following_run(self) -> None:
        """A short run at the start has only one neighbor to join."""
        labels = np.array([MUSIC] * 2 + [SPEECH] * 6, dtype=np.int8)
        smoothed = absorb_short_runs(labels, min_bins=3)
        assert smoothed.tolist() == [SPEECH] * 8

    def test_all_runs_short_collapses_to_one_class(self) -> None:
        """Repeated absorption terminates with a single class."""
        labels = np.array([SPEECH, MUSIC, OTHER, MUSIC], dtype=np.int8)
        smoothed = absorb_short_runs(labels, min_bins=5)
        assert len(set(smoothed.tolist())) == 1

    def test_min_bins_one_changes_nothing(self) -> None:
        """min_bins=1 keeps every run."""
        labels = np.array([SPEECH, MUSIC, OTHER], dtype=np.int8)
        smoothed = absorb_short_runs(labels, min_bins=1)
        assert smoothed.tolist() == [SPEECH, MUSIC, OTHER]


class TestLabelsToIntervals:
    def test_runs_become_rounded_second_intervals(self) -> None:
        """Adjacent equal labels merge into one [start, end) interval."""
        labels = np.array([SPEECH, SPEECH, MUSIC], dtype=np.int8)
        intervals = labels_to_intervals(labels)
        assert intervals == {
            "speech": [(0.0, 0.2)],
            "music": [(0.2, 0.3)],
            "other": [],
        }

    def test_repeated_class_keeps_separate_intervals(self) -> None:
        """A class interrupted by another class yields two intervals."""
        labels = np.array([SPEECH, OTHER, SPEECH], dtype=np.int8)
        intervals = labels_to_intervals(labels)
        assert intervals["speech"] == [(0.0, 0.1), (0.2, 0.3)]
        assert intervals["other"] == [(0.1, 0.2)]

    def test_timestamps_have_no_float_noise(self) -> None:
        """Bin 3 starts at exactly 0.3, not 0.30000000000000004."""
        labels = np.array([OTHER] * 3 + [MUSIC] * 4, dtype=np.int8)
        intervals = labels_to_intervals(labels)
        assert intervals["music"] == [(0.3, 0.7)]

    def test_empty_labels_give_three_empty_classes(self) -> None:
        """No bins: all three classes present and empty."""
        intervals = labels_to_intervals(np.array([], dtype=np.int8))
        assert intervals == {"speech": [], "music": [], "other": []}


class TestAudioClassificationConfig:
    def test_offset_above_onset_is_rejected(self) -> None:
        """The hysteresis offset must not exceed the onset probability."""
        with pytest.raises(ValueError):
            AudioClassificationConfig(
                silence_floor_dbfs=-50.0,
                music_threshold=0.5,
                music_override_speech_max=0.2,
                speech_onset_probability=0.4,
                speech_offset_probability=0.5,
                min_segment_seconds=0.3,
            )

    @pytest.mark.parametrize("music_threshold", [0.0, 1.0, -0.1, 1.5])
    def test_music_threshold_must_be_a_probability(self, music_threshold: float) -> None:
        """Thresholds outside (0, 1) are rejected."""
        with pytest.raises(ValueError):
            AudioClassificationConfig(
                silence_floor_dbfs=-50.0,
                music_threshold=music_threshold,
                music_override_speech_max=0.2,
                speech_onset_probability=0.5,
                speech_offset_probability=0.35,
                min_segment_seconds=0.3,
            )
