"""Tests for hallucination detection relative path generation."""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.config import Config
from src.processing.repetition_detector import RepetitionDetector

# Import the hallucination detection script
script_path = Path(__file__).parent.parent / "scripts" / "transcript-hallucination-detection.py"
spec = importlib.util.spec_from_file_location("transcript_hallucination_detection", script_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load script: {script_path}")
hallucination_module = importlib.util.module_from_spec(spec)
sys.modules["transcript_hallucination_detection"] = hallucination_module
spec.loader.exec_module(hallucination_module)

detect_hallucinations_in_file = hallucination_module.detect_hallucinations_in_file
process_srt_file = hallucination_module.process_srt_file


class TestHallucinationRelativePaths:
    """Test that hallucination detection generates correct relative paths."""

    def test_relative_path_format_in_detect_hallucinations_in_file(self, temp_config_with_paths: tuple[Path, Config]) -> None:
        """Test that detect_hallucinations_in_file generates relative paths."""
        temp_dir, config = temp_config_with_paths
        data_dir = temp_dir / config.getDataDir()

        # Create test SRT file
        srt_dir = data_dir / "downloads" / "transcripts" / "test_channel"
        srt_dir.mkdir(parents=True, exist_ok=True)
        srt_file = srt_dir / "test_video.srt"

        srt_content = """1
00:00:00,000 --> 00:00:02,000
Test subtitle content here

2
00:00:02,000 --> 00:00:04,000
More test content
"""
        srt_file.write_text(srt_content, encoding="utf-8")

        # Initialize detector
        detector = RepetitionDetector(
            min_k=config.getRepetitionMinK(),
            min_repetitions=config.getRepetitionMinRepetitions(),
            config=config,
        )

        # Run detection
        result = detect_hallucinations_in_file(
            srt_file=srt_file,
            detector=detector,
            min_window_size=50,
            overlap_percent=25.0,
            data_dir=data_dir,
        )

        # Verify source_file is relative
        source_file = result["source_file"]
        assert not source_file.startswith("/"), "Path should not be absolute"
        assert not (len(source_file) > 1 and source_file[1] == ":"), "Path should not have Windows drive letter"
        assert source_file == "downloads/transcripts/test_channel/test_video.srt"

    def test_relative_path_uses_posix_format(self, temp_config_with_paths: tuple[Path, Config]) -> None:
        """Test that generated paths use forward slashes (POSIX format)."""
        temp_dir, config = temp_config_with_paths
        data_dir = temp_dir / config.getDataDir()

        # Create test SRT file
        srt_dir = data_dir / "downloads" / "transcripts" / "test_channel"
        srt_dir.mkdir(parents=True, exist_ok=True)
        srt_file = srt_dir / "test_video.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest\n", encoding="utf-8")

        # Initialize detector
        detector = RepetitionDetector(
            min_k=config.getRepetitionMinK(),
            min_repetitions=config.getRepetitionMinRepetitions(),
            config=config,
        )

        # Run detection
        result = detect_hallucinations_in_file(
            srt_file=srt_file,
            detector=detector,
            min_window_size=50,
            overlap_percent=25.0,
            data_dir=data_dir,
        )

        # Verify POSIX format (forward slashes)
        source_file = result["source_file"]
        assert "\\" not in source_file, "Path should use forward slashes, not backslashes"
        assert "/" in source_file, "Path should contain forward slashes"

    def test_relative_path_can_be_resolved_back_to_file(self, temp_config_with_paths: tuple[Path, Config]) -> None:
        """Test that relative paths can be resolved back to the original file."""
        temp_dir, config = temp_config_with_paths
        data_dir = temp_dir / config.getDataDir()

        # Create test SRT file
        srt_dir = data_dir / "downloads" / "transcripts" / "test_channel"
        srt_dir.mkdir(parents=True, exist_ok=True)
        srt_file = srt_dir / "test_video.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest\n", encoding="utf-8")

        # Initialize detector
        detector = RepetitionDetector(
            min_k=config.getRepetitionMinK(),
            min_repetitions=config.getRepetitionMinRepetitions(),
            config=config,
        )

        # Run detection
        result = detect_hallucinations_in_file(
            srt_file=srt_file,
            detector=detector,
            min_window_size=50,
            overlap_percent=25.0,
            data_dir=data_dir,
        )

        # Resolve relative path back to absolute
        source_file = result["source_file"]
        resolved_path = data_dir / source_file

        # Verify it resolves to the original file
        assert resolved_path.exists(), f"Resolved path should exist: {resolved_path}"
        assert resolved_path == srt_file, "Resolved path should match original file path"

    def test_relative_path_raises_error_if_file_outside_data_dir(self, temp_config_with_paths: tuple[Path, Config]) -> None:
        """Test that files outside data_dir raise ValueError."""
        temp_dir, config = temp_config_with_paths
        data_dir = temp_dir / config.getDataDir()

        # Create SRT file OUTSIDE data_dir
        outside_dir = temp_dir / "outside"
        outside_dir.mkdir(parents=True, exist_ok=True)
        srt_file = outside_dir / "test_video.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest\n", encoding="utf-8")

        # Initialize detector
        detector = RepetitionDetector(
            min_k=config.getRepetitionMinK(),
            min_repetitions=config.getRepetitionMinRepetitions(),
            config=config,
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="not under data directory"):
            detect_hallucinations_in_file(
                srt_file=srt_file,
                detector=detector,
                min_window_size=50,
                overlap_percent=25.0,
                data_dir=data_dir,
            )

    def test_process_srt_file_creates_json_with_relative_paths(self, temp_config_with_paths: tuple[Path, Config]) -> None:
        """Test that process_srt_file writes JSON with relative paths."""
        temp_dir, config = temp_config_with_paths
        data_dir = temp_dir / config.getDataDir()

        # Create test SRT file
        srt_dir = data_dir / "downloads" / "transcripts" / "test_channel"
        srt_dir.mkdir(parents=True, exist_ok=True)
        srt_file = srt_dir / "test_video.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest\n", encoding="utf-8")

        # Create output directory
        output_dir = temp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize detector
        detector = RepetitionDetector(
            min_k=config.getRepetitionMinK(),
            min_repetitions=config.getRepetitionMinRepetitions(),
            config=config,
        )

        # Process file
        success, error_msg, _count = process_srt_file(
            srt_file=srt_file,
            output_base_dir=output_dir,
            detector=detector,
            min_window_size=50,
            overlap_percent=25.0,
            data_dir=data_dir,
        )

        assert success is True
        assert error_msg is None

        # Read generated JSON
        json_file = output_dir / "test_channel" / "test_video.json"
        assert json_file.exists()

        with json_file.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        # Verify relative path
        source_file = data["source_file"]
        assert source_file == "downloads/transcripts/test_channel/test_video.srt"
        assert not source_file.startswith("/")
        assert "\\" not in source_file


@pytest.fixture
def temp_config_with_paths() -> Any:
    """Create a temporary config with paths section for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create config structure
        config_dir = temp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.yaml"

        # Create data directories
        data_dir = temp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Write config
        config_content = f"""
paths:
  data_dir: {data_dir}
  data_downloads_dir: {data_dir / "downloads"}
  data_downloads_videos_dir: {data_dir / "downloads" / "videos"}
  data_downloads_transcripts_dir: {data_dir / "downloads" / "transcripts"}
  data_downloads_transcripts_hallucinations_dir: {data_dir / "downloads" / "transcripts-hallucinations"}
  data_downloads_audio_dir: {data_dir / "downloads" / "audio"}
  data_downloads_metadata_dir: {data_dir / "downloads" / "metadata"}
  data_output_dir: {data_dir / "output"}
  data_input_dir: {data_dir / "input"}
  data_temp_dir: {data_dir / "temp"}
  data_archive_dir: {data_dir / "archive"}
  data_archive_videos_dir: {data_dir / "archive" / "videos"}

channels: []

hallucination_detection:
  min_window_size: 500
  overlap_percent: 25.0
  output_dir: downloads/transcripts-hallucinations
  coef_repetitions: 0.8888460000
  coef_sequence_length: 0.6665380000
  intercept: -6.7770510000

defaults:
  encoding_name: o200k_base
  repetition_min_k: 1
  repetition_min_repetitions: 5
  detect_min_k: 3
"""
        config_file.write_text(config_content, encoding="utf-8")

        config = Config(config_file)
        yield temp_path, config
