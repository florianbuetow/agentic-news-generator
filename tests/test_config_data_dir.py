"""Tests for paths configuration."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import Config, PathsConfig


def get_valid_paths_config() -> dict[str, str]:
    """Return a valid paths configuration dictionary."""
    return {
        "data_dir": "./data/",
        "data_downloads_dir": "./data/downloads",
        "data_downloads_videos_dir": "./data/downloads/videos/",
        "data_downloads_transcripts_dir": "./data/downloads/transcripts",
        "data_downloads_transcripts_hallucinations_dir": "./data/downloads/transcripts-hallucinations",
        "data_downloads_transcripts_cleaned_dir": "./data/downloads/transcripts_cleaned",
        "data_downloads_audio_dir": "./data/downloads/audio",
        "data_downloads_metadata_dir": "./data/downloads/metadata",
        "data_output_dir": "./data/output/",
        "data_input_dir": "./data/input/",
        "data_temp_dir": "./data/temp",
        "data_archive_dir": "./data/archive",
        "data_archive_videos_dir": "./data/archive/videos",
    }


class TestPathsConfig:
    """Test cases for the PathsConfig Pydantic model."""

    def test_valid_paths_config(self) -> None:
        """Test that a valid paths configuration passes validation."""
        paths_data = get_valid_paths_config()
        paths = PathsConfig.model_validate(paths_data)
        assert paths.data_dir == "./data/"
        assert paths.data_downloads_dir == "./data/downloads"
        assert paths.data_archive_videos_dir == "./data/archive/videos"

    def test_missing_required_path(self) -> None:
        """Test that missing required paths raise ValidationError."""
        paths_data = get_valid_paths_config()
        del paths_data["data_dir"]

        with pytest.raises(Exception) as exc_info:
            PathsConfig.model_validate(paths_data)
        assert "data_dir" in str(exc_info.value)

    def test_empty_path_raises_error(self) -> None:
        """Test that empty path strings raise ValidationError."""
        paths_data = get_valid_paths_config()
        paths_data["data_dir"] = ""

        with pytest.raises(Exception) as exc_info:
            PathsConfig.model_validate(paths_data)
        assert "data_dir" in str(exc_info.value)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        paths_data = get_valid_paths_config()
        paths_data["extra_field"] = "should not be allowed"

        with pytest.raises(Exception) as exc_info:
            PathsConfig.model_validate(paths_data)
        assert "extra" in str(exc_info.value).lower()


class TestConfigPaths:
    """Test cases for Config class path handling."""

    def test_load_valid_paths_config(self) -> None:
        """Test loading a valid paths configuration."""
        config_data: dict[str, Any] = {
            "paths": get_valid_paths_config(),
            "channels": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            assert config.getDataDir() == Path("./data/")
            assert config.getDataDownloadsDir() == Path("./data/downloads")
            assert config.getDataArchiveVideosDir() == Path("./data/archive/videos")
        finally:
            temp_path.unlink()

    def test_missing_paths_section_raises_key_error(self) -> None:
        """Test that missing paths section raises KeyError."""
        config_data: dict[str, Any] = {
            "channels": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(KeyError) as exc_info:
                Config(temp_path)
            assert "Missing required key 'paths'" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_missing_path_field_raises_value_error(self) -> None:
        """Test that missing path field raises ValueError."""
        paths_data = get_valid_paths_config()
        del paths_data["data_dir"]
        config_data: dict[str, Any] = {
            "paths": paths_data,
            "channels": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            assert "Paths configuration validation failed" in str(exc_info.value)
            assert "data_dir" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_empty_path_raises_value_error(self) -> None:
        """Test that empty path strings raise ValueError."""
        paths_data = get_valid_paths_config()
        paths_data["data_output_dir"] = ""
        config_data: dict[str, Any] = {
            "paths": paths_data,
            "channels": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            assert "Paths configuration validation failed" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_all_path_getters_return_path_objects(self) -> None:
        """Test that all path getters return Path objects."""
        config_data: dict[str, Any] = {
            "paths": get_valid_paths_config(),
            "channels": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            assert isinstance(config.getDataDir(), Path)
            assert isinstance(config.getDataDownloadsDir(), Path)
            assert isinstance(config.getDataDownloadsVideosDir(), Path)
            assert isinstance(config.getDataDownloadsTranscriptsDir(), Path)
            assert isinstance(config.getDataDownloadsTranscriptsHallucinationsDir(), Path)
            assert isinstance(config.getDataDownloadsAudioDir(), Path)
            assert isinstance(config.getDataDownloadsMetadataDir(), Path)
            assert isinstance(config.getDataOutputDir(), Path)
            assert isinstance(config.getDataInputDir(), Path)
            assert isinstance(config.getDataTempDir(), Path)
            assert isinstance(config.getDataArchiveDir(), Path)
            assert isinstance(config.getDataArchiveVideosDir(), Path)
        finally:
            temp_path.unlink()

    def test_all_path_getters_return_correct_values(self) -> None:
        """Test that all path getters return correct values."""
        config_data: dict[str, Any] = {
            "paths": get_valid_paths_config(),
            "channels": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            # Path normalizes paths, so "./data/" becomes "data"
            assert config.getDataDir() == Path("./data/")
            assert config.getDataDownloadsDir() == Path("./data/downloads")
            assert config.getDataDownloadsVideosDir() == Path("./data/downloads/videos/")
            assert config.getDataDownloadsTranscriptsDir() == Path("./data/downloads/transcripts")
            assert config.getDataDownloadsTranscriptsHallucinationsDir() == Path("./data/downloads/transcripts-hallucinations")
            assert config.getDataDownloadsAudioDir() == Path("./data/downloads/audio")
            assert config.getDataDownloadsMetadataDir() == Path("./data/downloads/metadata")
            assert config.getDataOutputDir() == Path("./data/output/")
            assert config.getDataInputDir() == Path("./data/input/")
            assert config.getDataTempDir() == Path("./data/temp")
            assert config.getDataArchiveDir() == Path("./data/archive")
            assert config.getDataArchiveVideosDir() == Path("./data/archive/videos")
        finally:
            temp_path.unlink()

    def test_absolute_paths_preserved(self) -> None:
        """Test that absolute paths are preserved."""
        paths_data = get_valid_paths_config()
        paths_data["data_dir"] = "/absolute/path/to/data"
        config_data: dict[str, Any] = {
            "paths": paths_data,
            "channels": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            assert str(config.getDataDir()) == "/absolute/path/to/data"
        finally:
            temp_path.unlink()
