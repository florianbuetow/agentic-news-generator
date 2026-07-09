"""Tests for paths configuration."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import Config, PathsConfig, UrlProcessingConfig


def get_valid_paths_config() -> dict[str, str]:
    """Return a valid paths configuration dictionary."""
    return {
        "data_dir": "./data/",
        "data_models_dir": "./data/models/",
        "data_downloads_dir": "./data/downloads",
        "data_downloads_videos_dir": "./data/downloads/videos/",
        "data_downloads_transcripts_dir": "./data/downloads/transcripts",
        "data_downloads_transcripts_hallucinations_dir": "./data/downloads/transcripts-hallucinations",
        "data_downloads_transcripts_cleaned_dir": "./data/downloads/transcripts_cleaned",
        "data_downloads_transcripts_summaries_dir": "./data/downloads/transcripts_summaries",
        "data_downloads_audio_dir": "./data/downloads/audio",
        "data_downloads_metadata_dir": "./data/downloads/metadata",
        "data_output_dir": "./data/output/",
        "data_input_dir": "./data/input/",
        "data_temp_dir": "./data/temp",
        "data_archive_dir": "./data/archive",
        "data_archive_videos_dir": "./data/archive/videos",
        "data_logs_dir": "./data/logs",
        "reports_dir": "reports",
    }


def get_valid_url_processing_config() -> dict[str, str]:
    """Return a valid URL processing configuration dictionary."""
    return {
        "base_dir": "urls",
        "inbox_dir": "inbox_urls",
        "raw_dir": "data_raw",
        "cleaned_dir": "data_cleaned",
        "test_base_dir": ".test/urls",
    }


def get_valid_llm_config() -> dict[str, object]:
    """Return a valid LLM config dictionary."""
    return {
        "models": ["openai/test-model"],
        "api_base": "http://127.0.0.1:1234/v1",
        "api_key": "test-key",
        "context_window": 1000,
        "max_tokens": 100,
        "temperature": 0.0,
        "context_window_threshold": 90,
        "max_retries": 1,
        "retry_delay": 1.0,
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


class TestUrlProcessingConfig:
    """Test cases for the UrlProcessingConfig Pydantic model."""

    def test_valid_url_processing_config(self) -> None:
        """Test that a valid URL processing configuration passes validation."""
        url_processing = UrlProcessingConfig.model_validate(get_valid_url_processing_config())
        assert url_processing.base_dir == "urls"
        assert url_processing.inbox_dir == "inbox_urls"
        assert url_processing.raw_dir == "data_raw"
        assert url_processing.cleaned_dir == "data_cleaned"
        assert url_processing.test_base_dir == ".test/urls"

    def test_absolute_url_processing_path_raises_error(self) -> None:
        """Test that URL processing paths must be relative to data_dir."""
        url_processing_data = get_valid_url_processing_config()
        url_processing_data["base_dir"] = "/absolute/urls"

        with pytest.raises(Exception) as exc_info:
            UrlProcessingConfig.model_validate(url_processing_data)
        assert "relative to data_dir" in str(exc_info.value)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        url_processing_data = get_valid_url_processing_config()
        url_processing_data["extra_field"] = "should not be allowed"

        with pytest.raises(Exception) as exc_info:
            UrlProcessingConfig.model_validate(url_processing_data)
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
            assert config.get_data_dir() == Path("./data/")
            assert config.get_data_downloads_dir() == Path("./data/downloads")
            assert config.get_data_archive_videos_dir() == Path("./data/archive/videos")
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
            assert isinstance(config.get_data_dir(), Path)
            assert isinstance(config.get_data_models_dir(), Path)
            assert isinstance(config.get_data_downloads_dir(), Path)
            assert isinstance(config.get_data_downloads_videos_dir(), Path)
            assert isinstance(config.get_data_downloads_transcripts_dir(), Path)
            assert isinstance(config.get_data_downloads_transcripts_hallucinations_dir(), Path)
            assert isinstance(config.get_data_downloads_audio_dir(), Path)
            assert isinstance(config.get_data_downloads_metadata_dir(), Path)
            assert isinstance(config.get_data_output_dir(), Path)
            assert isinstance(config.get_data_input_dir(), Path)
            assert isinstance(config.get_data_temp_dir(), Path)
            assert isinstance(config.get_data_archive_dir(), Path)
            assert isinstance(config.get_data_archive_videos_dir(), Path)
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
            assert config.get_data_dir() == Path("./data/")
            assert config.get_data_models_dir() == Path("./data/models/")
            assert config.get_data_downloads_dir() == Path("./data/downloads")
            assert config.get_data_downloads_videos_dir() == Path("./data/downloads/videos/")
            assert config.get_data_downloads_transcripts_dir() == Path("./data/downloads/transcripts")
            assert config.get_data_downloads_transcripts_hallucinations_dir() == Path("./data/downloads/transcripts-hallucinations")
            assert config.get_data_downloads_audio_dir() == Path("./data/downloads/audio")
            assert config.get_data_downloads_metadata_dir() == Path("./data/downloads/metadata")
            assert config.get_data_output_dir() == Path("./data/output/")
            assert config.get_data_input_dir() == Path("./data/input/")
            assert config.get_data_temp_dir() == Path("./data/temp")
            assert config.get_data_archive_dir() == Path("./data/archive")
            assert config.get_data_archive_videos_dir() == Path("./data/archive/videos")
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
            assert str(config.get_data_dir()) == "/absolute/path/to/data"
        finally:
            temp_path.unlink()

    def test_url_processing_getters_resolve_relative_to_data_dir(self) -> None:
        """Test URL processing path getters resolve under data_dir and base_dir."""
        config_data: dict[str, Any] = {
            "paths": get_valid_paths_config(),
            "channels": [],
            "url_processing": get_valid_url_processing_config(),
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            assert config.get_url_base_dir() == Path("./data/") / "urls"
            assert config.get_url_inbox_dir() == Path("./data/") / "urls" / "inbox_urls"
            assert config.get_url_raw_dir() == Path("./data/") / "urls" / "data_raw"
            assert config.get_url_cleaned_dir() == Path("./data/") / "urls" / "data_cleaned"
            assert config.get_url_test_base_dir() == Path("./data/") / ".test" / "urls"
        finally:
            temp_path.unlink()

    def test_url_clean_content_config_loads(self) -> None:
        """Test URL clean-content config validation and getter."""
        config_data: dict[str, Any] = {
            "paths": get_valid_paths_config(),
            "channels": [],
            "url_clean_content": {
                "prompt_template": "prompts/format-url-content.md",
                "skip_documents_above_context_window_pct": 80,
                "llm": get_valid_llm_config(),
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            clean_config = config.get_url_clean_content_config()
            assert clean_config.prompt_template == "prompts/format-url-content.md"
            assert clean_config.skip_documents_above_context_window_pct == 80
            assert clean_config.llm.models == ["openai/test-model"]
        finally:
            temp_path.unlink()
