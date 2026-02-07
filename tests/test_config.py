"""Unit tests for the Config class and ChannelConfig model."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from src.config import ChannelConfig, Config


def get_valid_paths_config() -> dict[str, str]:
    """Return a valid paths configuration dictionary for tests."""
    return {
        "data_dir": "./data/",
        "data_models_dir": "./data/models/",
        "data_downloads_dir": "./data/downloads",
        "data_downloads_videos_dir": "./data/downloads/videos/",
        "data_downloads_transcripts_dir": "./data/downloads/transcripts",
        "data_downloads_transcripts_hallucinations_dir": "./data/downloads/transcripts-hallucinations",
        "data_downloads_transcripts_cleaned_dir": "./data/downloads/transcripts_cleaned",
        "data_transcripts_topics_dir": "./data/downloads/transcripts-topics",
        "data_downloads_audio_dir": "./data/downloads/audio",
        "data_downloads_metadata_dir": "./data/downloads/metadata",
        "data_output_dir": "./data/output/",
        "data_input_dir": "./data/input/",
        "data_temp_dir": "./data/temp",
        "data_archive_dir": "./data/archive",
        "data_archive_videos_dir": "./data/archive/videos",
        "data_logs_dir": "./logs",
        "data_output_articles_dir": "./data/output/articles",
        "data_articles_input_dir": "./data/articles/input",
        "reports_dir": "reports",
    }


class TestChannelConfig:
    """Test cases for the ChannelConfig Pydantic model."""

    def test_valid_channel_config(self) -> None:
        """Test that a valid channel configuration passes validation."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            "language": "en",
        }
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.url == "https://www.youtube.com/@test"
        assert channel.name == "Test Channel"
        assert channel.category == "test_category"
        assert channel.description == "Test content"
        assert channel.download_limiter == 20
        assert channel.language == "en"

    def test_missing_required_field(self) -> None:
        """Test that missing required fields raise ValidationError."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            # Missing category, description, download_limiter, language
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert len(errors) == 4  # category, description, download-limiter, language
        error_fields = {error["loc"][0] for error in errors}
        assert error_fields == {"category", "description", "download-limiter", "language"}

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden due to extra='forbid'."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            "language": "en",
            "extra_field": "should not be allowed",
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert any("extra_field" in str(error) for error in errors)

    def test_wrong_field_type(self) -> None:
        """Test that wrong field types raise ValidationError."""
        channel_data = {
            "url": 123,  # Should be string
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            "language": "en",
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "url" for error in errors)

    def test_missing_description_field(self) -> None:
        """Test that missing description field raises ValidationError."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            # Missing description, download_limiter, language
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert len(errors) == 3
        error_fields = {error["loc"][0] for error in errors}
        assert error_fields == {"description", "download-limiter", "language"}

    def test_empty_strings_allowed(self) -> None:
        """Test that empty strings are allowed (validation passes but may not be practical)."""
        channel_data = {
            "url": "",
            "name": "",
            "category": "",
            "description": "",
            "download-limiter": 0,
            "language": "en",
        }
        # Empty strings are valid str types, so validation passes
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.url == ""
        assert channel.name == ""
        assert channel.download_limiter == 0
        assert channel.language == "en"


class TestConfig:
    """Test cases for the Config class."""

    def test_load_valid_config(self) -> None:
        """Test loading a valid configuration file."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test1",
                    "name": "Test Channel 1",
                    "category": "category1",
                    "description": "Content 1",
                    "download-limiter": 20,
                    "language": "en",
                },
                {
                    "url": "https://www.youtube.com/@test2",
                    "name": "Test Channel 2",
                    "category": "category2",
                    "description": "Content 2",
                    "download-limiter": 20,
                    "language": "en",
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            channels = config.get_channels()
            assert len(channels) == 2
            assert channels[0].name == "Test Channel 1"
            assert channels[1].name == "Test Channel 2"
        finally:
            temp_path.unlink()

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised when config file doesn't exist."""
        non_existent_path = Path("/nonexistent/path/config.yaml")
        with pytest.raises(FileNotFoundError) as exc_info:
            Config(non_existent_path)
        assert "Config file not found" in str(exc_info.value)

    def test_invalid_yaml(self) -> None:
        """Test that invalid YAML raises yaml.YAMLError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            temp_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                Config(temp_path)
        finally:
            temp_path.unlink()

    def test_missing_channels_key(self) -> None:
        """Test that missing 'channels' key raises KeyError."""
        config_data = {"other_key": "value"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(KeyError) as exc_info:
                Config(temp_path)
            assert "Missing required key 'channels'" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_channels_not_list(self) -> None:
        """Test that channels not being a list raises ValueError."""
        config_data = {"paths": get_valid_paths_config(), "channels": "not a list"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            assert "'channels' must be a list" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_empty_channels_list(self) -> None:
        """Test that an empty channels list is valid."""
        config_data: dict[str, Any] = {"paths": get_valid_paths_config(), "channels": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            channels = config.get_channels()
            assert len(channels) == 0
        finally:
            temp_path.unlink()

    def test_channel_missing_required_field(self) -> None:
        """Test that channel missing required fields raises ValueError."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    # Missing category, description, download_limiter
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            assert "Channel validation failed" in str(exc_info.value)
            assert "Test Channel" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_channel_missing_name_field(self) -> None:
        """Test that channel missing name field uses index in error message."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    # Missing name, category, description, download_limiter
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            assert "channel at index 0" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_multiple_validation_errors(self) -> None:
        """Test that multiple channel validation errors are all reported."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test1",
                    "name": "Test Channel 1",
                    # Missing category, description, download_limiter
                },
                {
                    "url": "https://www.youtube.com/@test2",
                    "name": "Test Channel 2",
                    # Missing category, description, download_limiter
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            error_msg = str(exc_info.value)
            assert "Test Channel 1" in error_msg
            assert "Test Channel 2" in error_msg
            assert "index 0" in error_msg
            assert "index 1" in error_msg
        finally:
            temp_path.unlink()

    def test_get_channel_valid_index(self) -> None:
        """Test getting a channel by valid index."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test1",
                    "name": "Test Channel 1",
                    "category": "category1",
                    "description": "Content 1",
                    "download-limiter": 20,
                    "language": "en",
                },
                {
                    "url": "https://www.youtube.com/@test2",
                    "name": "Test Channel 2",
                    "category": "category2",
                    "description": "Content 2",
                    "download-limiter": 20,
                    "language": "en",
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            channel = config.get_channel(0)
            assert channel.name == "Test Channel 1"
            channel = config.get_channel(1)
            assert channel.name == "Test Channel 2"
        finally:
            temp_path.unlink()

    def test_get_channel_negative_index(self) -> None:
        """Test that negative index raises IndexError."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            with pytest.raises(IndexError) as exc_info:
                config.get_channel(-1)
            assert "out of range" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_get_channel_index_too_large(self) -> None:
        """Test that index too large raises IndexError."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            with pytest.raises(IndexError) as exc_info:
                config.get_channel(1)
            assert "out of range" in str(exc_info.value)
            assert "0-0" in str(exc_info.value)  # Should show valid range
        finally:
            temp_path.unlink()

    def test_get_channel_by_name_valid(self) -> None:
        """Test getting a channel by valid name."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test1",
                    "name": "Test Channel 1",
                    "category": "category1",
                    "description": "Content 1",
                    "download-limiter": 20,
                    "language": "en",
                },
                {
                    "url": "https://www.youtube.com/@test2",
                    "name": "Test Channel 2",
                    "category": "category2",
                    "description": "Content 2",
                    "download-limiter": 20,
                    "language": "en",
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            channel = config.get_channel_by_name("Test Channel 1")
            assert channel.url == "https://www.youtube.com/@test1"
            channel = config.get_channel_by_name("Test Channel 2")
            assert channel.url == "https://www.youtube.com/@test2"
        finally:
            temp_path.unlink()

    def test_get_channel_by_name_not_found(self) -> None:
        """Test that non-existent channel name raises KeyError."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            with pytest.raises(KeyError) as exc_info:
                config.get_channel_by_name("Non-existent Channel")
            assert "No channel found with name" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_get_channel_by_name_case_sensitive(self) -> None:
        """Test that channel name lookup is case-sensitive."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            with pytest.raises(KeyError):
                config.get_channel_by_name("test channel")  # Different case
        finally:
            temp_path.unlink()

    def test_channel_with_wrong_type(self) -> None:
        """Test that channel with wrong field type raises ValueError."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": 123,  # Should be string
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            assert "Channel validation failed" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_channel_with_extra_field(self) -> None:
        """Test that channel with extra field raises ValueError."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                    "language": "en",
                    "extra_field": "not allowed",
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            assert "Channel validation failed" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_config_path_as_string(self) -> None:
        """Test that config_path can be provided as a string."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path_str = str(f.name)

        try:
            config = Config(temp_path_str)  # Pass as string
            channels = config.get_channels()
            assert len(channels) == 1
        finally:
            Path(temp_path_str).unlink()

    def test_empty_yaml_file(self) -> None:
        """Test that empty YAML file raises KeyError for missing channels."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")  # Empty file
            temp_path = Path(f.name)

        try:
            with pytest.raises(KeyError) as exc_info:
                Config(temp_path)
            assert "Missing required key 'channels'" in str(exc_info.value)
        finally:
            temp_path.unlink()


class TestChannelConfigDownloadLimiter:
    """Test cases for the download_limiter field."""

    def test_download_limiter_zero(self) -> None:
        """Test that download_limiter=0 is valid (skip downloads)."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 0,
            "language": "en",
        }
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.download_limiter == 0

    def test_download_limiter_unlimited(self) -> None:
        """Test that download_limiter=-1 is valid (unlimited)."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": -1,
            "language": "en",
        }
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.download_limiter == -1

    def test_download_limiter_positive(self) -> None:
        """Test that positive download_limiter values are valid."""
        test_values = [1, 5, 10, 20, 50, 100, 99999]
        for value in test_values:
            channel_data = {
                "url": "https://www.youtube.com/@test",
                "name": "Test Channel",
                "category": "test_category",
                "description": "Test content",
                "download-limiter": value,
                "language": "en",
            }
            channel = ChannelConfig.model_validate(channel_data)
            assert channel.download_limiter == value

    def test_download_limiter_missing(self) -> None:
        """Test that missing download_limiter raises ValidationError."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            # Missing download_limiter and language
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert len(errors) == 2
        error_fields = {error["loc"][0] for error in errors}
        assert error_fields == {"download-limiter", "language"}

    def test_download_limiter_wrong_type_string(self) -> None:
        """Test that non-numeric string download_limiter raises ValidationError."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": "unlimited",  # Should be int, not string
            "language": "en",
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "download-limiter" for error in errors)

    def test_download_limiter_wrong_type_float(self) -> None:
        """Test that float download_limiter raises ValidationError."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20.5,  # Should be int, not float
            "language": "en",
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "download-limiter" for error in errors)

    def test_download_limiter_negative_other_than_minus_one(self) -> None:
        """Test that negative download_limiter values other than -1 are valid (no restriction)."""
        # Note: The spec only defines behavior for 0, -1, and >0
        # But technically any integer is valid from a type perspective
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": -5,
            "language": "en",
        }
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.download_limiter == -5

    def test_download_limiter_with_hyphen_alias(self) -> None:
        """Test that download-limiter (with hyphen) is accepted via alias."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,  # Using hyphen as in actual YAML
            "language": "en",
        }
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.download_limiter == 20


class TestChannelConfigLanguage:
    """Test cases for the language field validation."""

    def test_language_with_single_valid_code(self) -> None:
        """Test that single valid language code is accepted."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            "language": "en",
        }
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.language == "en"

    def test_language_rejects_invalid_code(self) -> None:
        """Test that invalid language code raises ValidationError."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            "language": "xyz",
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        error = str(exc_info.value)
        assert "Invalid language code" in error
        assert "Supported codes:" in error

    def test_language_rejects_another_invalid_code(self) -> None:
        """Test that invalid language code raises ValidationError."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            "language": "invalid",
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        error = str(exc_info.value)
        assert "Invalid language code" in error

    def test_language_field_required(self) -> None:
        """Test that language field is required."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            # Missing language
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "language" for error in errors)

    def test_language_rejects_empty_string(self) -> None:
        """Test that empty language string raises ValidationError."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            "language": "",
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        error = str(exc_info.value)
        # Empty string should fail validation
        assert "language" in error.lower()

    def test_language_rejects_list_type(self) -> None:
        """Test that passing a list for language field raises ValidationError (breaking change from old schema)."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            "language": ["en"],  # Old schema used list, new schema requires string
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "language" for error in errors)
        assert any("str" in str(error["type"]) or "string" in str(error) for error in errors)

    def test_language_must_be_string_not_int(self) -> None:
        """Test that language field must be a string, not an integer."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            "language": 123,
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "language" for error in errors)

    def test_language_accepts_common_codes(self) -> None:
        """Test that common language codes are accepted."""
        common_languages = ["en", "de", "fr", "es", "ja", "zh", "ko", "ar", "hi", "pt", "it", "ru"]
        for lang_code in common_languages:
            channel_data = {
                "url": "https://www.youtube.com/@test",
                "name": "Test Channel",
                "category": "test_category",
                "description": "Test content",
                "download-limiter": 20,
                "language": lang_code,
            }
            channel = ChannelConfig.model_validate(channel_data)
            assert channel.language == lang_code

    def test_language_german(self) -> None:
        """Test that German language code 'de' is accepted."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            "language": "de",
        }
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.language == "de"

    def test_language_japanese(self) -> None:
        """Test that Japanese language code 'ja' is accepted."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
            "language": "ja",
        }
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.language == "ja"

    def test_config_get_channel_includes_language(self) -> None:
        """Test that Config.get_channel() returns channel with language field."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test_category",
                    "description": "Test content",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            channel = config.get_channel(0)
            assert channel.language == "en"
        finally:
            temp_path.unlink()

    def test_config_getter_methods(self) -> None:
        """Test all config getter methods return expected values."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
            "defaults": {
                "encoding_name": "o200k_base",
                "repetition_min_k": 1,
                "repetition_min_repetitions": 5,
                "detect_min_k": 3,
            },
            "hallucination_detection": {
                "coef_repetitions": 0.1,
                "coef_sequence_length": 0.2,
                "intercept": -0.5,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)

            # Test default getters
            assert config.getEncodingName() == "o200k_base"
            assert config.getRepetitionMinK() == 1
            assert config.getRepetitionMinRepetitions() == 5
            assert config.getDetectMinK() == 3

            # Test hallucination classifier model getter
            coefs = config.getHallucinationClassifierModel()
            assert coefs == (0.1, 0.2, -0.5)

            # Test config path getter
            assert config.getConfigPath() == temp_path

            # Test all path getters
            assert config.getDataDir() == Path("./data/")
            assert config.getDataDownloadsDir() == Path("./data/downloads")
            assert config.getDataDownloadsVideosDir() == Path("./data/downloads/videos/")
            assert config.getDataDownloadsTranscriptsDir() == Path("./data/downloads/transcripts")
            assert config.getDataDownloadsTranscriptsHallucinationsDir() == Path("./data/downloads/transcripts-hallucinations")
            assert config.getDataDownloadsTranscriptsCleanedDir() == Path("./data/downloads/transcripts_cleaned")
            assert config.getDataDownloadsAudioDir() == Path("./data/downloads/audio")
            assert config.getDataDownloadsMetadataDir() == Path("./data/downloads/metadata")
            assert config.getDataOutputDir() == Path("./data/output/")
            assert config.getDataInputDir() == Path("./data/input/")
            assert config.getDataTempDir() == Path("./data/temp")
            assert config.getDataArchiveDir() == Path("./data/archive")
            assert config.getDataArchiveVideosDir() == Path("./data/archive/videos")
        finally:
            temp_path.unlink()

    def test_config_get_channels_all_have_language(self) -> None:
        """Test that Config.get_channels() returns all channels with language."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test1",
                    "name": "Test Channel 1",
                    "category": "test_category",
                    "description": "Test content",
                    "download-limiter": 20,
                    "language": "en",
                },
                {
                    "url": "https://www.youtube.com/@test2",
                    "name": "Test Channel 2",
                    "category": "test_category",
                    "description": "Test content",
                    "download-limiter": 20,
                    "language": "ja",
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            channels = config.get_channels()
            assert len(channels) == 2
            assert all(hasattr(ch, "language") for ch in channels)
            assert all(ch.language for ch in channels)
            assert channels[0].language == "en"
            assert channels[1].language == "ja"
        finally:
            temp_path.unlink()

    def test_config_get_topic_segmentation_when_missing(self) -> None:
        """Test that get_topic_segmentation_config raises KeyError when not configured."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
            # No topic_segmentation section
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            with pytest.raises(KeyError) as exc_info:
                config.get_topic_segmentation_config()
            assert "topic_segmentation" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_config_get_article_compiler_when_missing(self) -> None:
        """Test that get_article_compiler_config raises KeyError when not configured."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
            # No article_compiler section
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = Config(temp_path)
            with pytest.raises(KeyError) as exc_info:
                config.get_article_compiler_config()
            assert "article_compiler" in str(exc_info.value)
        finally:
            temp_path.unlink()
