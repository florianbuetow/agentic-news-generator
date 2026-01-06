"""Unit tests for the Config class and ChannelConfig model."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from src.config import ChannelConfig, Config


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
        }
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.url == "https://www.youtube.com/@test"
        assert channel.name == "Test Channel"
        assert channel.category == "test_category"
        assert channel.description == "Test content"
        assert channel.download_limiter == 20

    def test_missing_required_field(self) -> None:
        """Test that missing required fields raise ValidationError."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            # Missing category, description, download_limiter
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert len(errors) == 3  # category, description, download-limiter
        error_fields = {error["loc"][0] for error in errors}
        assert error_fields == {"category", "description", "download-limiter"}

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden due to extra='forbid'."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": 20,
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
            # Missing description, download_limiter
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert len(errors) == 2
        error_fields = {error["loc"][0] for error in errors}
        assert error_fields == {"description", "download-limiter"}

    def test_empty_strings_allowed(self) -> None:
        """Test that empty strings are allowed (validation passes but may not be practical)."""
        channel_data = {
            "url": "",
            "name": "",
            "category": "",
            "description": "",
            "download-limiter": 0,
        }
        # Empty strings are valid str types, so validation passes
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.url == ""
        assert channel.name == ""
        assert channel.download_limiter == 0


class TestConfig:
    """Test cases for the Config class."""

    def test_load_valid_config(self) -> None:
        """Test loading a valid configuration file."""
        config_data = {
            "channels": [
                {
                    "url": "https://www.youtube.com/@test1",
                    "name": "Test Channel 1",
                    "category": "category1",
                    "description": "Content 1",
                    "download-limiter": 20,
                },
                {
                    "url": "https://www.youtube.com/@test2",
                    "name": "Test Channel 2",
                    "category": "category2",
                    "description": "Content 2",
                    "download-limiter": 20,
                },
            ]
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
        config_data = {"channels": "not a list"}
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
        config_data: dict[str, list[dict[str, Any]]] = {"channels": []}
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
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    # Missing category, description, download_limiter
                }
            ]
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
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    # Missing name, category, description, download_limiter
                }
            ]
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
            ]
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
            "channels": [
                {
                    "url": "https://www.youtube.com/@test1",
                    "name": "Test Channel 1",
                    "category": "category1",
                    "description": "Content 1",
                    "download-limiter": 20,
                },
                {
                    "url": "https://www.youtube.com/@test2",
                    "name": "Test Channel 2",
                    "category": "category2",
                    "description": "Content 2",
                    "download-limiter": 20,
                },
            ]
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
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                }
            ]
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
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                }
            ]
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
            "channels": [
                {
                    "url": "https://www.youtube.com/@test1",
                    "name": "Test Channel 1",
                    "category": "category1",
                    "description": "Content 1",
                    "download-limiter": 20,
                },
                {
                    "url": "https://www.youtube.com/@test2",
                    "name": "Test Channel 2",
                    "category": "category2",
                    "description": "Content 2",
                    "download-limiter": 20,
                },
            ]
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
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                }
            ]
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
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                }
            ]
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
            "channels": [
                {
                    "url": 123,  # Should be string
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                }
            ]
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
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                    "extra_field": "not allowed",
                }
            ]
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
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "category",
                    "description": "Content",
                    "download-limiter": 20,
                }
            ]
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
            # Missing download_limiter
        }
        with pytest.raises(ValidationError) as exc_info:
            ChannelConfig.model_validate(channel_data)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"][0] == "download-limiter"
        assert "required" in errors[0]["msg"].lower()

    def test_download_limiter_wrong_type_string(self) -> None:
        """Test that non-numeric string download_limiter raises ValidationError."""
        channel_data = {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test_category",
            "description": "Test content",
            "download-limiter": "unlimited",  # Should be int, not string
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
        }
        channel = ChannelConfig.model_validate(channel_data)
        assert channel.download_limiter == 20
