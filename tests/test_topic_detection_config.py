"""Tests for topic detection configuration models."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import (
    Config,
    TopicDetectionConfig,
    TopicDetectionEmbeddingConfig,
    TopicDetectionSlidingWindowConfig,
)


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


def get_valid_channels_config() -> list[dict[str, str | int]]:
    """Return a valid channels configuration list for tests."""
    return [
        {
            "url": "https://www.youtube.com/@test",
            "name": "Test Channel",
            "category": "test",
            "description": "Test channel description",
            "download-limiter": 10,
            "language": "en",
        }
    ]


def get_valid_topic_detection_config() -> dict[str, object]:
    """Return a valid topic detection configuration dictionary for tests."""
    return {
        "embedding": {
            "provider": "lmstudio",
            "model_name": "multilingual-e5-base-mlx",
            "api_base": "http://127.0.0.1:1234/v1",
            "api_key": "LMSTUDIO_API_KEY",
        },
        "sliding_window": {
            "window_size": 50,
            "stride": 25,
            "threshold_method": "relative",
            "threshold_value": 0.4,
            "smoothing_passes": 1,
        },
        "topic_detection_llm": {
            "model": "openai/test-model",
            "api_base": "http://127.0.0.1:1234/v1",
            "api_key": "LMSTUDIO_API_KEY",
            "context_window": 262144,
            "max_tokens": 4096,
            "temperature": 0.3,
            "context_window_threshold": 90,
            "max_retries": 3,
            "retry_delay": 2.0,
            "timeout_seconds": 30,
        },
        "output_dir": "output/topics",
    }


class TestTopicDetectionEmbeddingConfig:
    """Tests for TopicDetectionEmbeddingConfig model."""

    def test_valid_embedding_config(self) -> None:
        """Test valid embedding configuration."""
        config = TopicDetectionEmbeddingConfig(
            provider="lmstudio",
            model_name="multilingual-e5-base-mlx",
            api_base="http://127.0.0.1:1234/v1",
            api_key="LMSTUDIO_API_KEY",
        )
        assert config.provider == "lmstudio"
        assert config.model_name == "multilingual-e5-base-mlx"
        assert config.api_base == "http://127.0.0.1:1234/v1"
        assert config.api_key == "LMSTUDIO_API_KEY"

    def test_embedding_config_with_none_api_key(self) -> None:
        """Test embedding config with None api_key."""
        config = TopicDetectionEmbeddingConfig(
            provider="lmstudio",
            model_name="test-model",
            api_base="http://localhost:1234/v1",
            api_key=None,
        )
        assert config.api_key is None

    def test_missing_provider_field(self) -> None:
        """Test embedding config with missing provider field."""
        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionEmbeddingConfig.model_validate(
                {
                    "model_name": "test-model",
                    "api_base": "http://localhost:1234/v1",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("provider",) and err["type"] == "missing" for err in errors)

    def test_missing_model_name_field(self) -> None:
        """Test embedding config with missing model_name field."""
        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionEmbeddingConfig.model_validate(
                {
                    "provider": "lmstudio",
                    "api_base": "http://localhost:1234/v1",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("model_name",) and err["type"] == "missing" for err in errors)


class TestTopicDetectionSlidingWindowConfig:
    """Tests for TopicDetectionSlidingWindowConfig model."""

    def test_valid_sliding_window_config(self) -> None:
        """Test valid sliding window configuration."""
        config = TopicDetectionSlidingWindowConfig(
            window_size=50,
            stride=25,
            threshold_method="relative",
            threshold_value=0.4,
            smoothing_passes=1,
        )
        assert config.window_size == 50
        assert config.stride == 25
        assert config.threshold_method == "relative"
        assert config.threshold_value == 0.4
        assert config.smoothing_passes == 1

    def test_window_size_must_be_positive(self) -> None:
        """Test that window_size must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionSlidingWindowConfig(
                window_size=0,
                stride=25,
                threshold_method="relative",
                threshold_value=0.4,
                smoothing_passes=1,
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("window_size",) for err in errors)

    def test_stride_must_be_positive(self) -> None:
        """Test that stride must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionSlidingWindowConfig(
                window_size=50,
                stride=0,
                threshold_method="relative",
                threshold_value=0.4,
                smoothing_passes=1,
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("stride",) for err in errors)


class TestTopicDetectionConfig:
    """Tests for TopicDetectionConfig model."""

    def test_valid_topic_detection_config(self) -> None:
        """Test valid complete topic detection configuration."""
        config_dict = get_valid_topic_detection_config()
        config = TopicDetectionConfig.model_validate(config_dict)

        assert config.embedding.provider == "lmstudio"
        assert config.sliding_window.window_size == 50
        assert config.topic_detection_llm.model == "openai/test-model"
        assert config.output_dir == "output/topics"

    def test_missing_embedding_section(self) -> None:
        """Test config with missing embedding section."""
        config_dict = get_valid_topic_detection_config()
        del config_dict["embedding"]

        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionConfig.model_validate(config_dict)
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("embedding",) and err["type"] == "missing" for err in errors)

    def test_missing_sliding_window_section(self) -> None:
        """Test config with missing sliding_window section."""
        config_dict = get_valid_topic_detection_config()
        del config_dict["sliding_window"]

        with pytest.raises(ValidationError) as exc_info:
            TopicDetectionConfig.model_validate(config_dict)
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("sliding_window",) and err["type"] == "missing" for err in errors)


class TestConfigTopicDetectionIntegration:
    """Integration tests for Config.get_topic_detection_config."""

    def test_config_get_topic_detection_config(self, tmp_path: Path) -> None:
        """Test loading topic detection config from Config class."""
        config_dict = {
            "channels": get_valid_channels_config(),
            "paths": get_valid_paths_config(),
            "topic_detection": get_valid_topic_detection_config(),
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = Config(config_file)
        td_config = config.get_topic_detection_config()

        assert td_config.embedding.provider == "lmstudio"
        assert td_config.sliding_window.window_size == 50
        assert td_config.topic_detection_llm.model == "openai/test-model"

    def test_config_get_topic_detection_config_when_missing(self, tmp_path: Path) -> None:
        """Test that get_topic_detection_config raises KeyError when section is missing."""
        config_dict = {
            "channels": get_valid_channels_config(),
            "paths": get_valid_paths_config(),
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = Config(config_file)

        with pytest.raises(KeyError, match="topic_detection"):
            config.get_topic_detection_config()

    def test_config_validates_topic_detection_on_load(self, tmp_path: Path) -> None:
        """Test that invalid topic_detection config raises error on Config load."""
        config_dict = {
            "channels": get_valid_channels_config(),
            "paths": get_valid_paths_config(),
            "topic_detection": {
                "embedding": {
                    # Missing required fields
                    "provider": "lmstudio",
                },
                "sliding_window": get_valid_topic_detection_config()["sliding_window"],
                "topic_detection_llm": get_valid_topic_detection_config()["topic_detection_llm"],
                "output_dir": "output/topics",
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        with pytest.raises(ValueError, match="Topic detection configuration validation failed"):
            Config(config_file)
