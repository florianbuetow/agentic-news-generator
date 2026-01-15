"""Tests for topic segmentation configuration models."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import Config, LLMConfig, TopicSegmentationConfig


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
        "reports_dir": "reports",
    }


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_valid_llm_config(self) -> None:
        """Test valid LLM configuration."""
        config = LLMConfig(
            model="qwen3-30b-a3b-thinking-2507-mlx@8bit",
            api_base="http://127.0.0.1:1234/v1",
            api_key="LMSTUDIO_API_KEY",
            context_window=262144,
            max_tokens=32000,
            temperature=0.7,
            context_window_threshold=90,
        )
        assert config.model == "qwen3-30b-a3b-thinking-2507-mlx@8bit"
        assert config.api_base == "http://127.0.0.1:1234/v1"
        assert config.api_key == "LMSTUDIO_API_KEY"
        assert config.context_window == 262144
        assert config.max_tokens == 32000
        assert config.temperature == 0.7
        assert config.context_window_threshold == 90

    def test_valid_llm_config_with_none_api_base(self) -> None:
        """Test valid LLM configuration with None api_base."""
        config = LLMConfig(
            model="anthropic/claude-3-5-sonnet-20241022",
            api_base=None,
            api_key="ANTHROPIC_API_KEY",
            context_window=200000,
            max_tokens=4096,
            temperature=0.5,
            context_window_threshold=85,
        )
        assert config.model == "anthropic/claude-3-5-sonnet-20241022"
        assert config.api_base is None
        assert config.api_key == "ANTHROPIC_API_KEY"
        assert config.context_window_threshold == 85

    def test_missing_model_field(self) -> None:
        """Test LLM config with missing model field."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 100000,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("model",) and err["type"] == "missing" for err in errors)

    def test_missing_api_key_env_field(self) -> None:
        """Test LLM config with missing api_key_env field."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "context_window": 100000,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("api_key",) and err["type"] == "missing" for err in errors)

    def test_missing_context_window_field(self) -> None:
        """Test LLM config with missing context_window field."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "max_tokens": 2048,
                    "temperature": 0.7,
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("context_window",) and err["type"] == "missing" for err in errors)

    def test_missing_max_tokens_field(self) -> None:
        """Test LLM config with missing max_tokens field."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 100000,
                    "temperature": 0.7,
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("max_tokens",) and err["type"] == "missing" for err in errors)

    def test_missing_temperature_field(self) -> None:
        """Test LLM config with missing temperature field."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 100000,
                    "max_tokens": 2048,
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("temperature",) and err["type"] == "missing" for err in errors)

    def test_missing_api_base_field(self) -> None:
        """Test LLM config with missing api_base field."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_key": "API_KEY",
                    "context_window": 100000,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("api_base",) and err["type"] == "missing" for err in errors)

    def test_wrong_type_for_context_window(self) -> None:
        """Test LLM config with wrong type for context_window."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": "not_an_int",
                    "max_tokens": 2048,
                    "temperature": 0.7,
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("context_window",) for err in errors)

    def test_wrong_type_for_max_tokens(self) -> None:
        """Test LLM config with wrong type for max_tokens."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 100000,
                    "max_tokens": "not_an_int",
                    "temperature": 0.7,
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("max_tokens",) for err in errors)

    def test_wrong_type_for_temperature(self) -> None:
        """Test LLM config with wrong type for temperature."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 100000,
                    "max_tokens": 2048,
                    "temperature": "not_a_float",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("temperature",) for err in errors)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden in LLM config."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 100000,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "extra_field": "not_allowed",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["type"] == "extra_forbidden" for err in errors)

    def test_frozen_model(self) -> None:
        """Test that LLM config is frozen."""
        config = LLMConfig(
            model="test-model",
            api_base="http://localhost:1234/v1",
            api_key="API_KEY",
            context_window=100000,
            max_tokens=2048,
            temperature=0.7,
            context_window_threshold=90,
        )
        with pytest.raises(ValidationError):
            config.temperature = 0.5

    def test_missing_context_window_threshold_field(self) -> None:
        """Test LLM config with missing context_window_threshold field."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 100000,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("context_window_threshold",) and err["type"] == "missing" for err in errors)

    def test_context_window_threshold_below_zero(self) -> None:
        """Test LLM config with context_window_threshold below 0."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 100000,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "context_window_threshold": -1,
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("context_window_threshold",) and err["type"] == "greater_than_equal" for err in errors)

    def test_context_window_threshold_above_100(self) -> None:
        """Test LLM config with context_window_threshold above 100."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(
                {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 100000,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "context_window_threshold": 101,
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("context_window_threshold",) and err["type"] == "less_than_equal" for err in errors)

    def test_context_window_threshold_boundary_values(self) -> None:
        """Test LLM config with context_window_threshold at boundary values (0 and 100)."""
        # Test 0
        config_0 = LLMConfig(
            model="test-model",
            api_base="http://localhost:1234/v1",
            api_key="API_KEY",
            context_window=100000,
            max_tokens=2048,
            temperature=0.7,
            context_window_threshold=0,
        )
        assert config_0.context_window_threshold == 0

        # Test 100
        config_100 = LLMConfig(
            model="test-model",
            api_base="http://localhost:1234/v1",
            api_key="API_KEY",
            context_window=100000,
            max_tokens=2048,
            temperature=0.7,
            context_window_threshold=100,
        )
        assert config_100.context_window_threshold == 100


class TestTopicSegmentationConfig:
    """Tests for TopicSegmentationConfig model."""

    def test_valid_topic_segmentation_config(self) -> None:
        """Test valid topic segmentation configuration."""
        agent_llm = LLMConfig(
            model="test-agent-model",
            api_base="http://localhost:1234/v1",
            api_key="AGENT_KEY",
            context_window=262144,
            max_tokens=32000,
            temperature=0.7,
            context_window_threshold=90,
        )
        critic_llm = LLMConfig(
            model="test-critic-model",
            api_base="http://localhost:1235/v1",
            api_key="CRITIC_KEY",
            context_window=262144,
            max_tokens=32000,
            temperature=0.3,
            context_window_threshold=90,
        )
        config = TopicSegmentationConfig(
            agent_llm=agent_llm,
            critic_llm=critic_llm,
            retry_limit=3,
        )
        assert config.agent_llm.model == "test-agent-model"
        assert config.critic_llm.model == "test-critic-model"
        assert config.retry_limit == 3

    def test_missing_agent_llm_field(self) -> None:
        """Test topic segmentation config with missing agent_llm field."""
        critic_llm_dict = {
            "model": "test-critic-model",
            "api_base": "http://localhost:1235/v1",
            "api_key": "CRITIC_KEY",
            "context_window": 262144,
            "max_tokens": 32000,
            "temperature": 0.3,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            TopicSegmentationConfig.model_validate({"critic_llm": critic_llm_dict, "retry_limit": 3})
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("agent_llm",) and err["type"] == "missing" for err in errors)

    def test_missing_critic_llm_field(self) -> None:
        """Test topic segmentation config with missing critic_llm field."""
        agent_llm_dict = {
            "model": "test-agent-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "AGENT_KEY",
            "context_window": 262144,
            "max_tokens": 32000,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            TopicSegmentationConfig.model_validate({"agent_llm": agent_llm_dict, "retry_limit": 3})
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("critic_llm",) and err["type"] == "missing" for err in errors)

    def test_missing_retry_limit_field(self) -> None:
        """Test topic segmentation config with missing retry_limit field."""
        agent_llm_dict = {
            "model": "test-agent-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "AGENT_KEY",
            "context_window": 262144,
            "max_tokens": 32000,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        critic_llm_dict = {
            "model": "test-critic-model",
            "api_base": "http://localhost:1235/v1",
            "api_key": "CRITIC_KEY",
            "context_window": 262144,
            "max_tokens": 32000,
            "temperature": 0.3,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            TopicSegmentationConfig.model_validate({"agent_llm": agent_llm_dict, "critic_llm": critic_llm_dict})
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("retry_limit",) and err["type"] == "missing" for err in errors)

    def test_wrong_type_for_retry_limit(self) -> None:
        """Test topic segmentation config with wrong type for retry_limit."""
        agent_llm_dict = {
            "model": "test-agent-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "AGENT_KEY",
            "context_window": 262144,
            "max_tokens": 32000,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        critic_llm_dict = {
            "model": "test-critic-model",
            "api_base": "http://localhost:1235/v1",
            "api_key": "CRITIC_KEY",
            "context_window": 262144,
            "max_tokens": 32000,
            "temperature": 0.3,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            TopicSegmentationConfig.model_validate(
                {"agent_llm": agent_llm_dict, "critic_llm": critic_llm_dict, "retry_limit": "not_an_int"}
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("retry_limit",) for err in errors)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden in topic segmentation config."""
        agent_llm_dict = {
            "model": "test-agent-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "AGENT_KEY",
            "context_window": 262144,
            "max_tokens": 32000,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        critic_llm_dict = {
            "model": "test-critic-model",
            "api_base": "http://localhost:1235/v1",
            "api_key": "CRITIC_KEY",
            "context_window": 262144,
            "max_tokens": 32000,
            "temperature": 0.3,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            TopicSegmentationConfig.model_validate(
                {
                    "agent_llm": agent_llm_dict,
                    "critic_llm": critic_llm_dict,
                    "retry_limit": 3,
                    "extra_field": "not_allowed",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["type"] == "extra_forbidden" for err in errors)

    def test_frozen_model(self) -> None:
        """Test that topic segmentation config is frozen."""
        agent_llm = LLMConfig(
            model="test-agent-model",
            api_base="http://localhost:1234/v1",
            api_key="AGENT_KEY",
            context_window=262144,
            max_tokens=32000,
            temperature=0.7,
            context_window_threshold=90,
        )
        critic_llm = LLMConfig(
            model="test-critic-model",
            api_base="http://localhost:1235/v1",
            api_key="CRITIC_KEY",
            context_window=262144,
            max_tokens=32000,
            temperature=0.3,
            context_window_threshold=90,
        )
        config = TopicSegmentationConfig(
            agent_llm=agent_llm,
            critic_llm=critic_llm,
            retry_limit=3,
        )
        with pytest.raises(ValidationError):
            config.retry_limit = 5


class TestConfigWithTopicSegmentation:
    """Tests for Config class with topic segmentation."""

    def test_load_config_with_topic_segmentation(self) -> None:
        """Test loading config with topic_segmentation section."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test description",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
            "topic_segmentation": {
                "agent_llm": {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 262144,
                    "max_tokens": 32000,
                    "temperature": 0.7,
                    "context_window_threshold": 90,
                },
                "critic_llm": {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 262144,
                    "max_tokens": 32000,
                    "temperature": 0.3,
                    "context_window_threshold": 90,
                },
                "retry_limit": 3,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            ts_config = config.get_topic_segmentation_config()
            assert ts_config.agent_llm.model == "test-model"
            assert ts_config.critic_llm.temperature == 0.3
            assert ts_config.retry_limit == 3
        finally:
            Path(temp_path).unlink()

    def test_load_config_without_topic_segmentation(self) -> None:
        """Test loading config without topic_segmentation section."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test description",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            with pytest.raises(KeyError) as exc_info:
                config.get_topic_segmentation_config()
            assert "topic_segmentation" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_invalid_topic_segmentation_config(self) -> None:
        """Test loading config with invalid topic_segmentation section."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test description",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
            "topic_segmentation": {
                "agent_llm": {
                    "model": "test-model",
                    # Missing required fields
                },
                "critic_llm": {
                    "model": "test-model",
                },
                "retry_limit": 3,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            assert "Topic segmentation configuration validation failed" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_topic_segmentation_with_missing_retry_limit(self) -> None:
        """Test topic_segmentation config with missing retry_limit."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test description",
                    "download-limiter": 20,
                    "language": "en",
                }
            ],
            "topic_segmentation": {
                "agent_llm": {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 262144,
                    "max_tokens": 32000,
                    "temperature": 0.7,
                    "context_window_threshold": 90,
                },
                "critic_llm": {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 262144,
                    "max_tokens": 32000,
                    "temperature": 0.3,
                    "context_window_threshold": 90,
                },
                # Missing retry_limit
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            assert "retry_limit" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()
