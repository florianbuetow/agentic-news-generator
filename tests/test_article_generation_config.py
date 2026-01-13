"""Tests for article generation configuration models."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import ArticleGenerationConfig, Config, LLMConfig


def get_valid_paths_config() -> dict[str, str]:
    """Return a valid paths configuration dictionary for tests."""
    return {
        "data_dir": "./data/",
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
    }


class TestArticleGenerationConfig:
    """Tests for ArticleGenerationConfig model."""

    def test_valid_article_generation_config(self) -> None:
        """Test valid article generation configuration."""
        writer_llm = LLMConfig(
            model="openai/mistralai/Mistral-7B-Instruct-v0.3",
            api_base="http://127.0.0.1:1234/v1",
            api_key="LMSTUDIO_API_KEY",
            context_window=32768,
            max_tokens=4096,
            temperature=0.7,
            context_window_threshold=90,
        )
        config = ArticleGenerationConfig(
            writer_llm=writer_llm,
            max_retries=3,
            timeout_seconds=30,
            allowed_styles=["NATURE_NEWS", "SCIAM_MAGAZINE"],
            default_style_mode="SCIAM_MAGAZINE",
            default_target_length_words="900-1200",
        )
        assert config.writer_llm.model == "openai/mistralai/Mistral-7B-Instruct-v0.3"
        assert config.max_retries == 3
        assert config.allowed_styles == ["NATURE_NEWS", "SCIAM_MAGAZINE"]
        assert config.default_style_mode == "SCIAM_MAGAZINE"
        assert config.default_target_length_words == "900-1200"

    def test_missing_writer_llm_field(self) -> None:
        """Test article generation config with missing writer_llm field."""
        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(
                {
                    "max_retries": 3,
                    "allowed_styles": ["NATURE_NEWS", "SCIAM_MAGAZINE"],
                    "default_style_mode": "NATURE_NEWS",
                    "default_target_length_words": "900-1200",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("writer_llm",) and err["type"] == "missing" for err in errors)

    def test_missing_max_retries_field(self) -> None:
        """Test article generation config with missing max_retries field."""
        writer_llm_dict = {
            "model": "test-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "API_KEY",
            "context_window": 32768,
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(
                {
                    "writer_llm": writer_llm_dict,
                    "allowed_styles": ["NATURE_NEWS"],
                    "default_style_mode": "NATURE_NEWS",
                    "default_target_length_words": "900-1200",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("max_retries",) and err["type"] == "missing" for err in errors)

    def test_missing_allowed_styles_field(self) -> None:
        """Test article generation config with missing allowed_styles field."""
        writer_llm_dict = {
            "model": "test-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "API_KEY",
            "context_window": 32768,
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(
                {
                    "writer_llm": writer_llm_dict,
                    "max_retries": 3,
                    "default_style_mode": "NATURE_NEWS",
                    "default_target_length_words": "900-1200",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("allowed_styles",) and err["type"] == "missing" for err in errors)

    def test_missing_default_style_mode_field(self) -> None:
        """Test article generation config with missing default_style_mode field."""
        writer_llm_dict = {
            "model": "test-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "API_KEY",
            "context_window": 32768,
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(
                {
                    "writer_llm": writer_llm_dict,
                    "max_retries": 3,
                    "allowed_styles": ["NATURE_NEWS"],
                    "default_target_length_words": "900-1200",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("default_style_mode",) and err["type"] == "missing" for err in errors)

    def test_missing_default_target_length_words_field(self) -> None:
        """Test article generation config with missing default_target_length_words field."""
        writer_llm_dict = {
            "model": "test-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "API_KEY",
            "context_window": 32768,
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(
                {
                    "writer_llm": writer_llm_dict,
                    "max_retries": 3,
                    "allowed_styles": ["NATURE_NEWS"],
                    "default_style_mode": "NATURE_NEWS",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("default_target_length_words",) and err["type"] == "missing" for err in errors)

    def test_wrong_type_for_max_retries(self) -> None:
        """Test article generation config with wrong type for max_retries."""
        writer_llm_dict = {
            "model": "test-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "API_KEY",
            "context_window": 32768,
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(
                {
                    "writer_llm": writer_llm_dict,
                    "max_retries": "not_an_int",
                    "allowed_styles": ["NATURE_NEWS"],
                    "default_style_mode": "NATURE_NEWS",
                    "default_target_length_words": "900-1200",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("max_retries",) for err in errors)

    def test_max_retries_below_zero(self) -> None:
        """Test article generation config with max_retries below 0."""
        writer_llm_dict = {
            "model": "test-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "API_KEY",
            "context_window": 32768,
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(
                {
                    "writer_llm": writer_llm_dict,
                    "max_retries": -1,
                    "allowed_styles": ["NATURE_NEWS"],
                    "default_style_mode": "NATURE_NEWS",
                    "default_target_length_words": "900-1200",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("max_retries",) and err["type"] == "greater_than_equal" for err in errors)

    def test_max_retries_above_10(self) -> None:
        """Test article generation config with max_retries above 10."""
        writer_llm_dict = {
            "model": "test-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "API_KEY",
            "context_window": 32768,
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(
                {
                    "writer_llm": writer_llm_dict,
                    "max_retries": 11,
                    "allowed_styles": ["NATURE_NEWS"],
                    "default_style_mode": "NATURE_NEWS",
                    "default_target_length_words": "900-1200",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("max_retries",) and err["type"] == "less_than_equal" for err in errors)

    def test_max_retries_boundary_values(self) -> None:
        """Test article generation config with max_retries at boundary values (0 and 10)."""
        writer_llm = LLMConfig(
            model="test-model",
            api_base="http://localhost:1234/v1",
            api_key="API_KEY",
            context_window=32768,
            max_tokens=4096,
            temperature=0.7,
            context_window_threshold=90,
        )

        # Test 0
        config_0 = ArticleGenerationConfig(
            writer_llm=writer_llm,
            max_retries=0,
            timeout_seconds=30,
            allowed_styles=["NATURE_NEWS"],
            default_style_mode="NATURE_NEWS",
            default_target_length_words="900-1200",
        )
        assert config_0.max_retries == 0

        # Test 10
        config_10 = ArticleGenerationConfig(
            writer_llm=writer_llm,
            max_retries=10,
            timeout_seconds=30,
            allowed_styles=["NATURE_NEWS"],
            default_style_mode="NATURE_NEWS",
            default_target_length_words="900-1200",
        )
        assert config_10.max_retries == 10

    def test_empty_allowed_styles_list(self) -> None:
        """Test article generation config with empty allowed_styles list."""
        writer_llm_dict = {
            "model": "test-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "API_KEY",
            "context_window": 32768,
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(
                {
                    "writer_llm": writer_llm_dict,
                    "max_retries": 3,
                    "allowed_styles": [],
                    "default_style_mode": "NATURE_NEWS",
                    "default_target_length_words": "900-1200",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("allowed_styles",) for err in errors)

    def test_allowed_styles_wrong_type(self) -> None:
        """Test article generation config with wrong type for allowed_styles."""
        writer_llm_dict = {
            "model": "test-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "API_KEY",
            "context_window": 32768,
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(
                {
                    "writer_llm": writer_llm_dict,
                    "max_retries": 3,
                    "allowed_styles": "NATURE_NEWS",
                    "default_style_mode": "NATURE_NEWS",
                    "default_target_length_words": "900-1200",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("allowed_styles",) for err in errors)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden in article generation config."""
        writer_llm_dict = {
            "model": "test-model",
            "api_base": "http://localhost:1234/v1",
            "api_key": "API_KEY",
            "context_window": 32768,
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window_threshold": 90,
        }
        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(
                {
                    "writer_llm": writer_llm_dict,
                    "max_retries": 3,
                    "allowed_styles": ["NATURE_NEWS"],
                    "default_style_mode": "NATURE_NEWS",
                    "default_target_length_words": "900-1200",
                    "extra_field": "not_allowed",
                }
            )
        errors = exc_info.value.errors()
        assert any(err["type"] == "extra_forbidden" for err in errors)

    def test_frozen_model(self) -> None:
        """Test that article generation config is frozen."""
        writer_llm = LLMConfig(
            model="test-model",
            api_base="http://localhost:1234/v1",
            api_key="API_KEY",
            context_window=32768,
            max_tokens=4096,
            temperature=0.7,
            context_window_threshold=90,
        )
        config = ArticleGenerationConfig(
            writer_llm=writer_llm,
            max_retries=3,
            timeout_seconds=30,
            allowed_styles=["NATURE_NEWS"],
            default_style_mode="NATURE_NEWS",
            default_target_length_words="900-1200",
        )
        with pytest.raises(ValidationError):
            config.max_retries = 5


class TestConfigWithArticleGeneration:
    """Tests for Config class with article generation."""

    def test_load_config_with_article_generation(self) -> None:
        """Test loading config with article_generation section."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test description",
                    "download-limiter": 20,
                }
            ],
            "article_generation": {
                "writer_llm": {
                    "model": "openai/mistralai/Mistral-7B-Instruct-v0.3",
                    "api_base": "http://127.0.0.1:1234/v1",
                    "api_key": "LMSTUDIO_API_KEY",
                    "context_window": 32768,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "context_window_threshold": 90,
                },
                "max_retries": 3,
                "timeout_seconds": 600,
                "allowed_styles": ["NATURE_NEWS", "SCIAM_MAGAZINE"],
                "default_style_mode": "SCIAM_MAGAZINE",
                "default_target_length_words": "900-1200",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            ag_config = config.get_article_generation_config()
            assert ag_config.writer_llm.model == "openai/mistralai/Mistral-7B-Instruct-v0.3"
            assert ag_config.max_retries == 3
            assert ag_config.allowed_styles == ["NATURE_NEWS", "SCIAM_MAGAZINE"]
            assert ag_config.default_style_mode == "SCIAM_MAGAZINE"
            assert ag_config.default_target_length_words == "900-1200"
        finally:
            Path(temp_path).unlink()

    def test_load_config_without_article_generation(self) -> None:
        """Test loading config without article_generation section."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test description",
                    "download-limiter": 20,
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            with pytest.raises(KeyError) as exc_info:
                config.get_article_generation_config()
            assert "article_generation" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_invalid_article_generation_config(self) -> None:
        """Test loading config with invalid article_generation section."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test description",
                    "download-limiter": 20,
                }
            ],
            "article_generation": {
                "writer_llm": {
                    "model": "test-model",
                    # Missing required fields
                },
                "max_retries": 3,
                "allowed_styles": ["NATURE_NEWS"],
                "default_style_mode": "NATURE_NEWS",
                "default_target_length_words": "900-1200",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            assert "Article generation configuration validation failed" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_article_generation_with_missing_max_retries(self) -> None:
        """Test article_generation config with missing max_retries."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test description",
                    "download-limiter": 20,
                }
            ],
            "article_generation": {
                "writer_llm": {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 32768,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "context_window_threshold": 90,
                },
                "allowed_styles": ["NATURE_NEWS"],
                "default_style_mode": "NATURE_NEWS",
                "default_target_length_words": "900-1200",
                # Missing max_retries
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                Config(temp_path)
            assert "max_retries" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()

    def test_get_allowed_article_styles(self) -> None:
        """Test get_allowed_article_styles() getter method."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test description",
                    "download-limiter": 20,
                }
            ],
            "article_generation": {
                "writer_llm": {
                    "model": "test-model",
                    "api_base": "http://localhost:1234/v1",
                    "api_key": "API_KEY",
                    "context_window": 32768,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "context_window_threshold": 90,
                },
                "max_retries": 3,
                "timeout_seconds": 30,
                "allowed_styles": ["NATURE_NEWS", "SCIAM_MAGAZINE", "CUSTOM_STYLE"],
                "default_style_mode": "NATURE_NEWS",
                "default_target_length_words": "900-1200",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            allowed_styles = config.get_allowed_article_styles()
            assert allowed_styles == ["NATURE_NEWS", "SCIAM_MAGAZINE", "CUSTOM_STYLE"]
            assert isinstance(allowed_styles, list)
            assert len(allowed_styles) == 3
        finally:
            Path(temp_path).unlink()

    def test_get_allowed_article_styles_without_config(self) -> None:
        """Test get_allowed_article_styles() raises error without article_generation config."""
        config_data = {
            "paths": get_valid_paths_config(),
            "channels": [
                {
                    "url": "https://www.youtube.com/@test",
                    "name": "Test Channel",
                    "category": "test",
                    "description": "Test description",
                    "download-limiter": 20,
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            with pytest.raises(KeyError) as exc_info:
                config.get_allowed_article_styles()
            assert "article_generation" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()
