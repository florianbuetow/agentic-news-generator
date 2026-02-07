"""Tests for multi-agent article-generation configuration."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import ArticleGenerationConfig, Config, LLMConfig


def get_valid_paths_config() -> dict[str, str]:
    """Return valid paths section for config tests."""
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


def get_valid_llm_config() -> dict[str, object]:
    """Return valid LLM config payload."""
    return {
        "model": "test-model",
        "api_base": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
        "context_window": 32768,
        "max_tokens": 2048,
        "temperature": 0.3,
        "context_window_threshold": 90,
        "max_retries": 3,
        "retry_delay": 2.0,
        "timeout_seconds": 60,
    }


def get_valid_article_generation_config_dict() -> dict[str, object]:
    """Return valid article_generation section payload."""
    llm_config = get_valid_llm_config()
    return {
        "editor": {
            "editor_max_rounds": 3,
            "output": {
                "final_articles_dir": "./data/output/articles",
                "run_artifacts_dir": "./data/output/article_editor_runs",
                "save_intermediate_results": True,
            },
            "prompts": {
                "root_dir": "./prompts/article_editor",
                "writer_prompt_file": "writer.md",
                "revision_prompt_file": "revision.md",
                "article_review_prompt_file": "article_review.md",
                "concern_mapping_prompt_file": "concern_mapping.md",
                "specialists_dir": "specialists",
            },
        },
        "agents": {
            "writer_llm": llm_config,
            "article_review_llm": llm_config,
            "concern_mapping_llm": llm_config,
            "specialists": {
                "fact_check_llm": llm_config,
                "evidence_finding_llm": llm_config,
                "opinion_llm": llm_config,
                "attribution_llm": llm_config,
                "style_review_llm": llm_config,
            },
        },
        "knowledge_base": {
            "data_dir": "./data/knowledgebase",
            "index_dir": "./data/knowledgebase_index",
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 50,
            "timeout_seconds": 30,
            "embedding": {
                "provider": "lmstudio",
                "model_name": "text-embedding-bge-large-en-v1.5",
                "api_base": "http://127.0.0.1:1234/v1",
                "api_key": "lm-studio",
                "timeout_seconds": 30,
            },
        },
        "perplexity": {
            "api_base": "https://api.perplexity.ai",
            "api_key": "key",
            "model": "sonar",
            "timeout_seconds": 45,
        },
        "institutional_memory": {
            "data_dir": "./data/institutional_memory",
            "fact_checking_subdir": "fact_checking",
            "evidence_finding_subdir": "evidence_finding",
        },
        "allowed_styles": ["NATURE_NEWS", "SCIAM_MAGAZINE"],
        "default_style_mode": "SCIAM_MAGAZINE",
    }


class TestLLMConfig:
    """Tests for base LLM config shared by article-generation agents."""

    def test_llm_config_requires_timeout(self) -> None:
        """Missing timeout_seconds must fail validation."""
        payload = get_valid_llm_config()
        del payload["timeout_seconds"]

        with pytest.raises(ValidationError) as exc_info:
            LLMConfig.model_validate(payload)

        assert any(error["loc"] == ("timeout_seconds",) for error in exc_info.value.errors())


class TestArticleGenerationConfig:
    """Tests for multi-agent article generation config model."""

    def test_valid_article_generation_config(self) -> None:
        """Full config payload validates successfully."""
        payload = get_valid_article_generation_config_dict()
        config = ArticleGenerationConfig.model_validate(payload)

        assert config.editor.editor_max_rounds == 3
        assert config.agents.writer_llm.timeout_seconds == 60
        assert config.perplexity.model == "sonar"
        assert config.allowed_styles == ["NATURE_NEWS", "SCIAM_MAGAZINE"]

    def test_missing_editor_section_fails(self) -> None:
        """Missing editor section is invalid."""
        payload = get_valid_article_generation_config_dict()
        del payload["editor"]

        with pytest.raises(ValidationError) as exc_info:
            ArticleGenerationConfig.model_validate(payload)

        assert any(error["loc"] == ("editor",) for error in exc_info.value.errors())


class TestConfigIntegration:
    """Tests for Config getters using article_generation section."""

    def test_config_loads_article_generation(self) -> None:
        """Config exposes article generation getters for nested schema."""
        config_data: dict[str, object] = {
            "paths": get_valid_paths_config(),
            "channels": [],
            "defaults": {
                "encoding_name": "o200k_base",
                "repetition_min_k": 1,
                "repetition_min_repetitions": 5,
                "detect_min_k": 3,
            },
            "article_generation": get_valid_article_generation_config_dict(),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
            yaml.dump(config_data, handle)
            temp_path = Path(handle.name)

        try:
            config = Config(temp_path)
            article_config = config.get_article_generation_config()
            assert article_config.editor.editor_max_rounds == 3
            assert config.get_allowed_article_styles() == ["NATURE_NEWS", "SCIAM_MAGAZINE"]
            assert config.get_article_editor_max_rounds() == 3
            assert config.get_article_timeout_seconds() == 60
        finally:
            temp_path.unlink()

    def test_missing_article_generation_raises(self) -> None:
        """Getters raise when article_generation section is absent."""
        config_data: dict[str, object] = {
            "paths": get_valid_paths_config(),
            "channels": [],
            "defaults": {
                "encoding_name": "o200k_base",
                "repetition_min_k": 1,
                "repetition_min_repetitions": 5,
                "detect_min_k": 3,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
            yaml.dump(config_data, handle)
            temp_path = Path(handle.name)

        try:
            config = Config(temp_path)
            with pytest.raises(KeyError):
                config.get_article_generation_config()
            with pytest.raises(KeyError):
                config.get_article_editor_max_rounds()
        finally:
            temp_path.unlink()
