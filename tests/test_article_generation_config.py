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
        "data_dir": "data",
        "data_models_dir": "data/models",
        "data_downloads_dir": "data/downloads",
        "data_downloads_videos_dir": "data/downloads/videos",
        "data_downloads_transcripts_dir": "data/downloads/transcripts",
        "data_downloads_transcripts_hallucinations_dir": "data/downloads/transcripts-hallucinations",
        "data_downloads_transcripts_cleaned_dir": "data/downloads/transcripts_cleaned",
        "data_transcripts_topics_dir": "data/downloads/transcripts-topics",
        "data_downloads_audio_dir": "data/downloads/audio",
        "data_downloads_metadata_dir": "data/downloads/metadata",
        "data_output_dir": "data/output",
        "data_input_dir": "data/input",
        "data_temp_dir": "data/temp",
        "data_archive_dir": "data/archive",
        "data_archive_videos_dir": "data/archive/videos",
        "data_logs_dir": "logs",
        "data_output_articles_dir": "data/output/articles",
        "data_articles_input_dir": "data/articles/input",
        "reports_dir": "reports",
        "data_article_generation_output_dir": "data/output/articles",
        "data_article_generation_artifacts_dir": "data/output/article_editor_runs",
        "data_article_generation_kb_dir": "data/knowledgebase",
        "data_article_generation_kb_index_dir": "data/knowledgebase_index",
        "data_article_generation_institutional_memory_dir": "data/institutional_memory",
        "data_article_generation_prompts_dir": "prompts/article_editor",
        "data_topic_detection_output_dir": "data/output/topics",
        "data_topic_detection_taxonomies_dir": "data/input/taxonomies",
        "data_topic_detection_taxonomy_cache_dir": "data/input/taxonomies/cache",
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


def get_valid_agent_slot(agent_name: str = "default") -> dict[str, object]:
    """Return a valid agent slot config payload."""
    return {
        "agent_name": agent_name,
        "llm": get_valid_llm_config(),
    }


def get_valid_article_generation_config_dict() -> dict[str, object]:
    """Return valid article_generation section payload."""
    return {
        "editor": {
            "editor_max_rounds": 3,
            "prompts": {
                "writer_prompt_file": "writer.md",
                "revision_prompt_file": "revision.md",
                "article_review_prompt_file": "article_review.md",
                "concern_mapping_prompt_file": "concern_mapping.md",
                "specialists_dir": "specialists",
                "fact_check_prompt_file": "fact_check.md",
                "evidence_finding_prompt_file": "evidence_finding.md",
                "opinion_prompt_file": "opinion.md",
                "attribution_prompt_file": "attribution.md",
                "style_review_prompt_file": "style_review.md",
            },
        },
        "agents": {
            "writer": get_valid_agent_slot(),
            "article_review": get_valid_agent_slot(),
            "concern_mapping": get_valid_agent_slot(),
            "specialists": {
                "fact_check": get_valid_agent_slot(),
                "evidence_finding": get_valid_agent_slot(),
                "opinion": get_valid_agent_slot(),
                "attribution": get_valid_agent_slot(),
                "style_review": get_valid_agent_slot(),
            },
        },
        "knowledge_base": {
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
        assert config.agents.writer.llm.timeout_seconds == 60
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


class TestAgentConfigWithImplementation:
    """Tests for agent config with agent_name field."""

    def test_agent_config_accepts_agent_name_field(self) -> None:
        """Agent config block must accept agent_name + llm sub-key."""
        from src.config import AgentSlotConfig

        payload = {
            "agent_name": "default",
            "llm": get_valid_llm_config(),
        }
        slot = AgentSlotConfig.model_validate(payload)
        assert slot.agent_name == "default"
        assert slot.llm.model == "test-model"

    def test_agent_config_agent_name_required(self) -> None:
        """Missing agent_name field must fail."""
        from src.config import AgentSlotConfig

        payload = {"llm": get_valid_llm_config()}
        with pytest.raises(ValidationError):
            AgentSlotConfig.model_validate(payload)
