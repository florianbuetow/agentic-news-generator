"""Tests for config-driven AgentFactory."""

from pathlib import Path

import pytest
import yaml

from src.agents.article_generation.agent import build_chief_editor_orchestrator
from src.agents.article_generation.article_review.mock_agent import MockArticleReviewAgent
from src.agents.article_generation.concern_mapping.mock_agent import MockConcernMappingAgent
from src.agents.article_generation.specialists.attribution.mock_agent import MockAttributionAgent
from src.agents.article_generation.specialists.evidence_finding.mock_agent import MockEvidenceFindingAgent
from src.agents.article_generation.specialists.fact_check.mock_agent import MockFactCheckAgent
from src.agents.article_generation.specialists.opinion.mock_agent import MockOpinionAgent
from src.agents.article_generation.specialists.style_review.mock_agent import MockStyleReviewAgent
from src.agents.article_generation.writer.mock_agent import MockWriterAgent
from src.config import Config


def _write_test_config(
    tmp_dir: Path,
    *,
    agent_overrides: dict[str, str] | None = None,
    specialist_overrides: dict[str, str] | None = None,
) -> Path:
    """Write a minimal config.yaml for factory tests."""
    llm = {
        "model": "test-model",
        "api_base": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
        "context_window": 32768,
        "max_tokens": 2048,
        "temperature": 0.3,
        "context_window_threshold": 90,
        "max_retries": 0,
        "retry_delay": 2.0,
        "timeout_seconds": 60,
    }

    overrides = agent_overrides or {}
    spec_overrides = specialist_overrides or {}

    def agent_slot(impl: str = "default") -> dict[str, object]:
        return {"agent_name": impl, "llm": llm}

    # Create required directories
    for subdir in [
        "knowledgebase",
        "knowledgebase_index",
        "institutional_memory",
        "institutional_memory/fact_checking",
        "institutional_memory/evidence_finding",
        "prompts/article_editor",
        "input/taxonomies/cache",
        "output/articles",
        "output/topics",
        "output/article_editor_runs",
        "articles/input",
    ]:
        (tmp_dir / subdir).mkdir(parents=True, exist_ok=True)

    config_data: dict[str, object] = {
        "paths": {
            "data_dir": str(tmp_dir),
            "data_models_dir": str(tmp_dir / "models"),
            "data_downloads_dir": str(tmp_dir / "downloads"),
            "data_downloads_videos_dir": str(tmp_dir / "downloads" / "videos"),
            "data_downloads_transcripts_dir": str(tmp_dir / "downloads" / "transcripts"),
            "data_downloads_transcripts_hallucinations_dir": str(tmp_dir / "downloads" / "transcripts-hallucinations"),
            "data_downloads_transcripts_cleaned_dir": str(tmp_dir / "downloads" / "transcripts_cleaned"),
            "data_transcripts_topics_dir": str(tmp_dir / "downloads" / "transcripts-topics"),
            "data_downloads_audio_dir": str(tmp_dir / "downloads" / "audio"),
            "data_downloads_metadata_dir": str(tmp_dir / "downloads" / "metadata"),
            "data_output_dir": str(tmp_dir / "output"),
            "data_input_dir": str(tmp_dir / "input"),
            "data_temp_dir": str(tmp_dir / "temp"),
            "data_archive_dir": str(tmp_dir / "archive"),
            "data_archive_videos_dir": str(tmp_dir / "archive" / "videos"),
            "data_logs_dir": str(tmp_dir / "logs"),
            "data_output_articles_dir": str(tmp_dir / "output" / "articles"),
            "data_articles_input_dir": str(tmp_dir / "articles" / "input"),
            "reports_dir": str(tmp_dir / "reports"),
            "data_article_generation_output_dir": str(tmp_dir / "output" / "articles"),
            "data_article_generation_artifacts_dir": str(tmp_dir / "output" / "article_editor_runs"),
            "data_article_generation_kb_dir": str(tmp_dir / "knowledgebase"),
            "data_article_generation_kb_index_dir": str(tmp_dir / "knowledgebase_index"),
            "data_article_generation_institutional_memory_dir": str(tmp_dir / "institutional_memory"),
            "data_article_generation_prompts_dir": str(tmp_dir / "prompts" / "article_editor"),
            "data_topic_detection_output_dir": str(tmp_dir / "output" / "topics"),
            "data_topic_detection_taxonomies_dir": str(tmp_dir / "input" / "taxonomies"),
            "data_topic_detection_taxonomy_cache_dir": str(tmp_dir / "input" / "taxonomies" / "cache"),
            "data_hallucination_detection_output_dir": str(tmp_dir / "downloads" / "transcripts-hallucinations"),
            "data_article_compiler_input_dir": str(tmp_dir / "input" / "newspaper" / "articles"),
            "data_article_compiler_output_file": str(tmp_dir / "input" / "newspaper" / "articles.js"),
        },
        "channels": [],
        "defaults": {
            "encoding_name": "o200k_base",
            "repetition_min_k": 1,
            "repetition_min_repetitions": 5,
            "detect_min_k": 3,
        },
        "article_generation": {
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
                "writer": agent_slot(overrides.get("writer", "default")),
                "article_review": agent_slot(overrides.get("article_review", "default")),
                "concern_mapping": agent_slot(overrides.get("concern_mapping", "default")),
                "specialists": {
                    "fact_check": agent_slot(spec_overrides.get("fact_check", "default")),
                    "evidence_finding": agent_slot(spec_overrides.get("evidence_finding", "default")),
                    "opinion": agent_slot(spec_overrides.get("opinion", "default")),
                    "attribution": agent_slot(spec_overrides.get("attribution", "default")),
                    "style_review": agent_slot(spec_overrides.get("style_review", "default")),
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
                "api_key": "test-key",
                "model": "sonar",
                "timeout_seconds": 45,
            },
            "institutional_memory": {
                "fact_checking_subdir": "fact_checking",
                "evidence_finding_subdir": "evidence_finding",
            },
            "allowed_styles": ["NATURE_NEWS", "SCIAM_MAGAZINE"],
            "default_style_mode": "SCIAM_MAGAZINE",
        },
    }

    config_path = tmp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


class TestAgentFactory:
    """Tests for config-driven agent factory."""

    def test_mock_specialists_selected_by_config(self, tmp_path: Path) -> None:
        """When config says agent_name=mock, factory returns Mock* agents."""
        config_path = _write_test_config(
            tmp_path,
            specialist_overrides={
                "fact_check": "mock",
                "evidence_finding": "mock",
            },
        )
        config = Config(config_path)
        orchestrator = build_chief_editor_orchestrator(config=config)

        assert isinstance(orchestrator._fact_check_agent, MockFactCheckAgent)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(orchestrator._evidence_finding_agent, MockEvidenceFindingAgent)  # pyright: ignore[reportPrivateUsage]

    def test_all_mock_agents_selected_by_config(self, tmp_path: Path) -> None:
        """When all agents are set to mock, factory returns all Mock* agents."""
        config_path = _write_test_config(
            tmp_path,
            agent_overrides={
                "writer": "mock",
                "article_review": "mock",
                "concern_mapping": "mock",
            },
            specialist_overrides={
                "fact_check": "mock",
                "evidence_finding": "mock",
                "opinion": "mock",
                "attribution": "mock",
                "style_review": "mock",
            },
        )
        config = Config(config_path)
        orchestrator = build_chief_editor_orchestrator(config=config)

        assert isinstance(orchestrator._writer_agent, MockWriterAgent)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(orchestrator._article_review_agent, MockArticleReviewAgent)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(orchestrator._concern_mapping_agent, MockConcernMappingAgent)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(orchestrator._fact_check_agent, MockFactCheckAgent)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(orchestrator._evidence_finding_agent, MockEvidenceFindingAgent)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(orchestrator._opinion_agent, MockOpinionAgent)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(orchestrator._attribution_agent, MockAttributionAgent)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(orchestrator._style_review_agent, MockStyleReviewAgent)  # pyright: ignore[reportPrivateUsage]

    def test_unknown_fact_check_agent_name_raises(self, tmp_path: Path) -> None:
        """Unknown agent_name value raises ValueError."""
        config_path = _write_test_config(
            tmp_path,
            specialist_overrides={"fact_check": "unknown_impl"},
        )
        config = Config(config_path)

        with pytest.raises(ValueError, match="Unknown fact_check agent_name: 'unknown_impl'"):
            build_chief_editor_orchestrator(config=config)

    def test_unknown_evidence_finding_agent_name_raises(self, tmp_path: Path) -> None:
        """Unknown agent_name value raises ValueError."""
        config_path = _write_test_config(
            tmp_path,
            specialist_overrides={"evidence_finding": "bogus"},
        )
        config = Config(config_path)

        with pytest.raises(ValueError, match="Unknown evidence_finding agent_name: 'bogus'"):
            build_chief_editor_orchestrator(config=config)

    def test_unknown_writer_agent_name_raises(self, tmp_path: Path) -> None:
        """Unknown writer agent_name raises ValueError."""
        config_path = _write_test_config(
            tmp_path,
            agent_overrides={"writer": "bogus"},
        )
        config = Config(config_path)

        with pytest.raises(ValueError, match="Unknown writer agent_name: 'bogus'"):
            build_chief_editor_orchestrator(config=config)

    def test_unknown_article_review_agent_name_raises(self, tmp_path: Path) -> None:
        """Unknown article_review agent_name raises ValueError."""
        config_path = _write_test_config(
            tmp_path,
            agent_overrides={"article_review": "bogus"},
        )
        config = Config(config_path)

        with pytest.raises(ValueError, match="Unknown article_review agent_name: 'bogus'"):
            build_chief_editor_orchestrator(config=config)

    def test_unknown_concern_mapping_agent_name_raises(self, tmp_path: Path) -> None:
        """Unknown concern_mapping agent_name raises ValueError."""
        config_path = _write_test_config(
            tmp_path,
            agent_overrides={"concern_mapping": "bogus"},
        )
        config = Config(config_path)

        with pytest.raises(ValueError, match="Unknown concern_mapping agent_name: 'bogus'"):
            build_chief_editor_orchestrator(config=config)

    def test_unknown_opinion_agent_name_raises(self, tmp_path: Path) -> None:
        """Unknown opinion agent_name raises ValueError."""
        config_path = _write_test_config(
            tmp_path,
            specialist_overrides={"opinion": "bogus"},
        )
        config = Config(config_path)

        with pytest.raises(ValueError, match="Unknown opinion agent_name: 'bogus'"):
            build_chief_editor_orchestrator(config=config)

    def test_unknown_attribution_agent_name_raises(self, tmp_path: Path) -> None:
        """Unknown attribution agent_name raises ValueError."""
        config_path = _write_test_config(
            tmp_path,
            specialist_overrides={"attribution": "bogus"},
        )
        config = Config(config_path)

        with pytest.raises(ValueError, match="Unknown attribution agent_name: 'bogus'"):
            build_chief_editor_orchestrator(config=config)

    def test_unknown_style_review_agent_name_raises(self, tmp_path: Path) -> None:
        """Unknown style_review agent_name raises ValueError."""
        config_path = _write_test_config(
            tmp_path,
            specialist_overrides={"style_review": "bogus"},
        )
        config = Config(config_path)

        with pytest.raises(ValueError, match="Unknown style_review agent_name: 'bogus'"):
            build_chief_editor_orchestrator(config=config)
