"""Tests for config-driven AgentFactory."""

from pathlib import Path

import yaml

from src.agents.article_generation.agent import build_chief_editor_orchestrator
from src.agents.article_generation.specialists.evidence_finding.mock_agent import MockEvidenceFindingAgent
from src.agents.article_generation.specialists.fact_check.mock_agent import MockFactCheckAgent
from src.config import Config


def _write_test_config(tmp_dir: Path, specialist_overrides: dict[str, str] | None = None) -> Path:
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

    def agent_slot(impl: str = "default") -> dict:
        return {"implementation": impl, "llm": llm}

    overrides = specialist_overrides or {}

    # Create required directories
    for subdir in ["knowledgebase", "knowledgebase_index", "institutional_memory",
                   "institutional_memory/fact_checking", "institutional_memory/evidence_finding",
                   "output/articles", "output/article_editor_runs", "articles/input"]:
        (tmp_dir / subdir).mkdir(parents=True, exist_ok=True)

    config_data = {
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
                "output": {
                    "final_articles_dir": str(tmp_dir / "output" / "articles"),
                    "run_artifacts_dir": str(tmp_dir / "output" / "article_editor_runs"),
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
                "writer": agent_slot(),
                "article_review": agent_slot(),
                "concern_mapping": agent_slot(),
                "specialists": {
                    "fact_check": agent_slot(overrides.get("fact_check", "default")),
                    "evidence_finding": agent_slot(overrides.get("evidence_finding", "default")),
                    "opinion": agent_slot(overrides.get("opinion", "default")),
                    "attribution": agent_slot(overrides.get("attribution", "default")),
                    "style_review": agent_slot(overrides.get("style_review", "default")),
                },
            },
            "knowledge_base": {
                "data_dir": str(tmp_dir / "knowledgebase"),
                "index_dir": str(tmp_dir / "knowledgebase_index"),
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
                "data_dir": str(tmp_dir / "institutional_memory"),
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
        """When config says implementation=mock, factory returns Mock* agents."""
        config_path = _write_test_config(
            tmp_path,
            specialist_overrides={
                "fact_check": "mock",
                "evidence_finding": "mock",
            },
        )
        config = Config(config_path)
        orchestrator = build_chief_editor_orchestrator(config=config)

        assert isinstance(orchestrator._fact_check_agent, MockFactCheckAgent)
        assert isinstance(orchestrator._evidence_finding_agent, MockEvidenceFindingAgent)

    def test_unknown_fact_check_implementation_raises(self, tmp_path: Path) -> None:
        """Unknown implementation value raises ValueError."""
        config_path = _write_test_config(
            tmp_path,
            specialist_overrides={"fact_check": "unknown_impl"},
        )
        config = Config(config_path)
        import pytest

        with pytest.raises(ValueError, match="Unknown fact_check implementation: 'unknown_impl'"):
            build_chief_editor_orchestrator(config=config)

    def test_unknown_evidence_finding_implementation_raises(self, tmp_path: Path) -> None:
        """Unknown implementation value raises ValueError."""
        config_path = _write_test_config(
            tmp_path,
            specialist_overrides={"evidence_finding": "bogus"},
        )
        config = Config(config_path)
        import pytest

        with pytest.raises(ValueError, match="Unknown evidence_finding implementation: 'bogus'"):
            build_chief_editor_orchestrator(config=config)

    def test_unknown_opinion_implementation_raises(self, tmp_path: Path) -> None:
        """Non-default implementation for opinion raises ValueError."""
        config_path = _write_test_config(
            tmp_path,
            specialist_overrides={"opinion": "mock"},
        )
        config = Config(config_path)
        import pytest

        with pytest.raises(ValueError, match="Unknown opinion implementation: 'mock'"):
            build_chief_editor_orchestrator(config=config)
