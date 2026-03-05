"""End-to-end test for the full article generation pipeline.

Requires a running LM Studio instance. Mock specialists are used for
KB (fact-check) and Perplexity (evidence-finding) to avoid external deps.

Run with: uv run pytest tests/e2e/ -v -s -m e2e
"""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from src.agents.article_generation.agent import build_chief_editor_orchestrator
from src.agents.article_generation.bundle_loader import bundle_to_source_metadata, load_bundle
from src.config import Config

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "article_generation" / "sample_bundle"

pytestmark = pytest.mark.e2e


def _build_e2e_config(tmp_path: Path) -> Config:
    """Build a Config pointing at tmp_path for outputs, real LLM, mock specialists."""
    # Create required directories
    for subdir in [
        "knowledgebase",
        "knowledgebase_index",
        "institutional_memory/fact_checking",
        "institutional_memory/evidence_finding",
        "output/articles",
        "output/article_editor_runs",
        "articles/input",
    ]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

    llm = {
        "model": "openai/qwen3-30b-a3b-thinking-2507-mlx@8bit",
        "api_base": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
        "context_window": 262144,
        "max_tokens": 4096,
        "temperature": 0.7,
        "context_window_threshold": 90,
        "max_retries": 0,
        "retry_delay": 2.0,
        "timeout_seconds": 300,
    }

    specialist_llm = {**llm, "max_tokens": 2048, "temperature": 0.3}

    def agent_slot(impl: str = "default", use_llm: dict | None = None) -> dict:
        return {"implementation": impl, "llm": use_llm or llm}

    config_data = {
        "paths": {
            "data_dir": str(tmp_path),
            "data_models_dir": str(tmp_path / "models"),
            "data_downloads_dir": str(tmp_path / "downloads"),
            "data_downloads_videos_dir": str(tmp_path / "downloads" / "videos"),
            "data_downloads_transcripts_dir": str(tmp_path / "downloads" / "transcripts"),
            "data_downloads_transcripts_hallucinations_dir": str(tmp_path / "downloads" / "transcripts-hallucinations"),
            "data_downloads_transcripts_cleaned_dir": str(tmp_path / "downloads" / "transcripts_cleaned"),
            "data_transcripts_topics_dir": str(tmp_path / "downloads" / "transcripts-topics"),
            "data_downloads_audio_dir": str(tmp_path / "downloads" / "audio"),
            "data_downloads_metadata_dir": str(tmp_path / "downloads" / "metadata"),
            "data_output_dir": str(tmp_path / "output"),
            "data_input_dir": str(tmp_path / "input"),
            "data_temp_dir": str(tmp_path / "temp"),
            "data_archive_dir": str(tmp_path / "archive"),
            "data_archive_videos_dir": str(tmp_path / "archive" / "videos"),
            "data_logs_dir": str(tmp_path / "logs"),
            "data_output_articles_dir": str(tmp_path / "output" / "articles"),
            "data_articles_input_dir": str(tmp_path / "articles" / "input"),
            "reports_dir": str(tmp_path / "reports"),
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
                "editor_max_rounds": 1,
                "output": {
                    "final_articles_dir": str(tmp_path / "output" / "articles"),
                    "run_artifacts_dir": str(tmp_path / "output" / "article_editor_runs"),
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
                "article_review": agent_slot(use_llm=specialist_llm),
                "concern_mapping": agent_slot(use_llm=specialist_llm),
                "specialists": {
                    "fact_check": agent_slot("mock", specialist_llm),
                    "evidence_finding": agent_slot("mock", specialist_llm),
                    "opinion": agent_slot(use_llm=specialist_llm),
                    "attribution": agent_slot(use_llm=specialist_llm),
                    "style_review": agent_slot(use_llm=specialist_llm),
                },
            },
            "knowledge_base": {
                "data_dir": str(tmp_path / "knowledgebase"),
                "index_dir": str(tmp_path / "knowledgebase_index"),
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
                "api_key": "not-used-in-test",
                "model": "sonar",
                "timeout_seconds": 45,
            },
            "institutional_memory": {
                "data_dir": str(tmp_path / "institutional_memory"),
                "fact_checking_subdir": "fact_checking",
                "evidence_finding_subdir": "evidence_finding",
            },
            "allowed_styles": ["NATURE_NEWS", "SCIAM_MAGAZINE"],
            "default_style_mode": "SCIAM_MAGAZINE",
        },
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return Config(config_path)


class TestFullPipeline:
    """E2E test: orchestrator produces an article from a sample bundle."""

    def test_orchestrator_produces_article(self, tmp_path: Path) -> None:
        """Full pipeline produces a valid ArticleGenerationResult."""
        config = _build_e2e_config(tmp_path)
        orchestrator = build_chief_editor_orchestrator(config=config)

        bundle = load_bundle(FIXTURES_DIR)
        source_metadata = bundle_to_source_metadata(bundle)

        result = orchestrator.generate_article(
            source_text=bundle.source_text,
            source_metadata=source_metadata,
            style_mode="SCIAM_MAGAZINE",
            reader_preference="",
        )

        # Structural assertions
        assert result.success, f"Expected success, got error: {result.error}"
        assert result.article is not None
        assert result.article.headline != ""
        assert result.article.article_body != ""
        assert result.editor_report is not None
        assert result.editor_report.total_iterations >= 1
        assert result.artifacts_dir is not None

        # Output files exist
        artifacts = Path(result.artifacts_dir)
        assert artifacts.exists()
