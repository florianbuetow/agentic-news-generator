"""Unit tests for ConcernMappingAgent and MockConcernMappingAgent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError

from src.agents.article_generation.concern_mapping.agent import ConcernMappingAgent
from src.agents.article_generation.concern_mapping.mock_agent import MockConcernMappingAgent
from src.agents.article_generation.models import AgentResult, Concern, ConcernMappingResult
from src.config import Config, LLMConfig


class RecordingLLMClient:
    """LLM client that returns configurable responses and records calls."""

    def __init__(self, responses: list[str]):
        self.calls: list[dict[str, object]] = []
        self._responses = responses
        self._call_index = 0

    def complete(self, *, llm_config: LLMConfig, messages: list[dict[str, str]]) -> str:
        self.calls.append({"llm_config": llm_config, "messages": messages})
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return response


class RecordingPromptLoader:
    """PromptLoader that returns configurable templates and records calls."""

    def __init__(self, template: str = "{source_text}"):
        self.load_calls: list[str] = []
        self.load_specialist_calls: list[tuple[str, str]] = []
        self._template = template

    def load_prompt(self, *, prompt_file: str) -> str:
        self.load_calls.append(prompt_file)
        return self._template

    def load_specialist_prompt(self, *, specialists_dir: str, prompt_file: str) -> str:
        self.load_specialist_calls.append((specialists_dir, prompt_file))
        return self._template


def make_test_llm_config(**overrides: Any) -> LLMConfig:
    defaults: dict[str, Any] = {
        "model": "test-model",
        "api_base": "http://localhost:1234/v1",
        "api_key": "test",
        "context_window": 32768,
        "max_tokens": 2048,
        "temperature": 0.3,
        "context_window_threshold": 90,
        "max_retries": 0,
        "retry_delay": 1,
        "timeout_seconds": 60,
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


def _write_test_config(tmp_dir: Path) -> Path:
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

    def agent_slot(impl: str = "default") -> dict[str, object]:
        return {"agent_name": impl, "llm": llm}

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
                "writer": agent_slot("default"),
                "article_review": agent_slot("default"),
                "concern_mapping": agent_slot("default"),
                "specialists": {
                    "fact_check": agent_slot("default"),
                    "evidence_finding": agent_slot("default"),
                    "opinion": agent_slot("default"),
                    "attribution": agent_slot("default"),
                    "style_review": agent_slot("default"),
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
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.dump(config_data, handle)
    return config_path


def make_test_concern(concern_id: int = 1) -> Concern:
    return Concern(concern_id=concern_id, excerpt="test excerpt", review_note="test review note")


def make_agent(*, tmp_path: Path, llm: RecordingLLMClient, loader: RecordingPromptLoader) -> ConcernMappingAgent:
    config_path = _write_test_config(tmp_path)
    return ConcernMappingAgent(
        llm_config=make_test_llm_config(),
        config=Config(config_path),
        llm_client=llm,
        prompt_loader=cast(Any, loader),
        prompt_file="concern_mapping.md",
    )


VALID_MAPPING_JSON = json.dumps(
    {
        "concern_id": 1,
        "concern_type": "scope_expansion",
        "selected_agent": "opinion",
        "confidence": "high",
        "reason": "test reason",
    }
)


class TestConcernMappingAgent:
    def test_returns_valid_mappings(self, tmp_path: Path) -> None:
        response = json.dumps(
            {
                "mappings": [
                    json.loads(VALID_MAPPING_JSON),
                    {
                        "concern_id": 2,
                        "concern_type": "unsupported_fact",
                        "selected_agent": "fact_check",
                        "confidence": "medium",
                        "reason": "needs verification",
                    },
                ]
            }
        )
        llm = RecordingLLMClient([response])
        loader = RecordingPromptLoader()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        result = agent.map_concerns(
            style_requirements="SCIAM_MAGAZINE",
            source_text="Source text",
            generated_article_json='{"headline":"Test"}',
            concerns=[make_test_concern(1), make_test_concern(2)],
        )

        assert isinstance(result, AgentResult)
        assert isinstance(result.output, ConcernMappingResult)
        assert len(result.output.mappings) == 2
        assert result.output.mappings[0].concern_id == 1
        assert result.output.mappings[1].concern_id == 2

    def test_invalid_concern_type_raises(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient(
            [
                json.dumps(
                    {
                        "mappings": [
                            {
                                "concern_id": 1,
                                "concern_type": "invalid_type",
                                "selected_agent": "opinion",
                                "confidence": "high",
                                "reason": "bad type",
                            }
                        ]
                    }
                )
            ]
        )
        loader = RecordingPromptLoader()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        with pytest.raises(ValidationError):
            agent.map_concerns(
                style_requirements="SCIAM_MAGAZINE",
                source_text="Source",
                generated_article_json="{}",
                concerns=[make_test_concern()],
            )

    def test_invalid_selected_agent_raises(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient(
            [
                json.dumps(
                    {
                        "mappings": [
                            {
                                "concern_id": 1,
                                "concern_type": "scope_expansion",
                                "selected_agent": "nonexistent",
                                "confidence": "high",
                                "reason": "bad agent",
                            }
                        ]
                    }
                )
            ]
        )
        loader = RecordingPromptLoader()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        with pytest.raises(ValidationError):
            agent.map_concerns(
                style_requirements="SCIAM_MAGAZINE",
                source_text="Source",
                generated_article_json="{}",
                concerns=[make_test_concern()],
            )

    def test_json_array_format(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([f"[{VALID_MAPPING_JSON}]"])
        loader = RecordingPromptLoader()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        result = agent.map_concerns(
            style_requirements="SCIAM_MAGAZINE",
            source_text="Source",
            generated_article_json="{}",
            concerns=[make_test_concern()],
        )

        assert len(result.output.mappings) == 1
        assert result.output.mappings[0].selected_agent == "opinion"

    def test_json_object_format(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([json.dumps({"mappings": [json.loads(VALID_MAPPING_JSON)]})])
        loader = RecordingPromptLoader()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        result = agent.map_concerns(
            style_requirements="SCIAM_MAGAZINE",
            source_text="Source",
            generated_article_json="{}",
            concerns=[make_test_concern()],
        )

        assert len(result.output.mappings) == 1
        assert result.output.mappings[0].concern_id == 1

    def test_array_non_object_items_raises(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient(["[1, 2, 3]"])
        loader = RecordingPromptLoader()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        with pytest.raises(ValueError, match="JSON objects"):
            agent.map_concerns(
                style_requirements="SCIAM_MAGAZINE",
                source_text="Source",
                generated_article_json="{}",
                concerns=[make_test_concern()],
            )

    def test_mock_satisfies_protocol(self) -> None:
        agent = MockConcernMappingAgent()
        result = agent.map_concerns(
            style_requirements="SCIAM_MAGAZINE",
            source_text="Source",
            generated_article_json="{}",
            concerns=[make_test_concern()],
        )

        assert isinstance(result, AgentResult)

    def test_mock_returns_empty(self) -> None:
        result = MockConcernMappingAgent().map_concerns(
            style_requirements="SCIAM_MAGAZINE",
            source_text="Source",
            generated_article_json="{}",
            concerns=[make_test_concern()],
        )

        assert result.prompt == "[mock]"
        assert result.output.mappings == []

    def test_single_concern_mapped(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([json.dumps({"mappings": [json.loads(VALID_MAPPING_JSON)]})])
        loader = RecordingPromptLoader()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        result = agent.map_concerns(
            style_requirements="SCIAM_MAGAZINE",
            source_text="Source",
            generated_article_json='{"headline":"Only one"}',
            concerns=[make_test_concern()],
        )

        assert len(result.output.mappings) == 1

    def test_invalid_agent_literal_rejected(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient(
            [
                json.dumps(
                    {
                        "mappings": [
                            {
                                "concern_id": 1,
                                "concern_type": "scope_expansion",
                                "selected_agent": "sentiment",
                                "confidence": "high",
                                "reason": "bad literal",
                            }
                        ]
                    }
                )
            ]
        )
        loader = RecordingPromptLoader()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        with pytest.raises(ValidationError):
            agent.map_concerns(
                style_requirements="SCIAM_MAGAZINE",
                source_text="Source",
                generated_article_json="{}",
                concerns=[make_test_concern()],
            )

    def test_markdown_fenced_object_stripped(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([f"```json\n{json.dumps({'mappings': [json.loads(VALID_MAPPING_JSON)]})}\n```"])
        loader = RecordingPromptLoader()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        result = agent.map_concerns(
            style_requirements="SCIAM_MAGAZINE",
            source_text="Source",
            generated_article_json="{}",
            concerns=[make_test_concern()],
        )

        assert len(result.output.mappings) == 1
        assert result.output.mappings[0].selected_agent == "opinion"

    def test_empty_array_accepted(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient(["[]"])
        loader = RecordingPromptLoader()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        result = agent.map_concerns(
            style_requirements="SCIAM_MAGAZINE",
            source_text="Source",
            generated_article_json="{}",
            concerns=[make_test_concern()],
        )

        assert result.output.mappings == []

    def test_prompt_formatted_with_context(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([json.dumps({"mappings": []})])
        loader = RecordingPromptLoader(template="{style_requirements} {source_text} {generated_article} {concerns}")
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        concerns = [make_test_concern(1), make_test_concern(2)]
        agent.map_concerns(
            style_requirements="NATURE_NEWS",
            source_text="Original source",
            generated_article_json='{"headline":"Generated"}',
            concerns=concerns,
        )

        messages = cast(list[dict[str, str]], llm.calls[0]["messages"])
        prompt = messages[0]["content"]
        assert "NATURE_NEWS" in prompt
        assert "Original source" in prompt
        assert '{"headline":"Generated"}' in prompt
        assert '"concern_id": 1' in prompt
