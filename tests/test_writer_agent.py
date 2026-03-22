"""Unit tests for WriterAgent and MockWriterAgent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError

from src.agents.article_generation.models import AgentResult, ArticleResponse, Verdict, WriterFeedback
from src.agents.article_generation.writer.agent import WriterAgent
from src.agents.article_generation.writer.mock_agent import MockWriterAgent
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


VALID_ARTICLE_JSON = json.dumps(
    {
        "headline": "Test",
        "alternative_headline": "Alt",
        "article_body": "Body text",
        "description": "Desc",
    }
)


def make_test_metadata() -> dict[str, str | None]:
    return {
        "channel_name": "TestChannel",
        "slug": "test-slug",
        "source_file": "test.txt",
        "video_id": "vid-1",
        "article_title": "Test",
        "publish_date": "2025-01-01",
        "references": "[]",
        "topic_slug": "test-topic",
    }


def make_test_feedback(*, rating: int = 3, passed: bool = False) -> WriterFeedback:
    return WriterFeedback(
        iteration=1,
        rating=rating,
        passed=passed,
        reasoning="Fix it",
        improvement_suggestions=["Improve C"],
        todo_list=["Fix A", "Fix B"],
        verdicts=[
            Verdict(
                concern_id=1,
                misleading=True,
                status="REWRITE",
                rationale="Needs rewrite",
                suggested_fix="Fix claim",
                evidence=None,
                citations=["https://example.com"],
            )
        ],
    )


def make_writer_agent(
    *,
    tmp_path: Path,
    llm: RecordingLLMClient,
    loader: RecordingPromptLoader,
    llm_config: LLMConfig | None = None,
) -> WriterAgent:
    config_path = _write_test_config(tmp_path)
    return WriterAgent(
        llm_config=llm_config or make_test_llm_config(),
        config=Config(config_path),
        llm_client=llm,
        prompt_loader=cast(Any, loader),
        writer_prompt_file="writer.md",
        revision_prompt_file="revision.md",
    )


class TestWriterAgent:
    def test_generate_returns_valid_result(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_ARTICLE_JSON])
        loader = RecordingPromptLoader(template="Template: {source_text}")
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        result = agent.generate(
            source_text="Source text",
            source_metadata=make_test_metadata(),
            style_mode="SCIAM_MAGAZINE",
            reader_preference="focus on methods",
        )

        assert isinstance(result, AgentResult)
        assert isinstance(result.output, ArticleResponse)
        assert result.output.headline != ""
        assert result.output.alternative_headline != ""
        assert result.output.article_body != ""
        assert result.output.description != ""

    def test_missing_fields_raises_validation_error(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([json.dumps({"headline": "X", "article_body": "Y"})])
        loader = RecordingPromptLoader()
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        with pytest.raises(ValidationError):
            agent.generate(
                source_text="Source text",
                source_metadata=make_test_metadata(),
                style_mode="SCIAM_MAGAZINE",
                reader_preference="pref",
            )

    def test_prompt_loaded_via_loader(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_ARTICLE_JSON])
        loader = RecordingPromptLoader()
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        agent.generate(
            source_text="Source text",
            source_metadata=make_test_metadata(),
            style_mode="SCIAM_MAGAZINE",
            reader_preference="pref",
        )

        assert loader.load_calls == ["writer.md"]

    def test_template_formatted_with_variables(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_ARTICLE_JSON])
        loader = RecordingPromptLoader(template="{style_mode} {reader_preference} {source_text} {source_metadata}")
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        agent.generate(
            source_text="Sample transcript",
            source_metadata=make_test_metadata(),
            style_mode="NATURE_NEWS",
            reader_preference="focus on methods",
        )

        messages = cast(list[dict[str, str]], llm.calls[0]["messages"])
        prompt = messages[0]["content"]
        assert "NATURE_NEWS" in prompt
        assert "focus on methods" in prompt
        assert "Sample transcript" in prompt
        assert "TestChannel" in prompt

    def test_single_user_message_sent(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_ARTICLE_JSON])
        loader = RecordingPromptLoader()
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        agent.generate(
            source_text="Source",
            source_metadata=make_test_metadata(),
            style_mode="SCIAM_MAGAZINE",
            reader_preference="",
        )

        assert len(llm.calls) == 1
        messages = cast(list[dict[str, str]], llm.calls[0]["messages"])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_non_json_response_raises(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient(["Not JSON"])
        loader = RecordingPromptLoader()
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        with pytest.raises(ValidationError):
            agent.generate(
                source_text="Source",
                source_metadata=make_test_metadata(),
                style_mode="SCIAM_MAGAZINE",
                reader_preference="",
            )

    def test_extra_fields_raise_validation_error(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient(
            [
                json.dumps(
                    {
                        "headline": "Test",
                        "alternative_headline": "Alt",
                        "article_body": "Body",
                        "description": "Desc",
                        "author": "X",
                    }
                )
            ]
        )
        loader = RecordingPromptLoader()
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        with pytest.raises(ValidationError):
            agent.generate(
                source_text="Source",
                source_metadata=make_test_metadata(),
                style_mode="SCIAM_MAGAZINE",
                reader_preference="",
            )

    def test_prompt_field_contains_assembled_text(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_ARTICLE_JSON])
        loader = RecordingPromptLoader(template="Template: {source_text}")
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        result = agent.generate(
            source_text="Hello",
            source_metadata=make_test_metadata(),
            style_mode="SCIAM_MAGAZINE",
            reader_preference="",
        )

        assert result.prompt == "Template: Hello"

    def test_revise_returns_valid_result(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_ARTICLE_JSON])
        loader = RecordingPromptLoader(template="{context}")
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        result = agent.revise(context='{"draft":"v1"}', feedback=make_test_feedback())

        assert isinstance(result, AgentResult)
        assert isinstance(result.output, ArticleResponse)
        assert result.output.headline != ""

    def test_revision_template_formatted_with_feedback(self, tmp_path: Path) -> None:
        template = "{rating} {pass_status} {reasoning} {todo_list} {improvement_suggestions} {verdicts} {context}"
        llm = RecordingLLMClient([VALID_ARTICLE_JSON])
        loader = RecordingPromptLoader(template=template)
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        feedback = make_test_feedback(rating=3, passed=False)
        agent.revise(context='{"iteration":1}', feedback=feedback)

        messages = cast(list[dict[str, str]], llm.calls[0]["messages"])
        prompt = messages[0]["content"]
        assert "3" in prompt
        assert "False" in prompt
        assert "Fix it" in prompt
        assert "- Fix A\n- Fix B" in prompt
        assert "- Improve C" in prompt
        assert '"status": "REWRITE"' in prompt
        assert '{"iteration":1}' in prompt

    def test_mock_satisfies_protocol(self) -> None:
        agent = MockWriterAgent()
        feedback = make_test_feedback()

        generated = agent.generate(
            source_text="source",
            source_metadata=make_test_metadata(),
            style_mode="SCIAM_MAGAZINE",
            reader_preference="",
        )
        revised = agent.revise(context="ctx", feedback=feedback)

        assert isinstance(generated, AgentResult)
        assert isinstance(revised, AgentResult)

    def test_mock_generate_returns_static(self) -> None:
        result = MockWriterAgent().generate(
            source_text="source",
            source_metadata=make_test_metadata(),
            style_mode="SCIAM_MAGAZINE",
            reader_preference="",
        )

        assert result.prompt == "[mock]"
        assert "Mock" in result.output.headline

    def test_mock_revise_returns_same_as_generate(self) -> None:
        agent = MockWriterAgent()
        generated = agent.generate(
            source_text="source",
            source_metadata=make_test_metadata(),
            style_mode="SCIAM_MAGAZINE",
            reader_preference="",
        )
        revised = agent.revise(context="ctx", feedback=make_test_feedback())

        assert revised.prompt == generated.prompt
        assert revised.output == generated.output

    def test_writer_prompt_contains_rules(self) -> None:
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "article_editor" / "writer.md"
        content = prompt_path.read_text(encoding="utf-8")

        assert "JSON" in content
        assert "source text" in content.lower()
        assert "NATURE_NEWS" in content
        assert "SCIAM_MAGAZINE" in content

    def test_writer_prompt_contains_target_length(self) -> None:
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "article_editor" / "writer.md"
        content = prompt_path.read_text(encoding="utf-8")

        assert "900" in content or "1200" in content

    def test_empty_reader_preference(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_ARTICLE_JSON])
        loader = RecordingPromptLoader(template="{reader_preference}|{source_text}")
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        result = agent.generate(
            source_text="Source",
            source_metadata=make_test_metadata(),
            style_mode="SCIAM_MAGAZINE",
            reader_preference="",
        )

        assert isinstance(result.output, ArticleResponse)

    def test_missing_required_json_field(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([json.dumps({"headline": "Test", "article_body": "Body"})])
        loader = RecordingPromptLoader()
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        with pytest.raises(ValidationError):
            agent.generate(
                source_text="Source",
                source_metadata=make_test_metadata(),
                style_mode="SCIAM_MAGAZINE",
                reader_preference="",
            )

    def test_long_source_text_proceeds(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_ARTICLE_JSON])
        loader = RecordingPromptLoader(template="{source_text}")
        agent = make_writer_agent(
            tmp_path=tmp_path,
            llm=llm,
            loader=loader,
            llm_config=make_test_llm_config(context_window=1_000_000, context_window_threshold=100),
        )

        result = agent.generate(
            source_text="x" * 100_000,
            source_metadata=make_test_metadata(),
            style_mode="SCIAM_MAGAZINE",
            reader_preference="",
        )

        assert isinstance(result.output, ArticleResponse)

    def test_null_metadata_values_serialized(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_ARTICLE_JSON])
        loader = RecordingPromptLoader(template="{source_metadata}")
        agent = make_writer_agent(tmp_path=tmp_path, llm=llm, loader=loader)

        metadata = make_test_metadata()
        metadata["publish_date"] = None
        result = agent.generate(
            source_text="Source",
            source_metadata=metadata,
            style_mode="SCIAM_MAGAZINE",
            reader_preference="",
        )

        assert isinstance(result.output, ArticleResponse)
        assert '"publish_date": null' in result.prompt
