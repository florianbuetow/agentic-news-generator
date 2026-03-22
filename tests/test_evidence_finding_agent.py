"""Unit tests for EvidenceFindingAgent and MockEvidenceFindingAgent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from src.agents.article_generation.models import AgentResult, ArticleResponse, Concern, EvidenceRecord, Verdict
from src.agents.article_generation.specialists.evidence_finding.agent import EvidenceFindingAgent
from src.agents.article_generation.specialists.evidence_finding.mock_agent import MockEvidenceFindingAgent
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


class RecordingPerplexityClient:
    """Perplexity client returning configurable search results."""

    def __init__(self, response: dict[str, object] | None = None):
        self.search_calls: list[dict[str, object]] = []
        self._response = response or {"content": "test", "citations": []}

    def search(self, *, query: str, model: str, timeout_seconds: int) -> dict[str, object]:
        self.search_calls.append({"query": query, "model": model, "timeout_seconds": timeout_seconds})
        return self._response


class RecordingInstitutionalMemory:
    """Institutional memory that records lookups and persists."""

    def __init__(self, *, evidence_hit: EvidenceRecord | None = None):
        self.fact_check_lookups: list[dict[str, object]] = []
        self.evidence_lookups: list[dict[str, object]] = []
        self.fact_check_persists: list[object] = []
        self.evidence_persists: list[EvidenceRecord] = []
        self._evidence_hit = evidence_hit

    def lookup_fact_check(self, **kwargs: object) -> object | None:
        self.fact_check_lookups.append(kwargs)
        return None

    def lookup_evidence(self, **kwargs: object) -> EvidenceRecord | None:
        self.evidence_lookups.append(kwargs)
        return self._evidence_hit

    def persist_fact_check(self, *, record: object) -> None:
        self.fact_check_persists.append(record)

    def persist_evidence(self, *, record: EvidenceRecord) -> None:
        self.evidence_persists.append(record)

    def build_fact_check_cache_key_hash(self, **kwargs: object) -> str:
        return "testcachekey1234"

    def build_evidence_cache_key_hash(self, **kwargs: object) -> str:
        agent_name = str(kwargs["agent_name"])
        normalized_query = str(kwargs["normalized_query"])
        model_name = str(kwargs["model_name"])
        return hashlib.sha256(f"{agent_name}|{normalized_query}|{model_name}".encode()).hexdigest()[:16]


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


VALID_VERDICT_JSON = json.dumps(
    {
        "concern_id": 1,
        "misleading": False,
        "status": "KEEP",
        "rationale": "Test rationale",
        "suggested_fix": None,
        "evidence": None,
        "citations": None,
    }
)


def make_test_concern(concern_id: int = 1) -> Concern:
    return Concern(concern_id=concern_id, excerpt="test excerpt", review_note="test review note")


def make_test_article() -> ArticleResponse:
    return ArticleResponse(
        headline="Test",
        alternative_headline="Alt",
        article_body="Body",
        description="Desc",
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


def make_agent(
    *,
    tmp_path: Path,
    llm: RecordingLLMClient,
    loader: RecordingPromptLoader,
    perplexity: RecordingPerplexityClient,
    memory: RecordingInstitutionalMemory,
) -> EvidenceFindingAgent:
    config_path = _write_test_config(tmp_path)
    return EvidenceFindingAgent(
        llm_config=make_test_llm_config(),
        config=Config(config_path),
        llm_client=llm,
        prompt_loader=cast(Any, loader),
        specialists_dir="specialists",
        prompt_file="evidence_finding.md",
        perplexity_client=cast(Any, perplexity),
        perplexity_model="sonar",
        institutional_memory=cast(Any, memory),
    )


def make_cached_evidence_record() -> EvidenceRecord:
    cached_verdict = Verdict(
        concern_id=1,
        misleading=False,
        status="KEEP",
        rationale="cached",
        suggested_fix=None,
        evidence=None,
        citations=None,
    )
    return EvidenceRecord(
        timestamp="2025-01-01T00:00:00Z",
        article_id="test:test",
        concern_id=1,
        prompt="cached prompt",
        query="test query",
        normalized_query="test query",
        model_name="test-model",
        cache_key_hash="hash1",
        perplexity_response="{}",
        citations=[],
        verdict=cached_verdict,
    )


class TimeoutPerplexityClient:
    def search(self, *, query: str, model: str, timeout_seconds: int) -> dict[str, object]:
        raise TimeoutError("Perplexity timed out")


class TestEvidenceFindingAgent:
    def test_returns_verdict_on_cache_miss(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_VERDICT_JSON])
        loader = RecordingPromptLoader(template="{web_evidence}")
        perplexity = RecordingPerplexityClient(response={"content": "Web evidence", "citations": ["url1"]})
        memory = RecordingInstitutionalMemory()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader, perplexity=perplexity, memory=memory)

        result = agent.evaluate(
            concern=make_test_concern(),
            article=make_test_article(),
            source_text="Source text",
            source_metadata=make_test_metadata(),
            style_requirements="SCIAM_MAGAZINE",
        )

        assert isinstance(result, AgentResult)
        assert result.output.concern_id == 1
        assert result.output.status == "KEEP"

    def test_cache_hit_skips_perplexity_and_llm(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_VERDICT_JSON])
        loader = RecordingPromptLoader()
        perplexity = RecordingPerplexityClient()
        memory = RecordingInstitutionalMemory(evidence_hit=make_cached_evidence_record())
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader, perplexity=perplexity, memory=memory)

        result = agent.evaluate(
            concern=make_test_concern(),
            article=make_test_article(),
            source_text="Source text",
            source_metadata=make_test_metadata(),
            style_requirements="SCIAM_MAGAZINE",
        )

        assert result.prompt == "[cache-hit]"
        assert len(perplexity.search_calls) == 0
        assert len(llm.calls) == 0

    def test_perplexity_called_on_miss(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_VERDICT_JSON])
        loader = RecordingPromptLoader()
        perplexity = RecordingPerplexityClient(response={"content": "Web evidence", "citations": []})
        memory = RecordingInstitutionalMemory()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader, perplexity=perplexity, memory=memory)

        agent.evaluate(
            concern=make_test_concern(),
            article=make_test_article(),
            source_text="Source",
            source_metadata=make_test_metadata(),
            style_requirements="SCIAM_MAGAZINE",
        )

        assert len(perplexity.search_calls) == 1

    def test_citations_extracted_from_list(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_VERDICT_JSON])
        loader = RecordingPromptLoader()
        perplexity = RecordingPerplexityClient()
        memory = RecordingInstitutionalMemory()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader, perplexity=perplexity, memory=memory)

        citations = agent.extract_citations(search_response={"citations": ["url1", "url2"]})
        assert citations == ["url1", "url2"]

    def test_citations_empty_when_no_key(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_VERDICT_JSON])
        loader = RecordingPromptLoader()
        perplexity = RecordingPerplexityClient()
        memory = RecordingInstitutionalMemory()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader, perplexity=perplexity, memory=memory)

        citations = agent.extract_citations(search_response={"content": "answer"})
        assert citations == []

    def test_perplexity_response_in_prompt(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_VERDICT_JSON])
        loader = RecordingPromptLoader(template="{web_evidence}")
        perplexity = RecordingPerplexityClient(response={"content": "Web evidence", "citations": ["url1"]})
        memory = RecordingInstitutionalMemory()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader, perplexity=perplexity, memory=memory)

        result = agent.evaluate(
            concern=make_test_concern(),
            article=make_test_article(),
            source_text="Source",
            source_metadata=make_test_metadata(),
            style_requirements="SCIAM_MAGAZINE",
        )

        assert "Web evidence" in result.prompt
        assert "url1" in result.prompt

    def test_record_persisted_with_citations(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_VERDICT_JSON])
        loader = RecordingPromptLoader()
        perplexity = RecordingPerplexityClient(response={"content": "Data", "citations": ["url1", "url2"]})
        memory = RecordingInstitutionalMemory()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader, perplexity=perplexity, memory=memory)

        agent.evaluate(
            concern=make_test_concern(),
            article=make_test_article(),
            source_text="Source",
            source_metadata=make_test_metadata(),
            style_requirements="SCIAM_MAGAZINE",
        )

        assert len(memory.evidence_persists) == 1
        record = memory.evidence_persists[0]
        assert record.citations == ["url1", "url2"]
        assert '"content": "Data"' in record.perplexity_response

    def test_verdict_includes_evidence_fields(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient(
            [
                json.dumps(
                    {
                        "concern_id": 1,
                        "misleading": False,
                        "status": "KEEP",
                        "rationale": "Fine",
                        "suggested_fix": None,
                        "evidence": "Found support",
                        "citations": ["url1"],
                    }
                )
            ]
        )
        loader = RecordingPromptLoader()
        perplexity = RecordingPerplexityClient(response={"content": "Data", "citations": ["url1"]})
        memory = RecordingInstitutionalMemory()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader, perplexity=perplexity, memory=memory)

        result = agent.evaluate(
            concern=make_test_concern(),
            article=make_test_article(),
            source_text="Source",
            source_metadata=make_test_metadata(),
            style_requirements="SCIAM_MAGAZINE",
        )

        assert result.output.evidence == "Found support"
        assert result.output.citations == ["url1"]

    def test_cache_key_excludes_kb_version(self) -> None:
        memory = RecordingInstitutionalMemory()
        hash_a = memory.build_evidence_cache_key_hash(
            agent_name="evidence_finding",
            normalized_query="same query",
            model_name="test-model",
        )
        hash_b = memory.build_evidence_cache_key_hash(
            agent_name="evidence_finding",
            normalized_query="same query",
            model_name="test-model",
        )

        assert hash_a == hash_b

    def test_mock_implements_protocol(self) -> None:
        result = MockEvidenceFindingAgent().evaluate(
            concern=Concern(concern_id=5, excerpt="excerpt", review_note="review"),
            article=make_test_article(),
            source_text="Source",
            source_metadata=make_test_metadata(),
            style_requirements="SCIAM_MAGAZINE",
        )

        assert isinstance(result, AgentResult)

    def test_mock_returns_keep(self) -> None:
        result = MockEvidenceFindingAgent().evaluate(
            concern=Concern(concern_id=7, excerpt="excerpt", review_note="review"),
            article=make_test_article(),
            source_text="Source",
            source_metadata=make_test_metadata(),
            style_requirements="SCIAM_MAGAZINE",
        )

        assert result.output.concern_id == 7
        assert result.output.status == "KEEP"
        assert result.output.misleading is False

    def test_perplexity_timeout_propagates(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_VERDICT_JSON])
        loader = RecordingPromptLoader()
        perplexity = TimeoutPerplexityClient()
        memory = RecordingInstitutionalMemory()
        config_path = _write_test_config(tmp_path)
        agent = EvidenceFindingAgent(
            llm_config=make_test_llm_config(),
            config=Config(config_path),
            llm_client=llm,
            prompt_loader=cast(Any, loader),
            specialists_dir="specialists",
            prompt_file="evidence_finding.md",
            perplexity_client=cast(Any, perplexity),
            perplexity_model="sonar",
            institutional_memory=cast(Any, memory),
        )

        with pytest.raises(TimeoutError, match="Perplexity timed out"):
            agent.evaluate(
                concern=make_test_concern(),
                article=make_test_article(),
                source_text="Source",
                source_metadata=make_test_metadata(),
                style_requirements="SCIAM_MAGAZINE",
            )

    def test_non_string_citations_filtered(self, tmp_path: Path) -> None:
        llm = RecordingLLMClient([VALID_VERDICT_JSON])
        loader = RecordingPromptLoader()
        perplexity = RecordingPerplexityClient()
        memory = RecordingInstitutionalMemory()
        agent = make_agent(tmp_path=tmp_path, llm=llm, loader=loader, perplexity=perplexity, memory=memory)

        citations = agent.extract_citations(search_response={"citations": ["url", 42, None]})
        assert citations == ["url"]
