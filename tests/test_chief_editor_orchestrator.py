"""Tests for ChiefEditorOrchestrator behavior."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml

from src.agents.article_generation.agent import build_chief_editor_orchestrator
from src.agents.article_generation.article_review.mock_agent import MockArticleReviewAgent
from src.agents.article_generation.base import (
    ArticleReviewAgentProtocol,
    ConcernMappingAgentProtocol,
    SpecialistAgentProtocol,
    WriterAgentProtocol,
)
from src.agents.article_generation.chief_editor.bullet_parser import ArticleReviewBulletParser
from src.agents.article_generation.chief_editor.institutional_memory import InstitutionalMemoryStore
from src.agents.article_generation.chief_editor.orchestrator import ChiefEditorOrchestrator
from src.agents.article_generation.chief_editor.output_handler import OutputHandler
from src.agents.article_generation.concern_mapping.mock_agent import MockConcernMappingAgent
from src.agents.article_generation.models import (
    AgentResult,
    ArticleGenerationResult,
    ArticleResponse,
    ArticleReviewRaw,
    Concern,
    ConcernMapping,
    ConcernMappingResult,
    Verdict,
    WriterFeedback,
)
from src.agents.article_generation.specialists.attribution.mock_agent import MockAttributionAgent
from src.agents.article_generation.specialists.evidence_finding.mock_agent import MockEvidenceFindingAgent
from src.agents.article_generation.specialists.fact_check.mock_agent import MockFactCheckAgent
from src.agents.article_generation.specialists.opinion.mock_agent import MockOpinionAgent
from src.agents.article_generation.specialists.style_review.mock_agent import MockStyleReviewAgent
from src.agents.article_generation.writer.mock_agent import MockWriterAgent
from src.config import Config


def make_article(**overrides: Any) -> ArticleResponse:
    defaults = {
        "headline": "Test Headline",
        "alternative_headline": "Test Alternative Headline",
        "article_body": "Test article body.",
        "description": "Test description.",
    }
    defaults.update(overrides)
    return ArticleResponse(**defaults)


def make_concern(concern_id: int, excerpt: str, review_note: str) -> Concern:
    return Concern(
        concern_id=concern_id,
        excerpt=excerpt,
        review_note=review_note,
    )


def make_verdict(
    concern_id: int,
    misleading: bool,
    status: Literal["KEEP", "REWRITE", "REMOVE"],
    rationale: str | None = None,
    suggested_fix: str | None = None,
    evidence: str | None = None,
    citations: list[str] | None = None,
) -> Verdict:
    return Verdict(
        concern_id=concern_id,
        misleading=misleading,
        status=status,
        rationale=rationale or f"Rationale {concern_id}",
        suggested_fix=suggested_fix,
        evidence=evidence,
        citations=citations,
    )


def make_mapping(
    concern_id: int,
    selected_agent: Literal["fact_check", "evidence_finding", "opinion", "attribution", "style_review"],
    concern_type: Literal[
        "unsupported_fact",
        "inferred_fact",
        "scope_expansion",
        "editorializing",
        "structured_addition",
        "attribution_gap",
        "certainty_inflation",
        "truncation_completion",
    ] = "scope_expansion",
    confidence: Literal["high", "medium", "low"] = "high",
    reason: str | None = None,
) -> ConcernMapping:
    return ConcernMapping(
        concern_id=concern_id,
        concern_type=concern_type,
        selected_agent=selected_agent,
        confidence=confidence,
        reason=reason or f"Route concern {concern_id}",
    )


def make_source_metadata(**overrides: Any) -> dict[str, str | None]:
    defaults: dict[str, str | None] = {
        "channel_name": "TestChannel",
        "slug": "test-article",
        "source_file": "transcript.txt",
        "video_id": "video-123",
        "article_title": "Test Article",
        "publish_date": "2025-01-01",
        "references": "[]",
    }
    defaults.update(overrides)
    return defaults


class RecordingWriterAgent:
    def __init__(self, articles: list[ArticleResponse] | None = None) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.revise_calls: list[dict[str, Any]] = []
        self._items = articles if articles is not None else [make_article()]
        self._call_index = 0

    def generate(
        self,
        *,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_mode: str,
        reader_preference: str,
    ) -> AgentResult[ArticleResponse]:
        self.generate_calls.append(
            {
                "source_text": source_text,
                "source_metadata": source_metadata,
                "style_mode": style_mode,
                "reader_preference": reader_preference,
            }
        )
        article = self._items[min(self._call_index, len(self._items) - 1)]
        self._call_index += 1
        return AgentResult(prompt="[test]", output=article)

    def revise(
        self,
        *,
        context: str,
        feedback: WriterFeedback,
    ) -> AgentResult[ArticleResponse]:
        self.revise_calls.append(
            {
                "context": context,
                "feedback": feedback,
            }
        )
        article = self._items[min(self._call_index, len(self._items) - 1)]
        self._call_index += 1
        return AgentResult(prompt="[test]", output=article)


class RecordingReviewAgent:
    def __init__(self, responses: list[str]) -> None:
        self.review_calls: list[dict[str, Any]] = []
        self._items = responses
        self._call_index = 0

    def review(
        self,
        *,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
    ) -> AgentResult[ArticleReviewRaw]:
        self.review_calls.append(
            {
                "article": article,
                "source_text": source_text,
                "source_metadata": source_metadata,
            }
        )
        response = self._items[min(self._call_index, len(self._items) - 1)]
        self._call_index += 1
        return AgentResult(prompt="[test]", output=ArticleReviewRaw(markdown_bullets=response))


class RecordingMappingAgent:
    def __init__(self, mappings_per_call: list[list[ConcernMapping]]) -> None:
        self.map_calls: list[dict[str, Any]] = []
        self._items = mappings_per_call
        self._call_index = 0

    def map_concerns(
        self,
        *,
        style_requirements: str,
        source_text: str,
        generated_article_json: str,
        concerns: list[Concern],
    ) -> AgentResult[ConcernMappingResult]:
        self.map_calls.append(
            {
                "style_requirements": style_requirements,
                "source_text": source_text,
                "generated_article_json": generated_article_json,
                "concerns": concerns,
            }
        )
        response = self._items[min(self._call_index, len(self._items) - 1)]
        self._call_index += 1
        return AgentResult(prompt="[test]", output=ConcernMappingResult(mappings=response))


class RecordingSpecialistAgent:
    def __init__(self, verdicts: list[Verdict]) -> None:
        self.evaluate_calls: list[dict[str, Any]] = []
        self._items = verdicts
        self._call_index = 0

    def evaluate(
        self,
        *,
        concern: Concern,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_requirements: str,
    ) -> AgentResult[Verdict]:
        self.evaluate_calls.append(
            {
                "concern": concern,
                "article": article,
                "source_text": source_text,
                "source_metadata": source_metadata,
                "style_requirements": style_requirements,
            }
        )
        response = self._items[min(self._call_index, len(self._items) - 1)]
        self._call_index += 1
        return AgentResult(prompt="[test]", output=response)


class ExplodingWriterAgent:
    def generate(
        self,
        *,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_mode: str,
        reader_preference: str,
    ) -> AgentResult[ArticleResponse]:
        raise RuntimeError("LLM failed")

    def revise(
        self,
        *,
        context: str,
        feedback: WriterFeedback,
    ) -> AgentResult[ArticleResponse]:
        raise RuntimeError("LLM failed")


def _make_test_config(
    tmp_path: Path,
    *,
    editor_max_rounds: int,
    writer_agent_name: str = "mock",
    article_review_agent_name: str = "mock",
    concern_mapping_agent_name: str = "mock",
    fact_check_agent_name: str = "mock",
    evidence_finding_agent_name: str = "mock",
    opinion_agent_name: str = "mock",
    attribution_agent_name: str = "mock",
    style_review_agent_name: str = "mock",
) -> Config:
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

    def agent_slot(agent_name: str) -> dict[str, object]:
        return {"agent_name": agent_name, "llm": llm}

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
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

    config_data: dict[str, object] = {
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
            "data_article_generation_output_dir": str(tmp_path / "output" / "articles"),
            "data_article_generation_artifacts_dir": str(tmp_path / "output" / "article_editor_runs"),
            "data_article_generation_kb_dir": str(tmp_path / "knowledgebase"),
            "data_article_generation_kb_index_dir": str(tmp_path / "knowledgebase_index"),
            "data_article_generation_institutional_memory_dir": str(tmp_path / "institutional_memory"),
            "data_article_generation_prompts_dir": str(tmp_path / "prompts" / "article_editor"),
            "data_topic_detection_output_dir": str(tmp_path / "output" / "topics"),
            "data_topic_detection_taxonomies_dir": str(tmp_path / "input" / "taxonomies"),
            "data_topic_detection_taxonomy_cache_dir": str(tmp_path / "input" / "taxonomies" / "cache"),
            "data_hallucination_detection_output_dir": str(tmp_path / "downloads" / "transcripts-hallucinations"),
            "data_article_compiler_input_dir": str(tmp_path / "input" / "newspaper" / "articles"),
            "data_article_compiler_output_file": str(tmp_path / "input" / "newspaper" / "articles.js"),
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
                "editor_max_rounds": editor_max_rounds,
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
                "writer": agent_slot(writer_agent_name),
                "article_review": agent_slot(article_review_agent_name),
                "concern_mapping": agent_slot(concern_mapping_agent_name),
                "specialists": {
                    "fact_check": agent_slot(fact_check_agent_name),
                    "evidence_finding": agent_slot(evidence_finding_agent_name),
                    "opinion": agent_slot(opinion_agent_name),
                    "attribution": agent_slot(attribution_agent_name),
                    "style_review": agent_slot(style_review_agent_name),
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

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.dump(config_data, handle)

    return Config(config_path)


def build_test_orchestrator(
    tmp_path: Path,
    *,
    writer: WriterAgentProtocol | None = None,
    review: ArticleReviewAgentProtocol | None = None,
    mapping: ConcernMappingAgentProtocol | None = None,
    fact_check: SpecialistAgentProtocol | None = None,
    evidence: SpecialistAgentProtocol | None = None,
    opinion: SpecialistAgentProtocol | None = None,
    attribution: SpecialistAgentProtocol | None = None,
    style_review: SpecialistAgentProtocol | None = None,
    editor_max_rounds: int = 3,
) -> ChiefEditorOrchestrator:
    config = _make_test_config(tmp_path, editor_max_rounds=editor_max_rounds)
    output_handler = OutputHandler(
        final_articles_dir=tmp_path / "articles",
        run_artifacts_dir=tmp_path / "runs",
    )
    return ChiefEditorOrchestrator(
        config=config,
        writer_agent=writer or MockWriterAgent(),
        article_review_agent=review or MockArticleReviewAgent(),
        concern_mapping_agent=mapping or MockConcernMappingAgent(),
        fact_check_agent=fact_check or MockFactCheckAgent(),
        evidence_finding_agent=evidence or MockEvidenceFindingAgent(),
        opinion_agent=opinion or MockOpinionAgent(),
        attribution_agent=attribution or MockAttributionAgent(),
        style_review_agent=style_review or MockStyleReviewAgent(),
        bullet_parser=ArticleReviewBulletParser(),
        institutional_memory=InstitutionalMemoryStore(
            data_dir=tmp_path / "memory",
            fact_checking_subdir="fact_checking",
            evidence_finding_subdir="evidence_finding",
        ),
        output_handler=output_handler,
    )


def _generate_article(
    orchestrator: ChiefEditorOrchestrator,
    *,
    source_metadata: dict[str, str | None] | None = None,
) -> ArticleGenerationResult:
    return orchestrator.generate_article(
        source_text="Source transcript text.",
        source_metadata=source_metadata if source_metadata is not None else make_source_metadata(),
        style_mode="SCIAM_MAGAZINE",
        reader_preference="",
    )


class TestOrchestratorHappyPath:
    def test_returns_article_generation_result(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)

        result = _generate_article(orchestrator)

        assert isinstance(result, ArticleGenerationResult)
        assert result.success is True
        assert result.article is not None
        assert result.metadata is not None
        assert result.editor_report is not None
        assert result.artifacts_dir is not None
        assert result.error is None

    def test_success_when_no_concerns(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)

        result = _generate_article(orchestrator)

        assert result.success is True
        assert result.editor_report is not None
        assert result.editor_report.final_status == "SUCCESS"
        assert result.editor_report.total_iterations == 1
        assert result.editor_report.iterations == []
        assert result.editor_report.blocking_concerns is None

    def test_success_when_no_misleading_verdicts(self, tmp_path: Path) -> None:
        writer = RecordingWriterAgent([make_article()])
        review = RecordingReviewAgent(["- Concern A\n- Concern B"])
        mapping = RecordingMappingAgent([[make_mapping(1, "opinion"), make_mapping(2, "opinion")]])
        opinion = RecordingSpecialistAgent(
            [
                make_verdict(1, misleading=False, status="KEEP", rationale="Keep 1"),
                make_verdict(2, misleading=False, status="KEEP", rationale="Keep 2"),
            ]
        )
        orchestrator = build_test_orchestrator(
            tmp_path,
            writer=writer,
            review=review,
            mapping=mapping,
            opinion=opinion,
        )

        result = _generate_article(orchestrator)

        assert result.success is True
        assert result.editor_report is not None
        assert result.editor_report.final_status == "SUCCESS"
        assert result.editor_report.total_iterations == 1
        assert len(result.editor_report.iterations) == 1
        iteration = result.editor_report.iterations[0]
        assert len(iteration.concerns) == 2
        assert len(iteration.mappings) == 2
        assert len(iteration.verdicts) == 2
        assert iteration.feedback_to_writer is None


class TestOrchestratorFeedbackLoop:
    def test_feedback_compiled_and_revision_requested(self, tmp_path: Path) -> None:
        writer = RecordingWriterAgent([make_article(headline="Draft 1"), make_article(headline="Draft 2")])
        review = RecordingReviewAgent(["- Concern A", ""])
        mapping = RecordingMappingAgent([[make_mapping(1, "opinion")]])
        opinion = RecordingSpecialistAgent([make_verdict(1, misleading=True, status="REWRITE", suggested_fix="Fix this")])
        orchestrator = build_test_orchestrator(
            tmp_path,
            writer=writer,
            review=review,
            mapping=mapping,
            opinion=opinion,
            editor_max_rounds=3,
        )

        result = _generate_article(orchestrator)

        assert len(writer.generate_calls) == 1
        assert len(writer.revise_calls) == 1
        assert len(writer.generate_calls) + len(writer.revise_calls) == 2
        assert result.success is True
        assert result.editor_report is not None
        assert result.editor_report.total_iterations == 2
        assert len(result.editor_report.iterations) == 1
        assert result.editor_report.iterations[0].feedback_to_writer is not None
        assert result.editor_report.iterations[0].feedback_to_writer is not None
        assert result.editor_report.iterations[0].feedback_to_writer.passed is False
        assert len(review.review_calls) == 2

    def test_multi_round_revision_converges(self, tmp_path: Path) -> None:
        writer = RecordingWriterAgent(
            [
                make_article(headline="Draft 1"),
                make_article(headline="Draft 2"),
                make_article(headline="Draft 3"),
            ]
        )
        review = RecordingReviewAgent(["- Concern A", "- Concern B", ""])
        mapping = RecordingMappingAgent([[make_mapping(1, "opinion")], [make_mapping(1, "opinion")]])
        opinion = RecordingSpecialistAgent(
            [
                make_verdict(1, misleading=True, status="REWRITE", suggested_fix="Fix A"),
                make_verdict(1, misleading=True, status="REWRITE", suggested_fix="Fix B"),
            ]
        )
        orchestrator = build_test_orchestrator(
            tmp_path,
            writer=writer,
            review=review,
            mapping=mapping,
            opinion=opinion,
            editor_max_rounds=3,
        )

        result = _generate_article(orchestrator)

        assert result.success is True
        assert result.editor_report is not None
        assert result.editor_report.total_iterations == 3
        assert len(writer.generate_calls) == 1
        assert len(writer.revise_calls) == 2
        assert len(writer.generate_calls) + len(writer.revise_calls) == 3

    def test_failed_after_max_rounds(self, tmp_path: Path) -> None:
        writer = RecordingWriterAgent([make_article(headline="Draft 1"), make_article(headline="Draft 2")])
        review = RecordingReviewAgent(["- Concern A"])
        mapping = RecordingMappingAgent([[make_mapping(1, "opinion")]])
        opinion = RecordingSpecialistAgent([make_verdict(1, misleading=True, status="REWRITE", suggested_fix="Fix A")])
        orchestrator = build_test_orchestrator(
            tmp_path,
            writer=writer,
            review=review,
            mapping=mapping,
            opinion=opinion,
            editor_max_rounds=2,
        )

        result = _generate_article(orchestrator)

        assert result.success is False
        assert result.error is not None
        assert "Unresolved misleading concerns after editor_max_rounds" in result.error
        assert result.editor_report is not None
        assert result.editor_report.final_status == "FAILED"
        assert result.editor_report.total_iterations == 2
        assert result.editor_report.blocking_concerns is not None
        assert len(result.editor_report.blocking_concerns) == 1
        assert result.article is not None


class TestOrchestratorLoopMechanics:
    def test_loop_step_execution_order(self, tmp_path: Path) -> None:
        call_order: list[str] = []

        class OrderedWriter(RecordingWriterAgent):
            def generate(
                self,
                *,
                source_text: str,
                source_metadata: dict[str, str | None],
                style_mode: str,
                reader_preference: str,
            ) -> AgentResult[ArticleResponse]:
                call_order.append("writer.generate")
                return super().generate(
                    source_text=source_text,
                    source_metadata=source_metadata,
                    style_mode=style_mode,
                    reader_preference=reader_preference,
                )

            def revise(
                self,
                *,
                context: str,
                feedback: WriterFeedback,
            ) -> AgentResult[ArticleResponse]:
                call_order.append("writer.revise")
                return super().revise(context=context, feedback=feedback)

        class OrderedReview(RecordingReviewAgent):
            def review(
                self,
                *,
                article: ArticleResponse,
                source_text: str,
                source_metadata: dict[str, str | None],
            ) -> AgentResult[ArticleReviewRaw]:
                call_order.append("review.review")
                return super().review(
                    article=article,
                    source_text=source_text,
                    source_metadata=source_metadata,
                )

        class OrderedMapping(RecordingMappingAgent):
            def map_concerns(
                self,
                *,
                style_requirements: str,
                source_text: str,
                generated_article_json: str,
                concerns: list[Concern],
            ) -> AgentResult[ConcernMappingResult]:
                call_order.append("mapping.map_concerns")
                return super().map_concerns(
                    style_requirements=style_requirements,
                    source_text=source_text,
                    generated_article_json=generated_article_json,
                    concerns=concerns,
                )

        class OrderedSpecialist(RecordingSpecialistAgent):
            def evaluate(
                self,
                *,
                concern: Concern,
                article: ArticleResponse,
                source_text: str,
                source_metadata: dict[str, str | None],
                style_requirements: str,
            ) -> AgentResult[Verdict]:
                call_order.append("specialist.evaluate")
                return super().evaluate(
                    concern=concern,
                    article=article,
                    source_text=source_text,
                    source_metadata=source_metadata,
                    style_requirements=style_requirements,
                )

        writer = OrderedWriter([make_article(headline="Draft 1"), make_article(headline="Draft 2")])
        review = OrderedReview(["- Concern A", ""])
        mapping = OrderedMapping([[make_mapping(1, "opinion")]])
        opinion = OrderedSpecialist([make_verdict(1, misleading=True, status="REWRITE", suggested_fix="Fix")])
        orchestrator = build_test_orchestrator(
            tmp_path,
            writer=writer,
            review=review,
            mapping=mapping,
            opinion=opinion,
            editor_max_rounds=2,
        )

        _generate_article(orchestrator)

        assert call_order == [
            "writer.generate",
            "review.review",
            "mapping.map_concerns",
            "specialist.evaluate",
            "writer.revise",
            "review.review",
        ]

    def test_single_round_fails_without_revision(self, tmp_path: Path) -> None:
        writer = RecordingWriterAgent([make_article()])
        review = RecordingReviewAgent(["- Concern A"])
        mapping = RecordingMappingAgent([[make_mapping(1, "opinion")]])
        opinion = RecordingSpecialistAgent([make_verdict(1, misleading=True, status="REWRITE", suggested_fix="Fix")])
        orchestrator = build_test_orchestrator(
            tmp_path,
            writer=writer,
            review=review,
            mapping=mapping,
            opinion=opinion,
            editor_max_rounds=1,
        )

        result = _generate_article(orchestrator)

        assert result.success is False
        assert result.editor_report is not None
        assert result.editor_report.total_iterations == 1
        assert len(writer.generate_calls) == 1
        assert len(writer.revise_calls) == 0

    def test_specialists_dispatched_in_mapping_order(self, tmp_path: Path) -> None:
        dispatch_order: list[str] = []

        class OrderedSpecialist(RecordingSpecialistAgent):
            def __init__(self, name: str, verdicts: list[Verdict]) -> None:
                super().__init__(verdicts)
                self._name = name

            def evaluate(
                self,
                *,
                concern: Concern,
                article: ArticleResponse,
                source_text: str,
                source_metadata: dict[str, str | None],
                style_requirements: str,
            ) -> AgentResult[Verdict]:
                dispatch_order.append(self._name)
                return super().evaluate(
                    concern=concern,
                    article=article,
                    source_text=source_text,
                    source_metadata=source_metadata,
                    style_requirements=style_requirements,
                )

        review = RecordingReviewAgent(["- Concern A\n- Concern B\n- Concern C"])
        mapping = RecordingMappingAgent(
            [
                [
                    make_mapping(1, "fact_check"),
                    make_mapping(2, "opinion"),
                    make_mapping(3, "style_review"),
                ]
            ]
        )
        fact_check = OrderedSpecialist("fact_check", [make_verdict(1, misleading=False, status="KEEP")])
        opinion = OrderedSpecialist("opinion", [make_verdict(2, misleading=False, status="KEEP")])
        style_review = OrderedSpecialist("style_review", [make_verdict(3, misleading=False, status="KEEP")])
        orchestrator = build_test_orchestrator(
            tmp_path,
            review=review,
            mapping=mapping,
            fact_check=fact_check,
            opinion=opinion,
            style_review=style_review,
        )

        _generate_article(orchestrator)

        assert dispatch_order == ["fact_check", "opinion", "style_review"]


class TestOrchestratorSpecialistRouting:
    def test_routes_each_concern_to_correct_specialist(self, tmp_path: Path) -> None:
        review = RecordingReviewAgent(["- A\n- B\n- C\n- D\n- E"])
        mapping = RecordingMappingAgent(
            [
                [
                    make_mapping(1, "fact_check"),
                    make_mapping(2, "evidence_finding"),
                    make_mapping(3, "opinion"),
                    make_mapping(4, "attribution"),
                    make_mapping(5, "style_review"),
                ]
            ]
        )
        fact_check = RecordingSpecialistAgent([make_verdict(1, misleading=False, status="KEEP")])
        evidence = RecordingSpecialistAgent([make_verdict(2, misleading=False, status="KEEP")])
        opinion = RecordingSpecialistAgent([make_verdict(3, misleading=False, status="KEEP")])
        attribution = RecordingSpecialistAgent([make_verdict(4, misleading=False, status="KEEP")])
        style_review = RecordingSpecialistAgent([make_verdict(5, misleading=False, status="KEEP")])
        orchestrator = build_test_orchestrator(
            tmp_path,
            review=review,
            mapping=mapping,
            fact_check=fact_check,
            evidence=evidence,
            opinion=opinion,
            attribution=attribution,
            style_review=style_review,
        )

        _generate_article(orchestrator)

        assert len(fact_check.evaluate_calls) == 1
        assert fact_check.evaluate_calls[0]["concern"].concern_id == 1
        assert len(evidence.evaluate_calls) == 1
        assert evidence.evaluate_calls[0]["concern"].concern_id == 2
        assert len(opinion.evaluate_calls) == 1
        assert opinion.evaluate_calls[0]["concern"].concern_id == 3
        assert len(attribution.evaluate_calls) == 1
        assert attribution.evaluate_calls[0]["concern"].concern_id == 4
        assert len(style_review.evaluate_calls) == 1
        assert style_review.evaluate_calls[0]["concern"].concern_id == 5

    def test_unknown_specialist_raises_value_error(self, tmp_path: Path) -> None:
        review = RecordingReviewAgent(["- Concern A"])
        unknown_mapping = ConcernMapping.model_construct(
            concern_id=1,
            concern_type="scope_expansion",
            selected_agent="unknown_agent",
            confidence="high",
            reason="Unsupported agent name",
        )
        mapping = RecordingMappingAgent([[unknown_mapping]])
        orchestrator = build_test_orchestrator(tmp_path, review=review, mapping=mapping)

        with pytest.raises(ValueError, match="Unknown specialist agent"):
            _generate_article(orchestrator)


class TestFeedbackCompilation:
    def test_feedback_compilation_deterministic(self, tmp_path: Path) -> None:
        writer = RecordingWriterAgent([make_article()])
        review = RecordingReviewAgent([""])
        orchestrator = build_test_orchestrator(tmp_path, writer=writer, review=review)
        verdicts = [
            make_verdict(1, misleading=False, status="KEEP"),
            make_verdict(2, misleading=True, status="REWRITE", suggested_fix="Fix 2"),
            make_verdict(3, misleading=True, status="REMOVE", suggested_fix="Remove 3"),
        ]

        feedback = orchestrator.compile_feedback(iteration=1, verdicts=verdicts)

        assert isinstance(feedback, WriterFeedback)
        assert writer.generate_calls == []
        assert writer.revise_calls == []
        assert review.review_calls == []

    def test_verdicts_sorted_by_concern_id(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)
        verdicts = [
            make_verdict(3, misleading=False, status="KEEP"),
            make_verdict(1, misleading=False, status="KEEP"),
            make_verdict(2, misleading=False, status="KEEP"),
        ]

        feedback = orchestrator.compile_feedback(iteration=1, verdicts=verdicts)

        assert [verdict.concern_id for verdict in feedback.verdicts] == [1, 2, 3]

    def test_todo_list_from_rewrite_remove(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)
        verdicts = [
            make_verdict(1, misleading=True, status="REWRITE", suggested_fix="Fix A"),
            make_verdict(2, misleading=False, status="KEEP"),
            make_verdict(3, misleading=True, status="REMOVE", suggested_fix="Remove C"),
        ]

        feedback = orchestrator.compile_feedback(iteration=1, verdicts=verdicts)

        assert feedback.todo_list == ["Fix A", "Remove C"]
        assert feedback.passed is False
        assert len(feedback.verdicts) == 3

    def test_null_fix_excluded_from_todo(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)
        verdicts = [
            make_verdict(1, misleading=True, status="REWRITE", suggested_fix=None),
            make_verdict(2, misleading=True, status="REWRITE", suggested_fix="Fix B"),
        ]

        feedback = orchestrator.compile_feedback(iteration=1, verdicts=verdicts)

        assert feedback.todo_list == ["Fix B"]

    def test_suggestions_limited_to_five(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)
        verdicts = [
            make_verdict(
                concern_id=index,
                misleading=False,
                status="KEEP",
                rationale=f"Reason {index}",
            )
            for index in [7, 1, 6, 2, 5, 3, 4]
        ]

        feedback = orchestrator.compile_feedback(iteration=1, verdicts=verdicts)

        assert len(feedback.improvement_suggestions) == 5
        assert feedback.improvement_suggestions == [
            "Reason 1",
            "Reason 2",
            "Reason 3",
            "Reason 4",
            "Reason 5",
        ]

    def test_rating_formula(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)
        verdicts = [
            make_verdict(1, misleading=True, status="REWRITE", suggested_fix="Fix 1"),
            make_verdict(2, misleading=True, status="REWRITE", suggested_fix="Fix 2"),
            make_verdict(3, misleading=False, status="KEEP"),
        ]

        feedback = orchestrator.compile_feedback(iteration=1, verdicts=verdicts)

        assert feedback.rating == 5

    def test_rating_clamped_to_minimum(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)
        rewrite_verdicts = [make_verdict(index, misleading=True, status="REWRITE", suggested_fix=f"Fix {index}") for index in range(1, 6)]
        keep_verdicts = [make_verdict(index, misleading=False, status="KEEP") for index in range(6, 9)]

        feedback = orchestrator.compile_feedback(
            iteration=1,
            verdicts=rewrite_verdicts + keep_verdicts,
        )

        assert feedback.rating == 1

    def test_reasoning_empty_todos(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)
        verdicts = [
            make_verdict(1, misleading=False, status="KEEP"),
            make_verdict(2, misleading=False, status="KEEP"),
        ]

        feedback = orchestrator.compile_feedback(iteration=1, verdicts=verdicts)

        assert feedback.reasoning == "No required rewrites or removals were identified."

    def test_reasoning_with_todos(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)
        verdicts = [
            make_verdict(1, misleading=True, status="REWRITE", suggested_fix="Fix A"),
            make_verdict(2, misleading=True, status="REWRITE", suggested_fix="Fix B"),
        ]

        feedback = orchestrator.compile_feedback(iteration=1, verdicts=verdicts)

        assert feedback.reasoning == "Required changes:\n1. Fix A\n2. Fix B"


class TestOrchestratorArtifacts:
    def test_run_artifacts_directory_created(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)

        result = _generate_article(orchestrator)

        assert result.artifacts_dir is not None
        artifacts_dir = Path(result.artifacts_dir)
        assert artifacts_dir.exists()
        assert artifacts_dir.is_dir()
        assert artifacts_dir.parent == tmp_path / "runs" / "TestChannel" / "test-article"
        assert re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{8}", artifacts_dir.name)

    def test_artifact_files_sequentially_numbered(self, tmp_path: Path) -> None:
        review = RecordingReviewAgent(["- Concern A\n- Concern B"])
        mapping = RecordingMappingAgent([[make_mapping(1, "opinion"), make_mapping(2, "opinion")]])
        opinion = RecordingSpecialistAgent(
            [
                make_verdict(1, misleading=False, status="KEEP"),
                make_verdict(2, misleading=False, status="KEEP"),
            ]
        )
        orchestrator = build_test_orchestrator(
            tmp_path,
            review=review,
            mapping=mapping,
            opinion=opinion,
        )

        result = _generate_article(orchestrator)

        assert result.artifacts_dir is not None
        artifacts_dir = Path(result.artifacts_dir)
        files = sorted(path.name for path in artifacts_dir.iterdir() if path.is_file())
        prefixes = [int(match.group(1)) for name in files if (match := re.match(r"^(\d{3})_", name)) is not None]

        assert len(prefixes) == len(files)
        assert prefixes == sorted(prefixes)
        assert all(prefixes[index] < prefixes[index + 1] for index in range(len(prefixes) - 1))
        assert [prefix for prefix in prefixes if prefix >= 900] == [900, 901, 902]


class TestOrchestratorCanonicalOutput:
    def test_canonical_output_written_on_success(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)

        _generate_article(orchestrator)

        output_path = tmp_path / "articles" / "TestChannel" / "test-article.json"
        assert output_path.exists()
        with open(output_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert isinstance(payload, dict)

    def test_canonical_output_shape(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)

        _generate_article(orchestrator)

        output_path = tmp_path / "articles" / "TestChannel" / "test-article.json"
        with open(output_path, encoding="utf-8") as handle:
            payload = json.load(handle)

        assert set(payload.keys()) == {
            "success",
            "article",
            "metadata",
            "editor_report",
            "artifacts_dir",
            "error",
        }
        assert set(payload["article"].keys()) == {
            "headline",
            "alternative_headline",
            "article_body",
            "description",
        }
        assert set(payload["metadata"].keys()) == {
            "source_file",
            "channel_name",
            "video_id",
            "article_title",
            "slug",
            "publish_date",
            "references",
            "style_mode",
            "generated_at",
        }
        generated_at = payload["metadata"]["generated_at"]
        parsed = datetime.fromisoformat(generated_at)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)

    def test_canonical_output_written_on_failure(self, tmp_path: Path) -> None:
        review = RecordingReviewAgent(["- Concern A"])
        mapping = RecordingMappingAgent([[make_mapping(1, "opinion")]])
        opinion = RecordingSpecialistAgent([make_verdict(1, misleading=True, status="REWRITE", suggested_fix="Fix")])
        orchestrator = build_test_orchestrator(
            tmp_path,
            review=review,
            mapping=mapping,
            opinion=opinion,
            editor_max_rounds=1,
        )

        _generate_article(orchestrator)

        output_path = tmp_path / "articles" / "TestChannel" / "test-article.json"
        assert output_path.exists()
        with open(output_path, encoding="utf-8") as handle:
            payload = json.load(handle)

        assert payload["success"] is False
        assert payload["error"] is not None
        assert payload["article"] is not None
        assert payload["editor_report"]["final_status"] == "FAILED"


class TestOrchestratorMetadataValidation:
    def test_missing_channel_name_raises(self, tmp_path: Path) -> None:
        writer = RecordingWriterAgent([make_article()])
        review = RecordingReviewAgent([""])
        orchestrator = build_test_orchestrator(tmp_path, writer=writer, review=review)

        with pytest.raises(ValueError, match="Missing required metadata field: channel_name"):
            _generate_article(
                orchestrator,
                source_metadata=make_source_metadata(channel_name=None),
            )

        assert writer.generate_calls == []
        assert review.review_calls == []

    def test_missing_slug_raises(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)

        with pytest.raises(ValueError, match="Missing required metadata field: slug"):
            _generate_article(
                orchestrator,
                source_metadata=make_source_metadata(slug=None),
            )

    def test_missing_source_file_raises(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)

        with pytest.raises(ValueError, match="Missing required metadata field: source_file"):
            _generate_article(
                orchestrator,
                source_metadata=make_source_metadata(source_file=None),
            )

    def test_null_publish_date_accepted(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path)

        result = _generate_article(
            orchestrator,
            source_metadata=make_source_metadata(publish_date=None),
        )

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata.publish_date is None


class TestOrchestratorDependencyInjection:
    def test_accepts_mock_agents_via_protocol(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(
            tmp_path,
            writer=MockWriterAgent(),
            review=MockArticleReviewAgent(),
            mapping=MockConcernMappingAgent(),
            fact_check=MockFactCheckAgent(),
            evidence=MockEvidenceFindingAgent(),
            opinion=MockOpinionAgent(),
            attribution=MockAttributionAgent(),
            style_review=MockStyleReviewAgent(),
        )

        result = _generate_article(orchestrator)

        assert result.success is True

    def test_factory_creates_mock_agents(self, tmp_path: Path) -> None:
        config = _make_test_config(tmp_path, editor_max_rounds=3)

        orchestrator = build_chief_editor_orchestrator(config=config)

        assert isinstance(orchestrator.writer_agent, MockWriterAgent)
        assert isinstance(orchestrator.article_review_agent, MockArticleReviewAgent)
        assert isinstance(orchestrator.concern_mapping_agent, MockConcernMappingAgent)
        assert isinstance(orchestrator.fact_check_agent, MockFactCheckAgent)
        assert isinstance(orchestrator.evidence_finding_agent, MockEvidenceFindingAgent)
        assert isinstance(orchestrator.opinion_agent, MockOpinionAgent)
        assert isinstance(orchestrator.attribution_agent, MockAttributionAgent)
        assert isinstance(orchestrator.style_review_agent, MockStyleReviewAgent)

        result = _generate_article(orchestrator)
        assert result.success is True

    def test_factory_rejects_invalid_agent_name(self, tmp_path: Path) -> None:
        config = _make_test_config(tmp_path, editor_max_rounds=3, writer_agent_name="invalid")

        with pytest.raises(ValueError, match="Unknown writer agent_name"):
            build_chief_editor_orchestrator(config=config)


class TestOrchestratorErrorHandling:
    def test_agent_error_propagates(self, tmp_path: Path) -> None:
        orchestrator = build_test_orchestrator(tmp_path, writer=ExplodingWriterAgent())

        with pytest.raises(RuntimeError, match="LLM failed"):
            _generate_article(orchestrator)

    def test_non_bullet_review_raises(self, tmp_path: Path) -> None:
        review = RecordingReviewAgent(["This is not a bullet list"])
        orchestrator = build_test_orchestrator(tmp_path, review=review)

        with pytest.raises(ValueError, match="contains no markdown bullets"):
            _generate_article(orchestrator)

    def test_unknown_concern_id_raises(self, tmp_path: Path) -> None:
        review = RecordingReviewAgent(["- Concern A"])
        mapping = RecordingMappingAgent([[make_mapping(99, "opinion")]])
        orchestrator = build_test_orchestrator(tmp_path, review=review, mapping=mapping)

        with pytest.raises(ValueError, match="Concern id from mapping not found: 99"):
            _generate_article(orchestrator)
