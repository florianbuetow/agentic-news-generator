"""Tests for the specialist base-module re-export."""

from __future__ import annotations

import pytest

from src.agents.article_generation.base import BaseSpecialistAgent as RootBaseSpecialistAgent
from src.agents.article_generation.models import AgentResult, ArticleResponse, Concern, Verdict
from src.agents.article_generation.specialists.base import BaseSpecialistAgent
from src.config import LLMConfig


class _DummyConfig:
    def getEncodingName(self) -> str:
        return "o200k_base"


class _DummyLLMClient:
    def complete(self, *, llm_config: LLMConfig, messages: list[dict[str, str]]) -> str:
        return ""


class _ConcreteSpecialist(BaseSpecialistAgent):
    def evaluate(
        self,
        *,
        concern: Concern,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_requirements: str,
    ) -> AgentResult[Verdict]:
        return AgentResult(
            prompt="[test]",
            output=Verdict(
                concern_id=concern.concern_id,
                misleading=False,
                status="KEEP",
                rationale="ok",
                suggested_fix=None,
                evidence=None,
                citations=None,
            ),
        )


def _make_specialist() -> _ConcreteSpecialist:
    return _ConcreteSpecialist(
        llm_config=LLMConfig(
            model="test-model",
            api_base="http://127.0.0.1:1234/v1",
            api_key="lm-studio",
            context_window=32768,
            max_tokens=256,
            temperature=0.2,
            context_window_threshold=90,
            max_retries=0,
            retry_delay=0.25,
            timeout_seconds=30,
        ),
        config=_DummyConfig(),  # type: ignore[arg-type]
        llm_client=_DummyLLMClient(),  # type: ignore[arg-type]
    )


def test_module_reexports_root_base_specialist_agent() -> None:
    assert BaseSpecialistAgent is RootBaseSpecialistAgent
    assert "BaseSpecialistAgent" in BaseSpecialistAgent.__name__


def test_reexported_base_exposes_helper_methods() -> None:
    specialist = _make_specialist()

    assert specialist.normalize_query("  Mixed \n  CASE\tQuery  ") == "mixed case query"
    assert specialist.build_article_id({"source_file": "source.txt", "topic_slug": "topic"}) == "source.txt:topic"


def test_build_article_id_requires_both_metadata_fields() -> None:
    specialist = _make_specialist()

    with pytest.raises(ValueError, match="source_file"):
        specialist.build_article_id({"topic_slug": "topic"})

    with pytest.raises(ValueError, match="topic_slug"):
        specialist.build_article_id({"source_file": "source.txt"})
