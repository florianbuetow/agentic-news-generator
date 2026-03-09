"""Mock fact-check specialist agent for testing."""

from __future__ import annotations

import logging

from src.agents.article_generation.models import AgentResult, ArticleResponse, Concern, Verdict

logger = logging.getLogger(__name__)


class MockFactCheckAgent:
    """Returns static KEEP verdicts without KB or LLM calls."""

    def evaluate(
        self,
        *,
        concern: Concern,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_requirements: str,
    ) -> AgentResult[Verdict]:
        """Return a static KEEP verdict."""
        logger.info("MockFactCheckAgent: returning static KEEP for concern #%d", concern.concern_id)
        return AgentResult(
            prompt="[mock]",
            output=Verdict(
                concern_id=concern.concern_id,
                misleading=False,
                status="KEEP",
                rationale="Mock fact-check: no knowledge base available in test configuration",
                suggested_fix=None,
                evidence=None,
                citations=None,
            ),
        )
