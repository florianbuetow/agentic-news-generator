"""Mock style-review specialist agent for testing."""

from __future__ import annotations

import logging

from src.agents.article_generation.models import AgentResult, ArticleResponse, Concern, Verdict

logger = logging.getLogger(__name__)


class MockStyleReviewAgent:
    """Returns static KEEP verdicts without LLM calls."""

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
        logger.info("MockStyleReviewAgent: returning static KEEP for concern #%d", concern.concern_id)
        return AgentResult(
            prompt="[mock]",
            output=Verdict(
                concern_id=concern.concern_id,
                misleading=False,
                status="KEEP",
                rationale="Mock style-review: no LLM available in test configuration",
                suggested_fix=None,
                evidence=None,
                citations=None,
            ),
        )
