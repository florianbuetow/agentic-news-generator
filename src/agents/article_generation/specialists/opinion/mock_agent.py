"""Mock opinion specialist agent for testing."""

from __future__ import annotations

import logging

from src.agents.article_generation.models import ArticleResponse, Concern, Verdict

logger = logging.getLogger(__name__)


class MockOpinionAgent:
    """Returns static KEEP verdicts without LLM calls."""

    def evaluate(
        self,
        *,
        concern: Concern,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_requirements: str,
    ) -> Verdict:
        """Return a static KEEP verdict."""
        logger.info("MockOpinionAgent: returning static KEEP for concern #%d", concern.concern_id)
        return Verdict(
            concern_id=concern.concern_id,
            misleading=False,
            status="KEEP",
            rationale="Mock opinion: no LLM available in test configuration",
            suggested_fix=None,
            evidence=None,
            citations=None,
        )
