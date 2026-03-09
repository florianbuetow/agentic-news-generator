"""Mock concern-mapping agent for testing."""

from __future__ import annotations

import logging

from src.agents.article_generation.models import AgentResult, Concern, ConcernMappingResult

logger = logging.getLogger(__name__)


class MockConcernMappingAgent:
    """Returns empty mappings without LLM calls."""

    def map_concerns(
        self,
        *,
        style_requirements: str,
        source_text: str,
        generated_article_json: str,
        concerns: list[Concern],
    ) -> AgentResult[ConcernMappingResult]:
        """Return empty mapping result."""
        logger.info("MockConcernMappingAgent: returning empty mappings")
        return AgentResult(prompt="[mock]", output=ConcernMappingResult(mappings=[]))
