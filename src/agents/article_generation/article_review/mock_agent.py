"""Mock article-review agent for testing."""

from __future__ import annotations

import logging

from src.agents.article_generation.models import ArticleResponse, ArticleReviewRaw

logger = logging.getLogger(__name__)


class MockArticleReviewAgent:
    """Returns empty review (no concerns) without LLM calls."""

    def review(
        self,
        *,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
    ) -> ArticleReviewRaw:
        """Return empty review — signals no concerns found."""
        logger.info("MockArticleReviewAgent: returning empty review (no concerns)")
        return ArticleReviewRaw(markdown_bullets="")
