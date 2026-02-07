"""Deterministic parser for article-review markdown bullet output."""

from __future__ import annotations

import logging
import re

from src.agents.article_generation.models import ArticleReviewResult, Concern

logger = logging.getLogger(__name__)


class ArticleReviewBulletParser:
    """Convert markdown bullet text into structured concerns."""

    _CURLY_QUOTE_PATTERN = re.compile(r"“([^”]+)”")
    _STRAIGHT_QUOTE_PATTERN = re.compile(r'"([^"]+)"')

    def parse(self, *, markdown_bullets: str) -> ArticleReviewResult:
        """Parse markdown bullets into concerns.

        Raises:
            ValueError: If content is non-empty but no valid bullets are detected.
        """
        stripped_text = markdown_bullets.strip()
        if stripped_text == "":
            return ArticleReviewResult(concerns=[])

        bullets: list[str] = []
        current_lines: list[str] = []

        for raw_line in markdown_bullets.splitlines():
            if raw_line.startswith("- ") or raw_line.startswith("* "):
                if current_lines:
                    bullets.append("\n".join(current_lines).strip())
                current_lines = [raw_line[2:].strip()]
            else:
                if current_lines:
                    current_lines.append(raw_line.strip())

        if current_lines:
            bullets.append("\n".join(current_lines).strip())

        if len(bullets) == 0:
            raise ValueError("Article-review output is non-empty but contains no markdown bullets")

        concerns: list[Concern] = []
        for index, bullet in enumerate(bullets, start=1):
            excerpt = self._extract_excerpt(bullet_text=bullet)
            concerns.append(
                Concern(
                    concern_id=index,
                    excerpt=excerpt,
                    review_note=bullet,
                )
            )

        logger.info("Parsed %d concerns from %d markdown bullets", len(concerns), len(bullets))
        return ArticleReviewResult(concerns=concerns)

    def _extract_excerpt(self, *, bullet_text: str) -> str:
        """Extract excerpt using required quote precedence."""
        curly_match = self._CURLY_QUOTE_PATTERN.search(bullet_text)
        if curly_match is not None:
            return curly_match.group(1)

        straight_match = self._STRAIGHT_QUOTE_PATTERN.search(bullet_text)
        if straight_match is not None:
            return straight_match.group(1)

        return bullet_text
