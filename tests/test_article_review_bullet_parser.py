"""Tests for deterministic article-review bullet parser."""

import pytest

from src.agents.article_generation.chief_editor.bullet_parser import ArticleReviewBulletParser


class TestArticleReviewBulletParser:
    """Tests for markdown bullet parsing behavior."""

    def test_parse_empty_output(self) -> None:
        """Empty review output yields zero concerns."""
        parser = ArticleReviewBulletParser()
        result = parser.parse(markdown_bullets="  \n")
        assert result.concerns == []

    def test_parse_with_multiline_bullets_and_excerpt_extraction(self) -> None:
        """Parser handles multiline bullets and quote extraction precedence."""
        parser = ArticleReviewBulletParser()
        markdown = (
            '* **"First excerpt"** - note line one\n'
            "  continuation line\n"
            "- “Second excerpt” and extra explanation\n"
            "- No quoted excerpt here\n"
        )

        result = parser.parse(markdown_bullets=markdown)

        assert len(result.concerns) == 3
        assert result.concerns[0].concern_id == 1
        assert result.concerns[0].excerpt == "First excerpt"
        assert "continuation line" in result.concerns[0].review_note
        assert result.concerns[1].excerpt == "Second excerpt"
        assert result.concerns[2].excerpt == "No quoted excerpt here"

    def test_non_empty_without_bullets_raises(self) -> None:
        """Non-empty review output without bullets fails fast."""
        parser = ArticleReviewBulletParser()
        with pytest.raises(ValueError):
            parser.parse(markdown_bullets="This text has no list bullets")
