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

    def test_whitespace_only_returns_empty(self) -> None:
        parser = ArticleReviewBulletParser()
        result = parser.parse(markdown_bullets="  \n\t\n  ")
        assert result.concerns == []

    def test_dash_and_asterisk_bullets(self) -> None:
        parser = ArticleReviewBulletParser()
        result = parser.parse(markdown_bullets="- First\n* Second")

        assert len(result.concerns) == 2
        assert result.concerns[0].review_note == "First"
        assert result.concerns[1].review_note == "Second"

    def test_concern_ids_sequential(self) -> None:
        parser = ArticleReviewBulletParser()
        result = parser.parse(markdown_bullets="- A\n- B\n- C\n- D\n- E")

        assert [concern.concern_id for concern in result.concerns] == [1, 2, 3, 4, 5]

    def test_review_note_is_full_bullet(self) -> None:
        parser = ArticleReviewBulletParser()
        result = parser.parse(markdown_bullets='- **"Excerpt"** -- explanation')

        assert len(result.concerns) == 1
        assert result.concerns[0].review_note == '**"Excerpt"** -- explanation'
        assert result.concerns[0].excerpt == "Excerpt"

    def test_numbered_list_raises(self) -> None:
        parser = ArticleReviewBulletParser()
        with pytest.raises(ValueError):
            parser.parse(markdown_bullets="1. First\n2. Second")

    def test_preamble_before_bullets_ignored(self) -> None:
        parser = ArticleReviewBulletParser()
        result = parser.parse(markdown_bullets="Here are concerns:\n\n- First\n- Second")

        assert len(result.concerns) == 2
        assert result.concerns[0].review_note == "First"
        assert result.concerns[1].review_note == "Second"
        assert all("Here are concerns" not in concern.review_note for concern in result.concerns)

    def test_nested_sub_bullets_as_continuation(self) -> None:
        parser = ArticleReviewBulletParser()
        result = parser.parse(markdown_bullets="- Parent\n  - Sub one\n  - Sub two\n- Next")

        assert len(result.concerns) == 2
        assert "Parent" in result.concerns[0].review_note
        assert "Sub one" in result.concerns[0].review_note
        assert "Sub two" in result.concerns[0].review_note
        assert result.concerns[1].review_note == "Next"

    def test_formatting_preserved(self) -> None:
        parser = ArticleReviewBulletParser()
        result = parser.parse(markdown_bullets="- **Bold** and *italic* text")

        assert len(result.concerns) == 1
        assert result.concerns[0].review_note == "**Bold** and *italic* text"
