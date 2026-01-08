"""Unit tests for the ArticleParser class."""

import tempfile
from pathlib import Path

import pytest

from src.config import Config
from src.processing.article_parser import ArticleParser


@pytest.fixture
def article_parser() -> ArticleParser:
    """Create ArticleParser instance with test configuration."""
    config_path = Path("config/config.yaml")
    config = Config(config_path)
    compiler_config = config.get_article_compiler_config()
    return ArticleParser(compiler_config)


@pytest.fixture
def valid_article_content() -> str:
    """Return valid markdown article content with all required YAML fields."""
    return """---
title: Test Article Title
subtitle: This is a test subtitle
slug: test-article-title
author: John Doe
date: 2024-01-15
tags: [Technology, AI, Testing]
summary: This is a test article summary for testing purposes.
---

# Test Article Title

## This is a test subtitle

**SAN FRANCISCO** — This is the first paragraph of the test article with some content.

This is the second paragraph with more content to test the parser.

![Test Image](https://example.com/test-image.jpg)

This is the third paragraph after an image.
"""


@pytest.fixture
def missing_title_content() -> str:
    """Return article content missing title in YAML."""
    return """---
subtitle: This is a test subtitle
slug: test-article
author: John Doe
date: 2024-01-15
tags: [Technology]
summary: Test summary
---

# Test Article Title

This is some content.
"""


@pytest.fixture
def missing_subtitle_content() -> str:
    """Return article content missing subtitle in YAML."""
    return """---
title: Test Article Title
slug: test-article
author: John Doe
date: 2024-01-15
tags: [Technology]
summary: Test summary
---

# Test Article Title

This is some content.
"""


@pytest.fixture
def missing_slug_content() -> str:
    """Return article content missing slug in YAML."""
    return """---
title: Test Article Title
subtitle: Test subtitle
author: John Doe
date: 2024-01-15
tags: [Technology]
summary: Test summary
---

# Test Article Title

This is some content.
"""


class TestArticleParser:
    """Test cases for the ArticleParser class."""

    def test_parse_valid_article(self, article_parser: ArticleParser, valid_article_content: str) -> None:
        """Test parsing a valid article with all required YAML fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(valid_article_content)
            temp_path = Path(f.name)

        try:
            article = article_parser.parse_file(temp_path)

            # Verify title, subtitle, slug from frontmatter
            assert article.title == "Test Article Title"
            assert article.subtitle == "This is a test subtitle"
            assert article.slug == "test-article-title"

            # Verify other frontmatter fields
            assert article.frontmatter.author == "John Doe"
            assert article.frontmatter.tags == ["Technology", "AI", "Testing"]
            assert article.frontmatter.summary == "This is a test article summary for testing purposes."

            # Verify content extraction still works
            assert article.dateline == "SAN FRANCISCO"
            assert len(article.paragraphs) >= 2
            # The parser skips the dateline paragraph, so first paragraph in array is the "second paragraph"
            assert "second paragraph" in article.paragraphs[0]
            assert len(article.images) == 1
            assert "https://example.com/test-image.jpg" in article.images
        finally:
            temp_path.unlink()

    def test_parse_missing_title(self, article_parser: ArticleParser, missing_title_content: str) -> None:
        """Test that missing title in YAML raises ValidationError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(missing_title_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                article_parser.parse_file(temp_path)
            assert "Invalid frontmatter" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_parse_missing_subtitle(self, article_parser: ArticleParser, missing_subtitle_content: str) -> None:
        """Test that missing subtitle in YAML raises ValidationError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(missing_subtitle_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                article_parser.parse_file(temp_path)
            assert "Invalid frontmatter" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_parse_missing_slug(self, article_parser: ArticleParser, missing_slug_content: str) -> None:
        """Test that missing slug in YAML raises ValidationError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(missing_slug_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                article_parser.parse_file(temp_path)
            assert "Invalid frontmatter" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_parse_directory(self, article_parser: ArticleParser, valid_article_content: str) -> None:
        """Test parsing multiple articles from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple test articles
            for i in range(3):
                article_file = temp_path / f"test_article_{i}.md"
                with open(article_file, "w") as f:
                    f.write(valid_article_content)

            articles = article_parser.parse_directory(temp_path)

            assert len(articles) == 3
            for article in articles:
                assert article.title == "Test Article Title"
                assert article.subtitle == "This is a test subtitle"
                assert article.slug == "test-article-title"

    def test_parse_invalid_frontmatter(self, article_parser: ArticleParser) -> None:
        """Test that invalid YAML structure raises ValueError."""
        invalid_content = """---
title: Test
subtitle: Test
slug: test
author: Test
date: invalid-date-format
tags: [Test]
summary: Test
---

Content here.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(invalid_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                article_parser.parse_file(temp_path)
            assert "Invalid frontmatter" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_parse_file_not_found(self, article_parser: ArticleParser) -> None:
        """Test that non-existent file raises appropriate error."""
        non_existent_path = Path("/nonexistent/path/article.md")
        with pytest.raises(ValueError) as exc_info:
            article_parser.parse_file(non_existent_path)
        assert "Failed to load file" in str(exc_info.value)

    def test_parse_empty_file(self, article_parser: ArticleParser) -> None:
        """Test empty markdown file handling."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                article_parser.parse_file(temp_path)
            # Empty file will fail validation since required fields are missing
            assert "Invalid frontmatter" in str(exc_info.value) or "Failed to load" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_parse_directory_with_invalid_files(self, article_parser: ArticleParser, valid_article_content: str) -> None:
        """Test that parser skips invalid files and continues with valid ones."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create one valid article
            valid_file = temp_path / "valid_article.md"
            with open(valid_file, "w") as f:
                f.write(valid_article_content)

            # Create one invalid article (missing required field)
            invalid_file = temp_path / "invalid_article.md"
            with open(invalid_file, "w") as f:
                f.write("---\ntitle: Test\n---\n\nContent")

            articles = article_parser.parse_directory(temp_path)

            # Should successfully parse 1 out of 2 files
            assert len(articles) == 1
            assert articles[0].title == "Test Article Title"

    def test_article_properties(self, article_parser: ArticleParser, valid_article_content: str) -> None:
        """Test that article properties (section, byline) work correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(valid_article_content)
            temp_path = Path(f.name)

        try:
            article = article_parser.parse_file(temp_path)

            # Test section property (should be first tag)
            assert article.section == "Technology"

            # Test byline property (should be uppercase author)
            assert article.byline == "JOHN DOE"
        finally:
            temp_path.unlink()
