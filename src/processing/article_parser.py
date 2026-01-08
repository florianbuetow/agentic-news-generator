"""Parser for markdown article files."""

import logging
import re
from pathlib import Path

import frontmatter
from markdown_it import MarkdownIt
from pydantic import ValidationError

from src.config import ArticleCompilerConfig
from src.models.article import ArticleFrontmatter, MarkdownArticle

logger = logging.getLogger(__name__)


class ArticleParser:
    """Parse markdown articles into MarkdownArticle objects."""

    def __init__(self, config: ArticleCompilerConfig) -> None:
        """Initialize the parser with configuration.

        Args:
            config: Article compiler configuration.
        """
        self.config = config
        self.md = MarkdownIt()

    def parse_file(self, file_path: Path) -> MarkdownArticle:
        """Parse single markdown file into MarkdownArticle.

        Args:
            file_path: Path to markdown file.

        Returns:
            Parsed MarkdownArticle instance.

        Raises:
            ValueError: If file cannot be parsed or required fields are missing.
        """
        # Load file with frontmatter
        try:
            post = frontmatter.load(str(file_path))
        except Exception as e:
            raise ValueError(f"Failed to load file {file_path.name}: {e}") from e

        # Validate frontmatter
        try:
            article_frontmatter = ArticleFrontmatter.model_validate(post.metadata)
        except ValidationError as e:
            raise ValueError(f"Invalid frontmatter in {file_path.name}: {e}") from e

        # Extract content
        content = post.content

        # Get title, subtitle, and slug from frontmatter
        title = article_frontmatter.title
        subtitle = article_frontmatter.subtitle
        slug = article_frontmatter.slug

        # Extract dateline (e.g., **SAN FRANCISCO** —)
        dateline = self._extract_dateline(content)

        # Extract paragraphs
        paragraphs = self._extract_paragraphs(content)

        # Extract images
        images = self._extract_images(content)

        return MarkdownArticle(
            filename=file_path.name,
            frontmatter=article_frontmatter,
            title=title,
            subtitle=subtitle,
            slug=slug,
            dateline=dateline,
            paragraphs=paragraphs,
            images=images,
        )

    def parse_directory(self, directory: Path) -> list[MarkdownArticle]:
        """Parse all markdown files in directory.

        Args:
            directory: Directory containing markdown files.

        Returns:
            List of successfully parsed MarkdownArticle instances.
        """
        articles: list[MarkdownArticle] = []

        markdown_files = list(directory.glob("*.md"))
        logger.info(f"Found {len(markdown_files)} markdown files in {directory}")

        for md_file in markdown_files:
            try:
                article = self.parse_file(md_file)
                articles.append(article)
                logger.debug(f"Successfully parsed: {md_file.name}")
            except Exception as e:
                logger.warning(f"Skipping {md_file.name}: {e}")
                continue

        logger.info(f"Successfully parsed {len(articles)}/{len(markdown_files)} articles")
        return articles

    def _extract_dateline(self, content: str) -> str:
        """Extract dateline city from **CITY** — pattern.

        Args:
            content: Markdown content.

        Returns:
            Dateline city, or empty if not found.
        """
        pattern = r"\*\*([A-Z][A-Z\s,\.]+)\*\*\s*—"
        match = re.search(pattern, content)
        return match.group(1).strip() if match else ""

    def _extract_paragraphs(self, content: str) -> list[str]:
        """Extract clean text paragraphs from markdown.

        Args:
            content: Markdown content.

        Returns:
            List of paragraph strings.
        """
        paragraphs: list[str] = []

        # Parse markdown into tokens
        tokens = self.md.parse(content)

        # Extract paragraph content
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "paragraph_open":
                # Next token should be inline with content
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    para_content = tokens[i + 1].content.strip()

                    # Skip dateline paragraph (contains ** and —)
                    if "**" in para_content and "—" in para_content:
                        i += 3  # Skip paragraph_open, inline, paragraph_close
                        continue

                    # Clean up markdown formatting
                    para_content = self._clean_markdown_text(para_content)

                    if para_content:
                        paragraphs.append(para_content)

                i += 3  # Skip paragraph_open, inline, paragraph_close
            else:
                i += 1

        return paragraphs

    def _clean_markdown_text(self, text: str) -> str:
        """Remove markdown formatting from text.

        Args:
            text: Text with markdown formatting.

        Returns:
            Cleaned text.
        """
        # Remove bold/italic markers
        text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)  # Bold italic
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)  # Italic

        # Remove inline code markers
        text = re.sub(r"`(.+?)`", r"\1", text)

        # Remove links but keep text
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)

        return text.strip()

    def _extract_images(self, content: str) -> list[str]:
        """Extract all image URLs from markdown.

        Args:
            content: Markdown content.

        Returns:
            List of image URLs.
        """
        pattern = r"!\[.*?\]\((https://[^\)]+)\)"
        return re.findall(pattern, content)
