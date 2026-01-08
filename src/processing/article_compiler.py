"""Compiler for transforming parsed articles into output format."""

import logging
from typing import Any

from src.config import ArticleCompilerConfig
from src.models.article import HeroArticle, MarkdownArticle, SecondaryMain

logger = logging.getLogger(__name__)


class ArticleCompiler:
    """Compile parsed articles into output format."""

    def __init__(self, config: ArticleCompilerConfig) -> None:
        """Initialize compiler with configuration."""
        self.config = config

    def compile(self, articles: list[MarkdownArticle]) -> dict[str, Any]:
        """Compile articles into output structure.

        Sorts by date (newest first), validates count, assigns to positions, and transforms to output format.
        """
        # Sort by date descending and validate
        sorted_articles = sorted(articles, key=lambda a: a.frontmatter.date, reverse=True)

        if len(sorted_articles) < self.config.min_articles:
            raise ValueError(f"Need {self.config.min_articles} articles, found {len(sorted_articles)}")

        # Assign to positions: hero (1), featured (5), secondary (1), sidebar (6), briefs (8)
        return {
            "hero": self._to_hero(sorted_articles[0]).model_dump(by_alias=True),
            "featured": [self._to_featured(a, i == 0) for i, a in enumerate(sorted_articles[1:6])],
            "secondary": self._to_secondary(sorted_articles[6]).model_dump(by_alias=True),
            "sidebar": [
                [self._to_sidebar(a) for a in sorted_articles[7:10]],
                [self._to_sidebar(a) for a in sorted_articles[10:13]],
            ],
            "briefs": self._to_briefs(sorted_articles[13:21]),
        }

    def _to_hero(self, a: MarkdownArticle) -> HeroArticle:
        """Convert to hero article format."""
        # Get image or raise error if none available
        if a.images and self.config.images.extract_first:
            image = a.images[0]
        elif self.config.images.fallback_url:
            image = self.config.images.fallback_url
        else:
            raise ValueError(f"No image found for hero article: {a.filename}")

        # Get paragraphs with fallback
        paragraphs = a.paragraphs[: self.config.paragraphs.hero_count]
        if not paragraphs:
            paragraphs = [a.frontmatter.summary]

        return HeroArticle(
            image=image,
            caption_label=f"{a.section.upper()}:",
            caption=a.frontmatter.summary,
            headline=a.title,
            subhead=a.subtitle,
            byline=a.byline,
            dateline=a.dateline,
            paragraphs=paragraphs,
            link=self._link(a.slug),
        )

    def _to_featured(self, a: MarkdownArticle, is_large: bool) -> dict[str, Any]:
        """Convert to featured article format."""
        para_count = self.config.paragraphs.featured_count
        text = " ".join(a.paragraphs[:para_count]) if a.paragraphs else a.frontmatter.summary

        result: dict[str, Any] = {
            "section": a.section,
            "headline": a.title,
            "byline": a.byline,
            "text": text,
            "link": self._link(a.slug),
        }

        if is_large:
            result["size"] = "large"
            result["subhead"] = a.subtitle
            result["image"] = a.images[0] if a.images else None
        elif a.images:
            result["image"] = a.images[0]

        return result

    def _to_secondary(self, a: MarkdownArticle) -> SecondaryMain:
        """Convert to secondary main format."""
        # Get image or raise error if none available
        if a.images and self.config.images.extract_first:
            image = a.images[0]
        elif self.config.images.fallback_url:
            image = self.config.images.fallback_url
        else:
            raise ValueError(f"No image found for secondary article: {a.filename}")

        # Get paragraphs with fallback
        paragraphs = a.paragraphs[: self.config.paragraphs.secondary_count]
        if not paragraphs:
            paragraphs = [a.frontmatter.summary]

        return SecondaryMain(
            image=image,
            headline=a.title,
            byline=a.byline,
            dateline=a.dateline,
            paragraphs=paragraphs,
            link=self._link(a.slug),
        )

    def _to_sidebar(self, a: MarkdownArticle) -> dict[str, Any]:
        """Convert to sidebar article format."""
        result = {"headline": a.title, "text": a.frontmatter.summary, "link": self._link(a.slug)}
        if len(a.byline) < 30:
            result["byline"] = a.byline
        return result

    def _to_briefs(self, articles: list[MarkdownArticle]) -> list[dict[str, Any]]:
        """Convert 8 articles to 4 brief columns with 2 items each."""
        sections = ["National", "International", "Business", "Arts & Culture"]
        columns: list[dict[str, Any]] = []

        for i in range(4):
            start_idx = i * 2
            items = [
                {"headline": a.title, "text": a.frontmatter.summary, "link": self._link(a.slug)}
                for a in articles[start_idx : start_idx + 2]
            ]
            columns.append({"section": sections[i], "items": items})

        return columns

    def _link(self, slug: str) -> str:
        """Generate article link from slug."""
        return f"{self.config.links.base_path}{slug}"
