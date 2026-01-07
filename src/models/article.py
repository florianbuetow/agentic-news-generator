"""Pydantic models for article data structures."""

import datetime

from pydantic import BaseModel, ConfigDict, Field


class ArticleFrontmatter(BaseModel):
    """Parsed frontmatter from markdown article."""

    title: str = Field(..., description="Article title")
    subtitle: str = Field(..., description="Article subtitle")
    slug: str = Field(..., description="URL slug")
    author: str = Field(..., description="Article author")
    date: datetime.date = Field(..., description="Publication date")
    tags: list[str] = Field(..., description="Article tags")
    summary: str = Field(..., description="Article summary")

    model_config = ConfigDict(frozen=True)


class MarkdownArticle(BaseModel):
    """Parsed markdown article with frontmatter and content."""

    filename: str = Field(..., description="Source filename")
    frontmatter: ArticleFrontmatter
    title: str = Field(..., description="Article title from frontmatter")
    subtitle: str = Field(..., description="Article subtitle from frontmatter")
    slug: str = Field(..., description="URL slug from frontmatter")
    dateline: str = ""
    paragraphs: list[str] = Field(..., description="Body paragraphs")
    images: list[str] = Field(default_factory=list, description="Image URLs")

    model_config = ConfigDict(frozen=False)

    @property
    def section(self) -> str:
        """Get section from first tag."""
        return self.frontmatter.tags[0] if self.frontmatter.tags else "General"

    @property
    def byline(self) -> str:
        """Get formatted byline (uppercase)."""
        return self.frontmatter.author.upper()


class HeroArticle(BaseModel):
    """Hero article format for articles.js."""

    image: str
    caption_label: str = Field(..., serialization_alias="captionLabel")
    caption: str
    headline: str
    subhead: str
    byline: str
    dateline: str
    paragraphs: list[str]
    link: str

    model_config = ConfigDict(frozen=False)


class FeaturedArticle(BaseModel):
    """Featured article format."""

    section: str
    headline: str
    subhead: str | None = None
    byline: str
    text: str
    link: str
    image: str | None = None
    size: str | None = None

    model_config = ConfigDict(frozen=False)


class SecondaryMain(BaseModel):
    """Secondary main article format."""

    image: str
    headline: str
    byline: str
    dateline: str
    paragraphs: list[str]
    link: str

    model_config = ConfigDict(frozen=False)


class SidebarArticle(BaseModel):
    """Sidebar article format."""

    headline: str
    byline: str | None = None
    text: str

    model_config = ConfigDict(frozen=False)


class BriefItem(BaseModel):
    """Brief item format."""

    headline: str
    text: str

    model_config = ConfigDict(frozen=False)


class BriefColumn(BaseModel):
    """Brief column containing multiple items."""

    section: str
    items: list[BriefItem]

    model_config = ConfigDict(frozen=False)
