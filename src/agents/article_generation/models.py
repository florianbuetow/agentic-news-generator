"""Pydantic models for article generation agent system."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ArticleResponse(BaseModel):
    """Response from the article writer agent (JSON output from LLM)."""

    headline: str = Field(
        ...,
        description="Primary title displayed prominently, matching page's H1 tag",
        min_length=10,
        max_length=200,
    )
    alternative_headline: str = Field(
        ...,
        alias="alternativeHeadline",
        description="Secondary or variant title for alternative phrasing or subtitles",
        min_length=10,
        max_length=200,
    )
    article_body: str = Field(
        ...,
        alias="articleBody",
        description="Full article in Markdown format with \\n line breaks",
        min_length=100,
    )
    description: str = Field(
        ...,
        description="Short teaser summary (1-2 sentences) for snippets/search",
        min_length=20,
        max_length=500,
    )

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class ArticleMetadata(BaseModel):
    """Metadata about the generated article."""

    topic_slug: str = Field(..., description="Normalized topic identifier from source")
    topic_title: str = Field(..., description="Human-readable topic title from source")
    style_mode: Literal["NATURE_NEWS", "SCIAM_MAGAZINE"] = Field(..., description="Writing style used")
    target_length_words: str = Field(..., description="Target word count range (e.g., '900-1200')")
    source_channel: str = Field(..., description="YouTube channel name")
    source_video_id: str = Field(..., description="YouTube video ID")
    source_video_title: str = Field(..., description="Original video title")
    source_publish_date: str | None = Field(None, description="Video publish date if available")
    source_file: str = Field(..., description="Source topic file path (channel/filename)")
    generated_at: str = Field(..., description="ISO 8601 timestamp of article generation")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleGenerationResult(BaseModel):
    """Final result of the article generation process."""

    success: bool = Field(..., description="Whether article generation succeeded")
    article: ArticleResponse | None = Field(..., description="Generated article (None if failed)")
    metadata: ArticleMetadata | None = Field(..., description="Article metadata (None if failed)")
    error: str | None = Field(..., description="Error message if generation failed")

    model_config = ConfigDict(frozen=True, extra="forbid")
