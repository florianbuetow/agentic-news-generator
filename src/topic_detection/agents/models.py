"""Pydantic models for topic extraction agent."""

from pydantic import BaseModel, ConfigDict, Field


class TopicDetectionResult(BaseModel):
    """Result of topic detection for a text segment.

    Attributes:
        should_index: Whether the segment has substantive content worth indexing.
        high_level_topics: Broad category topics (e.g., "AI", "Technology", "Business").
        mid_level_topics: Specific domain topics (e.g., "Large Language Models", "AI Safety").
        specific_topics: Concrete named entities, products, methods, standards, etc.
        keywords: Additional searchable phrases from the segment.
        entities: Proper names explicitly mentioned (people, orgs, products, places, laws).
        description: 1-2 sentence description of the segment content.
    """

    should_index: bool = Field(
        ...,
        description="Whether the segment has substantive content worth indexing",
    )
    high_level_topics: list[str] = Field(
        ...,
        description="Broad domain categories (1-2)",
    )
    mid_level_topics: list[str] = Field(
        ...,
        description="Specific subdomains (2-5)",
    )
    specific_topics: list[str] = Field(
        ...,
        description="Concrete named entities, products, methods, standards, etc. (3-10)",
    )
    keywords: list[str] = Field(
        ...,
        description="Additional searchable phrases from the segment (5-15)",
    )
    entities: list[str] = Field(
        ...,
        description="Proper names explicitly mentioned (0-10)",
    )
    description: str = Field(
        ...,
        description="1-2 sentence description of the segment content",
        min_length=1,
    )

    model_config = ConfigDict(frozen=True, extra="forbid")
