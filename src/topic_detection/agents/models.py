"""Pydantic models for topic extraction agent."""

from pydantic import BaseModel, ConfigDict, Field


class TopicDetectionResult(BaseModel):
    """Result of topic detection for a text segment.

    Attributes:
        high_level_topics: Broad category topics (e.g., "AI", "Technology", "Business").
        mid_level_topics: Specific domain topics (e.g., "Large Language Models", "AI Safety").
        specific_topics: Particular events/products/concepts (e.g., "GPT-4 vision capabilities").
        description: 1-2 sentence description of the segment content.
    """

    high_level_topics: list[str] = Field(
        ...,
        description="Broad category topics (1-2)",
        min_length=1,
    )
    mid_level_topics: list[str] = Field(
        ...,
        description="Specific domain topics (1-3)",
        min_length=1,
    )
    specific_topics: list[str] = Field(
        ...,
        description="Particular events, products, or concepts discussed (1-3)",
        min_length=1,
    )
    description: str = Field(
        ...,
        description="1-2 sentence description of the segment content",
        min_length=1,
    )

    model_config = ConfigDict(frozen=True, extra="forbid")
