"""Pydantic models for topic extraction agent."""

from pydantic import BaseModel, ConfigDict, Field


class TopicDetectionResult(BaseModel):
    """Result of topic detection for a text segment.

    Attributes:
        topics: List of topics at multiple granularity levels:
                - High-level: "AI", "Technology", "Business"
                - Mid-level: "Large Language Models", "AI Safety", "Robotics"
                - Specific: "GPT-4 vision capabilities", "Anthropic Claude 3 release"
        description: 1-2 sentence description of the segment content.
    """

    topics: list[str] = Field(
        ...,
        description="List of topics at various granularity levels",
        min_length=1,
    )
    description: str = Field(
        ...,
        description="1-2 sentence description of the segment content",
        min_length=1,
    )

    model_config = ConfigDict(frozen=True, extra="forbid")
