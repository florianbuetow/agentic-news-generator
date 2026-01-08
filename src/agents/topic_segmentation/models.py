"""Pydantic models for topic segmentation agent system."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SRTSubtitle(BaseModel):
    """Single SRT subtitle entry."""

    index: int = Field(..., description="Subtitle sequence number")
    start_ms: int = Field(..., description="Start time in milliseconds")
    end_ms: int = Field(..., description="End time in milliseconds")
    text: str = Field(..., description="Subtitle text content")

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicBlock(BaseModel):
    """Single topic segment from a video."""

    id: int = Field(..., description="Segment ID (1-indexed)")
    start: str = Field(..., description="Start timestamp in SRT format (HH:MM:SS,mmm)")
    end: str = Field(..., description="End timestamp in SRT format (HH:MM:SS,mmm)")
    topic: str = Field(..., description="Topic (lowercase-with-hyphens)")
    summary: str = Field(..., description="AI-generated summary of segment")

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicDocument(BaseModel):
    """Complete topic with aggregated blocks."""

    topic_slug: str = Field(..., description="Normalized topic identifier")
    topic_title: str = Field(..., description="Human-readable topic title")
    blocks: list[TopicBlock] = Field(..., description="Topic segments")

    model_config = ConfigDict(frozen=True, extra="forbid")


class AgentSegmentationResponse(BaseModel):
    """Response from the segmentation agent."""

    segments: list[TopicBlock] = Field(..., description="Identified topic segments")

    model_config = ConfigDict(frozen=True, extra="forbid")


class CriticRating(BaseModel):
    """Quality rating from critic agent."""

    rating: Literal["bad", "ok", "great"] = Field(..., description="Quality assessment")
    pass_: bool = Field(..., alias="pass", description="Whether segmentation passes quality standards")
    reasoning: str = Field(..., description="Detailed explanation of rating")
    improvement_suggestions: str = Field(..., description="Specific suggestions for improvement")

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class SegmentationAttempt(BaseModel):
    """Single segmentation attempt."""

    attempt_number: int = Field(..., description="Attempt counter (1-indexed)")
    response: AgentSegmentationResponse = Field(..., description="Agent response")
    critic_rating: CriticRating | None = Field(..., description="Critic evaluation (None if not yet evaluated)")

    model_config = ConfigDict(frozen=True, extra="forbid")


class SegmentationResult(BaseModel):
    """Final result of the segmentation process."""

    success: bool = Field(..., description="Whether segmentation succeeded")
    attempts: list[SegmentationAttempt] = Field(..., description="All segmentation attempts")
    best_attempt: SegmentationAttempt | None = Field(..., description="Best attempt (highest rated)")
    failure_reason: str | None = Field(..., description="Reason for failure if unsuccessful")

    model_config = ConfigDict(frozen=True, extra="forbid")
