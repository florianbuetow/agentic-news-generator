"""Pydantic models for JSON serialization of pipeline outputs.

These schemas define the JSON structure for the three-step topic detection pipeline:
1. EmbeddingsOutput - Output of generate-embeddings.py
2. SegmentationOutput - Output of detect-boundaries.py
3. (TranscriptTopics is in extract-topics.py for topic extraction)
"""

from pydantic import BaseModel, ConfigDict


class SRTEntryData(BaseModel):
    """SRT entry with word position mapping."""

    index: int
    start_timestamp: str
    end_timestamp: str
    content: str
    word_start: int
    word_end: int

    model_config = ConfigDict(frozen=True, extra="forbid")


class EmbeddingConfigData(BaseModel):
    """Configuration used for embedding generation."""

    embedding_model: str
    window_size: int
    stride: int

    model_config = ConfigDict(frozen=True, extra="forbid")


class WindowData(BaseModel):
    """A single embedding window."""

    offset: int
    embedding: list[float]

    model_config = ConfigDict(frozen=True, extra="forbid")


class EmbeddingsOutput(BaseModel):
    """Complete output for _embeddings.json file."""

    source_file: str
    generated_at: str
    config: EmbeddingConfigData
    total_words: int
    words: list[str]
    srt_entries: list[SRTEntryData]
    windows: list[WindowData]

    model_config = ConfigDict(frozen=True, extra="forbid")


class SegmentationConfigData(BaseModel):
    """Configuration used for segmentation."""

    embedding_model: str
    window_size: int
    stride: int
    threshold_method: str
    threshold_value: float
    smoothing_passes: int

    model_config = ConfigDict(frozen=True, extra="forbid")


class SegmentData(BaseModel):
    """A single segment with timestamps and word positions."""

    segment_id: int
    start_timestamp: str
    end_timestamp: str
    text: str
    start_word: int
    end_word: int

    model_config = ConfigDict(frozen=True, extra="forbid")


class SegmentationOutput(BaseModel):
    """Complete output for _segmentation.json file."""

    source_file: str
    embeddings_file: str
    segmented_at: str
    config: SegmentationConfigData
    total_segments: int
    segments: list[SegmentData]

    model_config = ConfigDict(frozen=True, extra="forbid")
