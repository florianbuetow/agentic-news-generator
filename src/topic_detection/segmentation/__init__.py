"""Topic segmentation components for embedding-based boundary detection."""

from src.topic_detection.segmentation.segmenter import SlidingWindowTopicSegmenter

from src.topic_detection.segmentation.data_types import (
    BoundaryData,
    ChunkData,
    Segment,
    SegmentationResult,
    SimilarityData,
)

__all__ = [
    "BoundaryData",
    "ChunkData",
    "Segment",
    "SegmentationResult",
    "SimilarityData",
    "SlidingWindowTopicSegmenter",
]
