"""Main orchestrator for the topic segmentation pipeline.

This module provides the SlidingWindowTopicSegmenter class, which
wires together all pipeline components and exposes a clean public API.
"""

import numpy as np
from numpy.typing import NDArray

from src.topic_detection.embedding.lmstudio import LMStudioEmbeddingGenerator
from src.topic_detection.segmentation.boundary_detector import BoundaryDetector
from src.topic_detection.segmentation.chunk_encoder import ChunkEncoder
from src.topic_detection.segmentation.data_types import Segment, SegmentationResult
from src.topic_detection.segmentation.segment_assembler import SegmentAssembler
from src.topic_detection.segmentation.similarity_calculator import SimilarityCalculator
from src.topic_detection.segmentation.tokenizer import Tokenizer


class SlidingWindowTopicSegmenter:
    """Unsupervised topic segmenter using embedding-based sliding window analysis.

    This class orchestrates a pipeline that:
    1. Tokenizes input text into words (word-level, not BPE)
    2. Creates overlapping chunks of words
    3. Embeds each chunk using the injected EmbeddingGenerator
    4. Computes similarity between adjacent chunks
    5. Detects topic boundaries where similarity drops
    6. Assembles final text segments

    Args:
        embedding_generator: Generator for creating embeddings.
        window_size: Number of words per chunk for embedding.
        stride: Number of words to advance between chunks.
        threshold_method: Boundary detection method:
                         - "relative": depth scoring at local minima
                         - "absolute": fixed similarity threshold
                         - "percentile": bottom N percentile of scores
        threshold_value: Threshold parameter (meaning depends on method).
        min_segment_tokens: Minimum words per segment (prevents over-segmentation).
        smoothing_passes: Number of smoothing iterations on similarity curve.
    """

    def __init__(
        self,
        embedding_generator: LMStudioEmbeddingGenerator,
        window_size: int,
        stride: int,
        threshold_method: str,
        threshold_value: float,
        min_segment_tokens: int,
        smoothing_passes: int,
    ) -> None:
        self.window_size = window_size
        self.stride = stride

        # Initialize pipeline components
        self.tokenizer = Tokenizer()
        self.encoder = ChunkEncoder(
            embedding_generator=embedding_generator,
            window_size=window_size,
            stride=stride,
        )
        self.similarity_calculator = SimilarityCalculator(smoothing_passes=smoothing_passes)
        self.boundary_detector = BoundaryDetector(method=threshold_method, threshold=threshold_value)
        self.segment_assembler = SegmentAssembler(tokenizer=self.tokenizer, min_segment_tokens=min_segment_tokens)

    def segment(self, text: str) -> SegmentationResult:
        """Segment text into topically coherent sections.

        Args:
            text: Input text to segment.

        Returns:
            SegmentationResult containing segments, boundaries, and scores.
        """
        # Handle empty input
        if not text or not text.strip():
            return SegmentationResult(
                segments=[],
                boundary_indices=[],
                similarity_scores=np.array([]),
                num_tokens=0,
            )

        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        num_tokens = len(tokens)

        # Handle text too short for sliding window
        if num_tokens < 2 * self.window_size:
            return self._single_segment_result(text, tokens)

        # Run pipeline
        chunk_data = self.encoder.encode(tokens)
        similarity_data = self.similarity_calculator.calculate(chunk_data)
        boundary_data = self.boundary_detector.detect(similarity_data)
        segments = self.segment_assembler.assemble(tokens, boundary_data)

        return SegmentationResult(
            segments=segments,
            boundary_indices=boundary_data.boundary_indices,
            similarity_scores=similarity_data.scores,
            num_tokens=num_tokens,
            chunk_data=chunk_data,
        )

    def get_similarity_scores(self, text: str) -> NDArray[np.float64]:
        """Get the similarity curve for visualization/debugging.

        Runs the pipeline up to similarity calculation and returns
        the scores. Useful for plotting and understanding boundary decisions.

        Args:
            text: Input text to analyze.

        Returns:
            numpy array of similarity scores between adjacent chunks.
        """
        if not text or not text.strip():
            return np.array([])

        tokens = self.tokenizer.tokenize(text)

        if len(tokens) < 2 * self.window_size:
            return np.array([])

        chunk_data = self.encoder.encode(tokens)
        similarity_data = self.similarity_calculator.calculate(chunk_data)

        return similarity_data.scores

    def segment_to_strings(self, text: str) -> list[str]:
        """Convenience method that returns just the segment texts.

        Args:
            text: Input text to segment.

        Returns:
            List of segment text strings.
        """
        result = self.segment(text)
        return [seg.text for seg in result.segments]

    def _single_segment_result(self, text: str, tokens: list[str]) -> SegmentationResult:
        """Create a result with the entire text as a single segment.

        Used when the text is too short for meaningful segmentation.

        Args:
            text: Original text.
            tokens: Tokenized text.

        Returns:
            SegmentationResult with one segment.
        """
        segment = Segment(
            text=text,
            start_token=0,
            end_token=len(tokens),
            token_count=len(tokens),
        )

        return SegmentationResult(
            segments=[segment],
            boundary_indices=[],
            similarity_scores=np.array([]),
            num_tokens=len(tokens),
        )
