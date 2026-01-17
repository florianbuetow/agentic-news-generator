"""Main orchestrator for the topic segmentation pipeline.

This module provides the SlidingWindowTopicSegmenter class, which
wires together all pipeline components and exposes a clean public API.
"""

import numpy as np
from numpy.typing import NDArray

from src.topic_detection.embedding.lmstudio import LMStudioEmbeddingGenerator
from src.topic_detection.segmentation.boundary_detector import BoundaryDetector
from src.topic_detection.segmentation.chunk_encoder import ChunkEncoder
from src.topic_detection.segmentation.data_types import ChunkData, Segment, SegmentationResult
from src.topic_detection.segmentation.segment_assembler import SegmentAssembler
from src.topic_detection.segmentation.similarity_calculator import SimilarityCalculator
from src.topic_detection.segmentation.tokenizer import Tokenizer


class SlidingWindowTopicSegmenter:
    """Unsupervised topic segmenter using embedding-based sliding window analysis.

    This class orchestrates a pipeline that:
    1. Tokenizes input text into words
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
        smoothing_passes: Number of smoothing iterations on similarity curve.
    """

    def __init__(
        self,
        embedding_generator: LMStudioEmbeddingGenerator,
        window_size: int,
        stride: int,
        threshold_method: str,
        threshold_value: float,
        smoothing_passes: int,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.embedding_generator = embedding_generator

        # Initialize pipeline components
        self.tokenizer = Tokenizer()
        self.encoder = ChunkEncoder(
            embedding_generator=embedding_generator,
            window_size=window_size,
            stride=stride,
        )
        self.similarity_calculator = SimilarityCalculator(smoothing_passes=smoothing_passes)
        self.boundary_detector = BoundaryDetector(method=threshold_method, threshold=threshold_value)
        self.segment_assembler = SegmentAssembler(tokenizer=self.tokenizer)

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
                num_words=0,
            )

        # Tokenize
        words = self.tokenizer.tokenize(text)
        num_words = len(words)

        # Handle text too short for sliding window
        if num_words < 2 * self.window_size:
            return self._single_segment_result(text, words)

        # Run pipeline
        chunk_data = self.encoder.encode(words)
        similarity_data = self.similarity_calculator.calculate(chunk_data)
        boundary_data = self.boundary_detector.detect(similarity_data)
        segments = self.segment_assembler.assemble(words, boundary_data)

        return SegmentationResult(
            segments=segments,
            boundary_indices=boundary_data.boundary_indices,
            similarity_scores=similarity_data.scores,
            num_words=num_words,
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

        words = self.tokenizer.tokenize(text)

        if len(words) < 2 * self.window_size:
            return np.array([])

        chunk_data = self.encoder.encode(words)
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

    def generate_embeddings(self, text: str) -> tuple[list[str], ChunkData, bool]:
        """Generate embeddings for text without performing segmentation.

        Step 1 of the decoupled pipeline: tokenize and embed only.

        Args:
            text: Input text to process.

        Returns:
            Tuple of (words, chunk_data, is_short) where:
            - words: list of words
            - chunk_data: embeddings and chunk positions
            - is_short: True if text was shorter than window_size (warning)

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text is empty or contains only whitespace")

        words = self.tokenizer.tokenize(text)
        num_words = len(words)

        if num_words < self.window_size:
            # Short text: create single embedding for entire text
            full_text = self.tokenizer.detokenize(words)
            embedding = self.embedding_generator.generate([full_text])
            chunk_data = ChunkData(embeddings=embedding, chunk_positions=[0])
            return words, chunk_data, True

        chunk_data = self.encoder.encode(words)
        return words, chunk_data, False

    def segment_from_chunk_data(self, words: list[str], chunk_data: ChunkData) -> SegmentationResult:
        """Segment text using pre-computed embeddings.

        Step 2 of the decoupled pipeline: boundary detection from embeddings.

        Args:
            words: List of words (from generate_embeddings).
            chunk_data: Pre-computed embeddings (from generate_embeddings).

        Returns:
            SegmentationResult containing segments, boundaries, and scores.
        """
        num_words = len(words)

        # Handle text too short for meaningful segmentation
        if num_words < 2 * self.window_size:
            text = self.tokenizer.detokenize(words)
            return self._single_segment_result(text, words)

        # Run pipeline from similarity calculation onwards
        similarity_data = self.similarity_calculator.calculate(chunk_data)
        boundary_data = self.boundary_detector.detect(similarity_data)
        segments = self.segment_assembler.assemble(words, boundary_data)

        return SegmentationResult(
            segments=segments,
            boundary_indices=boundary_data.boundary_indices,
            similarity_scores=similarity_data.scores,
            num_words=num_words,
            chunk_data=chunk_data,
        )

    def _single_segment_result(self, text: str, words: list[str]) -> SegmentationResult:
        """Create a result with the entire text as a single segment.

        Used when the text is too short for meaningful segmentation.

        Args:
            text: Original text.
            words: Tokenized text.

        Returns:
            SegmentationResult with one segment.
        """
        segment = Segment(
            text=text,
            start_word=0,
            end_word=len(words),
            word_count=len(words),
        )

        return SegmentationResult(
            segments=[segment],
            boundary_indices=[],
            similarity_scores=np.array([]),
            num_words=len(words),
        )
