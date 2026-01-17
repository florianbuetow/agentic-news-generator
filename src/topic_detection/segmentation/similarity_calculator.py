"""Similarity calculator for the topic segmentation pipeline.

Computes cosine similarity between adjacent chunk embeddings
and applies optional smoothing to the similarity curve.
"""

import numpy as np
from numpy.typing import NDArray

from src.topic_detection.segmentation.data_types import ChunkData, SimilarityData


class SimilarityCalculator:
    """Calculates similarity scores between adjacent chunks.

    Takes chunk embeddings and computes cosine similarity between
    each consecutive pair. Optionally applies smoothing to reduce noise.

    Args:
        smoothing_passes: Number of smoothing iterations to apply.
                         0 = no smoothing, higher = smoother curve.

    Example:
        >>> calc = SimilarityCalculator(smoothing_passes=2)
        >>> similarity_data = calc.calculate(chunk_data)
        >>> print(similarity_data.scores)  # Array of similarity values
    """

    def __init__(self, smoothing_passes: int) -> None:
        self.smoothing_passes = smoothing_passes

    def calculate(self, chunk_data: ChunkData) -> SimilarityData:
        """Compute similarity scores between adjacent chunks.

        Args:
            chunk_data: ChunkData from the ChunkEncoder.

        Returns:
            SimilarityData with scores and comparison positions.

        Raises:
            ValueError: If chunk_data has fewer than 2 chunks.
        """
        n_chunks = len(chunk_data.chunk_positions)

        if n_chunks < 2:
            raise ValueError("Need at least 2 chunks to compute similarity")

        embeddings = chunk_data.embeddings
        positions = chunk_data.chunk_positions

        # Compute cosine similarity between adjacent pairs
        scores = self._compute_cosine_similarities(embeddings)

        # Apply smoothing if requested
        if self.smoothing_passes > 0:
            scores = self._smooth(scores, self.smoothing_passes)

        # Comparison position is the midpoint between chunk starts
        # For chunks at positions [0, 25, 50, ...] with window_size=50,
        # comparison i is between chunk i and chunk i+1
        # Position is end of first chunk (or start of second chunk)
        # This represents the "gap" between the two blocks
        comparison_positions = [positions[i + 1] for i in range(len(scores))]

        return SimilarityData(scores=scores, comparison_positions=comparison_positions)

    def _compute_cosine_similarities(self, embeddings: NDArray[np.float32]) -> NDArray[np.float64]:
        """Compute cosine similarity between each adjacent pair of embeddings.

        Args:
            embeddings: Array of shape [n_chunks, embedding_dim]

        Returns:
            Array of shape [n_chunks - 1] with similarity scores.
        """
        # Normalize embeddings to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)  # Avoid division by zero

        # Cosine similarity between adjacent pairs
        # dot product of normalized vectors = cosine similarity
        similarities: NDArray[np.float64] = np.sum(normalized[:-1] * normalized[1:], axis=1)

        return similarities

    def _smooth(self, scores: NDArray[np.float64], passes: int) -> NDArray[np.float64]:
        """Apply moving average smoothing to the scores.

        Uses a 3-point moving average, preserving array length
        by keeping edge values.

        Args:
            scores: Array of similarity scores.
            passes: Number of smoothing iterations.

        Returns:
            Smoothed scores array.
        """
        result = scores.copy()

        for _ in range(passes):
            if len(result) < 3:
                break

            smoothed = np.zeros_like(result)
            smoothed[0] = (result[0] + result[1]) / 2
            smoothed[-1] = (result[-2] + result[-1]) / 2

            for i in range(1, len(result) - 1):
                smoothed[i] = (result[i - 1] + result[i] + result[i + 1]) / 3

            result = smoothed

        return result
