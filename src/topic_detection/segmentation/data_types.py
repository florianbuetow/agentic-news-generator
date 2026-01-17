"""Data types for the topic segmentation pipeline.

All dataclasses used to pass data between pipeline stages are defined here.
This keeps type definitions in one place and avoids circular imports.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ChunkData:
    """Output of the ChunkEncoder stage.

    Attributes:
        embeddings: numpy array of shape [n_chunks, embedding_dim]
                    Each row is the embedding vector for one text chunk.
        chunk_positions: List of word indices where each chunk starts.
                        Used to map chunks back to positions in the original text.
    """

    embeddings: NDArray[np.float32]
    chunk_positions: list[int]


@dataclass
class SimilarityData:
    """Output of the SimilarityCalculator stage.

    Attributes:
        scores: numpy array of shape [n_chunks - 1]
                Cosine similarity between each adjacent pair of chunks.
                scores[i] = similarity(chunk[i], chunk[i+1])
        comparison_positions: Word index for each comparison point.
                             This is the position between the two compared chunks.
    """

    scores: NDArray[np.float64]
    comparison_positions: list[int]


@dataclass
class BoundaryData:
    """Output of the BoundaryDetector stage.

    Attributes:
        boundary_indices: List of word positions where topic boundaries occur.
        depths: Depth score at each detected boundary (for debugging/analysis).
                Higher depth = more confident boundary.
    """

    boundary_indices: list[int]
    depths: list[float]


@dataclass
class Segment:
    """A single topic segment extracted from the text.

    Attributes:
        text: The reconstructed text content of this segment.
        start_word: Index of first word in this segment (inclusive).
        end_word: Index of last word in this segment (exclusive).
        word_count: Number of words in this segment.
    """

    text: str
    start_word: int
    end_word: int
    word_count: int


@dataclass
class SegmentationResult:
    """Final output of the segmentation pipeline.

    Attributes:
        segments: Ordered list of Segment objects.
        boundary_indices: Word indices where boundaries were placed.
        similarity_scores: The similarity curve (useful for visualization).
        num_words: Total number of words in the input text.
        chunk_data: Optional ChunkData containing embeddings and positions.
    """

    segments: list[Segment]
    boundary_indices: list[int]
    similarity_scores: NDArray[np.float64]
    num_words: int
    chunk_data: ChunkData | None = None
