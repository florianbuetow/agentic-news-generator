"""Chunk encoder for the topic segmentation pipeline.

Generates embeddings for overlapping windows of words using an
injected EmbeddingGenerator.
"""

from src.topic_detection.embedding.lmstudio import LMStudioEmbeddingGenerator
from src.topic_detection.segmentation.data_types import ChunkData


class ChunkEncoder:
    """Encodes overlapping word windows into embedding vectors.

    Uses an injected EmbeddingGenerator to create embeddings for each
    text chunk. Note: "words" here refers to word-level tokens from the
    Tokenizer (not BPE/subword tokens).

    Args:
        embedding_generator: Generator instance for creating embeddings.
        window_size: Number of words per chunk.
        stride: Number of words to advance between chunks.
    """

    def __init__(
        self,
        embedding_generator: LMStudioEmbeddingGenerator,
        window_size: int,
        stride: int,
    ) -> None:
        self.embedding_generator = embedding_generator
        self.window_size = window_size
        self.stride = stride

    def encode(self, words: list[str]) -> ChunkData:
        """Generate embeddings for overlapping word windows.

        Args:
            words: List of words from the Tokenizer.

        Returns:
            ChunkData containing embeddings and chunk positions.

        Raises:
            ValueError: If words list is shorter than window_size.
        """
        if len(words) < self.window_size:
            raise ValueError(f"Word list length ({len(words)}) must be >= window_size ({self.window_size})")

        chunks: list[str] = []
        positions: list[int] = []

        # Create overlapping chunks
        for i in range(0, len(words) - self.window_size + 1, self.stride):
            chunk_text = " ".join(words[i : i + self.window_size])
            chunks.append(chunk_text)
            positions.append(i)

        # Get embeddings for all chunks in one batch
        embeddings = self.embedding_generator.generate(chunks)

        return ChunkData(embeddings=embeddings, chunk_positions=positions)
