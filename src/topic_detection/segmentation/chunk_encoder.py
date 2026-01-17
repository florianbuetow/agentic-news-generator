"""Chunk encoder for the topic segmentation pipeline.

Generates embeddings for overlapping windows of tokens using an
injected EmbeddingGenerator.
"""

from src.topic_detection.embedding.lmstudio import LMStudioEmbeddingGenerator
from src.topic_detection.segmentation.data_types import ChunkData


class ChunkEncoder:
    """Encodes overlapping token windows into embedding vectors.

    Uses an injected EmbeddingGenerator to create embeddings for each
    text chunk.

    Args:
        embedding_generator: Generator instance for creating embeddings.
        window_size: Number of tokens per chunk.
        stride: Number of tokens to advance between chunks.

    Example:
        >>> from src.topic_detection.embedding.factory import EmbeddingGeneratorFactory
        >>> generator = EmbeddingGeneratorFactory.create(config.embedding)
        >>> encoder = ChunkEncoder(
        ...     embedding_generator=generator,
        ...     window_size=50,
        ...     stride=25,
        ... )
        >>> tokens = ["word"] * 100
        >>> chunk_data = encoder.encode(tokens)
        >>> print(chunk_data.embeddings.shape)  # (3, embedding_dim)
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

    def encode(self, tokens: list[str]) -> ChunkData:
        """Generate embeddings for overlapping token windows.

        Args:
            tokens: List of tokens from the Tokenizer.

        Returns:
            ChunkData containing embeddings and chunk positions.

        Raises:
            ValueError: If tokens list is shorter than window_size.
        """
        if len(tokens) < self.window_size:
            raise ValueError(f"Token list length ({len(tokens)}) must be >= window_size ({self.window_size})")

        chunks: list[str] = []
        positions: list[int] = []

        # Create overlapping chunks
        for i in range(0, len(tokens) - self.window_size + 1, self.stride):
            chunk_text = " ".join(tokens[i : i + self.window_size])
            chunks.append(chunk_text)
            positions.append(i)

        # Get embeddings for all chunks in one batch
        embeddings = self.embedding_generator.generate(chunks)

        return ChunkData(embeddings=embeddings, chunk_positions=positions)
