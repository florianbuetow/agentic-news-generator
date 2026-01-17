"""LM Studio embedding generator for topic detection."""

import litellm
import numpy as np
from numpy.typing import NDArray

from src.config import TopicDetectionEmbeddingConfig


class LMStudioEmbeddingGenerator:
    """Embedding generator using LM Studio API via litellm."""

    def __init__(self, config: TopicDetectionEmbeddingConfig) -> None:
        """Initialize the embedding generator.

        Args:
            config: Embedding configuration with model name and API settings.
        """
        self._config = config
        self._embedding_dim: int | None = None

    def generate(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            NumPy array of shape (len(texts), embedding_dim) with float32 embeddings.

        Raises:
            litellm.exceptions.APIError: If the API call fails.
        """
        # LM Studio uses OpenAI-compatible API, so prefix with openai/
        model = f"openai/{self._config.model_name}"

        response = litellm.embedding(
            model=model,
            input=texts,
            api_base=self._config.api_base,
            api_key=self._config.api_key,
        )
        embeddings = [item["embedding"] for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension by generating a test embedding.

        Returns:
            The embedding dimension.
        """
        if self._embedding_dim is None:
            test = self.generate(["test"])
            self._embedding_dim = test.shape[1]
        return self._embedding_dim
