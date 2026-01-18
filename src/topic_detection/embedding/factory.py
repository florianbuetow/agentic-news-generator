"""Factory for creating embedding generators."""

from src.config import TopicDetectionEmbeddingConfig
from src.topic_detection.embedding.lmstudio import LMStudioEmbeddingGenerator


class EmbeddingGeneratorFactory:
    """Factory for creating EmbeddingGenerator instances based on provider type."""

    _PROVIDERS: dict[str, type[LMStudioEmbeddingGenerator]] = {
        "lmstudio": LMStudioEmbeddingGenerator,
    }

    @classmethod
    def create(cls, config: TopicDetectionEmbeddingConfig) -> LMStudioEmbeddingGenerator:
        """Create an embedding generator based on config provider type.

        Args:
            config: Embedding configuration with provider type.

        Returns:
            Configured embedding generator instance.

        Raises:
            ValueError: If provider is not supported.
        """
        provider = config.provider
        if provider not in cls._PROVIDERS:
            supported = ", ".join(cls._PROVIDERS.keys())
            raise ValueError(f"Unknown embedding provider: {provider}. Supported: {supported}")

        generator_class = cls._PROVIDERS[provider]
        return generator_class(config)
