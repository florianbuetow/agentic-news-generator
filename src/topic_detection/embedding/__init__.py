"""Embedding generators for topic detection."""

from src.topic_detection.embedding.factory import EmbeddingGeneratorFactory
from src.topic_detection.embedding.lmstudio import LMStudioEmbeddingGenerator

__all__ = ["EmbeddingGeneratorFactory", "LMStudioEmbeddingGenerator"]
