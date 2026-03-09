"""Tests for YAKE and KeyBERT keyphrase extractors."""

import numpy as np
from numpy.typing import NDArray

from src.config import (
    TopicDetectionKeyBERTKeyphrasesConfig,
    TopicDetectionYAKEKeyphrasesConfig,
)
from src.topic_detection.hierarchy.labeling import (
    KeyBERTKeyphraseExtractor,
    YAKEKeyphraseExtractor,
)

SAMPLE_TEXT = (
    "Machine learning is a subset of artificial intelligence that focuses on building "
    "systems that learn from data. Deep learning is a further subset of machine learning "
    "that uses neural networks with many layers. Natural language processing enables "
    "computers to understand and generate human language."
)


class _MockEmbeddingGenerator:
    """Mock embedding generator for testing KeyBERT backend."""

    def generate(self, texts: list[str]) -> NDArray[np.float32]:
        rng = np.random.default_rng(42)
        return rng.random((len(texts), 64), dtype=np.float32)


class TestYAKEKeyphraseExtractor:
    """Tests for YAKEKeyphraseExtractor."""

    def _make_config(self, *, enabled: bool = True) -> TopicDetectionYAKEKeyphrasesConfig:
        return TopicDetectionYAKEKeyphrasesConfig(
            enabled=enabled,
            top_k_per_node=5,
            max_ngram_size=3,
            deduplication_threshold=0.9,
            deduplication_algo="seqm",
            window_size=1,
        )

    def test_extract_returns_keyphrases(self):
        extractor = YAKEKeyphraseExtractor(config=self._make_config())
        result = extractor.extract(text=SAMPLE_TEXT)
        assert len(result) > 0
        assert len(result) <= 5
        for kp in result:
            assert kp.phrase.strip() != ""
            assert kp.score > 0

    def test_extract_disabled_returns_empty(self):
        extractor = YAKEKeyphraseExtractor(config=self._make_config(enabled=False))
        result = extractor.extract(text=SAMPLE_TEXT)
        assert result == []

    def test_extract_empty_text_returns_empty(self):
        extractor = YAKEKeyphraseExtractor(config=self._make_config())
        result = extractor.extract(text="")
        assert result == []

    def test_extract_whitespace_only_returns_empty(self):
        extractor = YAKEKeyphraseExtractor(config=self._make_config())
        result = extractor.extract(text="   ")
        assert result == []

    def test_scores_are_inverted(self):
        """YAKE scores are inverted: lower YAKE = higher relevance = higher our score."""
        extractor = YAKEKeyphraseExtractor(config=self._make_config())
        result = extractor.extract(text=SAMPLE_TEXT)
        for kp in result:
            assert kp.score <= 1.0
            assert kp.score > 0


class TestKeyBERTKeyphraseExtractor:
    """Tests for KeyBERTKeyphraseExtractor."""

    def _make_config(self, *, enabled: bool = True) -> TopicDetectionKeyBERTKeyphrasesConfig:
        return TopicDetectionKeyBERTKeyphrasesConfig(
            enabled=enabled,
            top_k_per_node=5,
            keyphrase_ngram_range_min=1,
            keyphrase_ngram_range_max=2,
            use_mmr=True,
            mmr_diversity=0.5,
            stop_words="english",
        )

    def test_extract_returns_keyphrases(self):
        generator = _MockEmbeddingGenerator()
        extractor = KeyBERTKeyphraseExtractor(config=self._make_config(), embedding_generator=generator)
        result = extractor.extract(text=SAMPLE_TEXT)
        assert len(result) > 0
        assert len(result) <= 5
        for kp in result:
            assert kp.phrase.strip() != ""

    def test_extract_disabled_returns_empty(self):
        generator = _MockEmbeddingGenerator()
        extractor = KeyBERTKeyphraseExtractor(config=self._make_config(enabled=False), embedding_generator=generator)
        result = extractor.extract(text=SAMPLE_TEXT)
        assert result == []

    def test_extract_empty_text_returns_empty(self):
        generator = _MockEmbeddingGenerator()
        extractor = KeyBERTKeyphraseExtractor(config=self._make_config(), embedding_generator=generator)
        result = extractor.extract(text="")
        assert result == []

    def test_extract_whitespace_only_returns_empty(self):
        generator = _MockEmbeddingGenerator()
        extractor = KeyBERTKeyphraseExtractor(config=self._make_config(), embedding_generator=generator)
        result = extractor.extract(text="   ")
        assert result == []
