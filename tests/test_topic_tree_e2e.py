"""End-to-end test for build_tree_output with all agents mocked."""

# Import build_tree_output from the script
# scripts/ is not a package, so we import via importlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import (
    TopicDetectionHierarchicalSegmentationConfig,
    TopicDetectionKeyBERTKeyphrasesConfig,
    TopicDetectionKeyphrasesConfig,
    TopicDetectionTaxonomyConfig,
    TopicDetectionTFIDFKeyphrasesConfig,
    TopicDetectionYAKEKeyphrasesConfig,
)
from src.topic_detection.hierarchy.labeling import (
    KeyBERTKeyphraseExtractor,
    Keyphrase,
    LLMTopicLabeler,
    TaxonomyMatch,
    TaxonomyMatcher,
    TFIDFKeyphraseExtractor,
    YAKEKeyphraseExtractor,
)
from src.topic_detection.hierarchy.schemas import LLMTopicLabelData

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "topic-tree.py"
_spec = importlib.util.spec_from_file_location("topic_tree_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)  # type: ignore[union-attr]
build_tree_output = _module.build_tree_output  # type: ignore[attr-defined]

SRT_FIXTURE = Path(__file__).parent / "fixtures" / "sample_transcript.srt"
EMBEDDING_DIM = 32


@pytest.fixture()
def seg_cfg() -> TopicDetectionHierarchicalSegmentationConfig:
    return TopicDetectionHierarchicalSegmentationConfig(
        enabled=True,
        method="treeseg_divisive_sse",
        context_window_entries=3,
        max_depth=2,
        min_leaf_entries=3,
        min_leaf_seconds=10.0,
        min_gain=0.001,
    )


@pytest.fixture()
def taxonomy_cfg() -> TopicDetectionTaxonomyConfig:
    return TopicDetectionTaxonomyConfig(
        enabled=True,
        taxonomy_name="acm_ccs_2012",
        acm_ccs_2012_xml_file="acm_ccs2012.xml",
        top_k_per_node=2,
        min_similarity=0.1,
    )


@pytest.fixture()
def keyphrases_cfg() -> TopicDetectionKeyphrasesConfig:
    return TopicDetectionKeyphrasesConfig(
        tfidf=TopicDetectionTFIDFKeyphrasesConfig(
            enabled=True,
            top_k_per_node=5,
            ngram_range_min=1,
            ngram_range_max=2,
            min_df=1,
            max_df=1.0,
            stop_words="english",
            max_features=1000,
            lowercase=True,
        ),
        yake=TopicDetectionYAKEKeyphrasesConfig(
            enabled=True,
            top_k_per_node=5,
            max_ngram_size=2,
            deduplication_threshold=0.9,
            deduplication_algo="seqm",
            window_size=1,
        ),
        keybert=TopicDetectionKeyBERTKeyphrasesConfig(
            enabled=True,
            top_k_per_node=5,
            keyphrase_ngram_range_min=1,
            keyphrase_ngram_range_max=2,
            use_mmr=False,
            mmr_diversity=0.5,
            min_score=0.0,
            stop_words="english",
        ),
    )


@pytest.fixture()
def mock_embedding_generator() -> MagicMock:
    """Mock embedding generator that returns deterministic random embeddings."""
    rng = np.random.default_rng(seed=42)
    generator = MagicMock()

    def fake_generate(texts: list[str]) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        embeddings = rng.standard_normal((len(texts), EMBEDDING_DIM)).astype(np.float32)
        # Normalize so cosine similarity is meaningful
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)

    generator.generate.side_effect = fake_generate
    return generator


@pytest.fixture()
def mock_taxonomy_matcher() -> MagicMock:
    """Mock taxonomy matcher returning fixed matches."""
    matcher = MagicMock(spec=TaxonomyMatcher)
    matcher.match.return_value = [
        TaxonomyMatch(
            taxonomy="acm_ccs_2012",
            concept_id="10011007",
            label="Artificial intelligence",
            path=["Computing methodologies", "Artificial intelligence"],
            level=2,
            score=0.85,
        ),
    ]
    return matcher


@pytest.fixture()
def mock_tfidf_extractor() -> MagicMock:
    """Mock TF-IDF extractor."""
    extractor = MagicMock(spec=TFIDFKeyphraseExtractor)
    extractor.fit_vectorizer.return_value = (MagicMock(), None)
    extractor.extract.return_value = [
        Keyphrase(phrase="artificial intelligence", score=0.9),
        Keyphrase(phrase="language models", score=0.7),
    ]
    return extractor


@pytest.fixture()
def mock_yake_extractor() -> MagicMock:
    """Mock YAKE extractor."""
    extractor = MagicMock(spec=YAKEKeyphraseExtractor)
    extractor.extract.return_value = [
        Keyphrase(phrase="ai models", score=0.8),
        Keyphrase(phrase="regulation", score=0.6),
    ]
    return extractor


@pytest.fixture()
def mock_keybert_extractor() -> MagicMock:
    """Mock KeyBERT extractor."""
    extractor = MagicMock(spec=KeyBERTKeyphraseExtractor)
    extractor.extract.return_value = [
        Keyphrase(phrase="large language models", score=0.85),
        Keyphrase(phrase="ai safety", score=0.72),
    ]
    return extractor


@pytest.fixture()
def mock_llm_labeler() -> MagicMock:
    """Mock LLM topic labeler."""
    labeler = MagicMock(spec=LLMTopicLabeler)
    labeler.label.return_value = LLMTopicLabelData(
        summary="Discusses AI developments including language models and regulation.",
        about="Recent advances in AI technology and policy.",
        topic_labels=["ai", "regulation", "models"],
    )
    return labeler


class TestBuildTreeOutputE2E:
    """End-to-end tests for build_tree_output with all agents mocked."""

    def test_produces_valid_output(
        self,
        tmp_path: Path,
        seg_cfg: TopicDetectionHierarchicalSegmentationConfig,
        taxonomy_cfg: TopicDetectionTaxonomyConfig,
        keyphrases_cfg: TopicDetectionKeyphrasesConfig,
        mock_embedding_generator: MagicMock,
        mock_taxonomy_matcher: MagicMock,
        mock_tfidf_extractor: MagicMock,
        mock_yake_extractor: MagicMock,
        mock_keybert_extractor: MagicMock,
        mock_llm_labeler: MagicMock,
    ) -> None:
        output = build_tree_output(
            srt_path=SRT_FIXTURE,
            data_dir=SRT_FIXTURE.parent.parent,
            seg_cfg=seg_cfg,
            taxonomy_cfg=taxonomy_cfg,
            keyphrases_cfg=keyphrases_cfg,
            embedding_generator=mock_embedding_generator,
            embedding_model="test-embedding-model",
            matcher=mock_taxonomy_matcher,
            tfidf_extractor=mock_tfidf_extractor,
            yake_extractor=mock_yake_extractor,
            keybert_extractor=mock_keybert_extractor,
            llm_labeler=mock_llm_labeler,
        )

        # Basic structure
        assert output.total_nodes >= 1
        assert output.embedding_model == "test-embedding-model"
        assert output.root_node_id is not None
        assert len(output.nodes) == output.total_nodes

        # Root node
        root = output.nodes[0]
        assert root.node_id == output.root_node_id
        assert root.parent_id is None
        assert root.depth == 0
        assert root.duration_seconds > 0

        # All nodes have required fields
        for node in output.nodes:
            assert node.node_id is not None
            assert node.start_timestamp is not None
            assert node.end_timestamp is not None
            assert node.duration_seconds >= 0
            assert len(node.keyphrases) == 3  # tfidf, yake, keybert

        # All agents were called
        mock_embedding_generator.generate.assert_called_once()
        assert mock_taxonomy_matcher.match.call_count == output.total_nodes
        assert mock_tfidf_extractor.extract.call_count == output.total_nodes
        assert mock_yake_extractor.extract.call_count == output.total_nodes
        assert mock_keybert_extractor.extract.call_count == output.total_nodes
        assert mock_llm_labeler.label.call_count == output.total_nodes

    def test_taxonomy_labels_present(
        self,
        seg_cfg: TopicDetectionHierarchicalSegmentationConfig,
        taxonomy_cfg: TopicDetectionTaxonomyConfig,
        keyphrases_cfg: TopicDetectionKeyphrasesConfig,
        mock_embedding_generator: MagicMock,
        mock_taxonomy_matcher: MagicMock,
        mock_tfidf_extractor: MagicMock,
        mock_yake_extractor: MagicMock,
        mock_keybert_extractor: MagicMock,
        mock_llm_labeler: MagicMock,
    ) -> None:
        output = build_tree_output(
            srt_path=SRT_FIXTURE,
            data_dir=SRT_FIXTURE.parent.parent,
            seg_cfg=seg_cfg,
            taxonomy_cfg=taxonomy_cfg,
            keyphrases_cfg=keyphrases_cfg,
            embedding_generator=mock_embedding_generator,
            embedding_model="test-model",
            matcher=mock_taxonomy_matcher,
            tfidf_extractor=mock_tfidf_extractor,
            yake_extractor=mock_yake_extractor,
            keybert_extractor=mock_keybert_extractor,
            llm_labeler=mock_llm_labeler,
        )

        for node in output.nodes:
            assert len(node.taxonomy_labels) == 1
            assert node.taxonomy_labels[0].label == "Artificial intelligence"
            assert node.taxonomy_labels[0].score == 0.85

    def test_llm_labels_present(
        self,
        seg_cfg: TopicDetectionHierarchicalSegmentationConfig,
        taxonomy_cfg: TopicDetectionTaxonomyConfig,
        keyphrases_cfg: TopicDetectionKeyphrasesConfig,
        mock_embedding_generator: MagicMock,
        mock_taxonomy_matcher: MagicMock,
        mock_tfidf_extractor: MagicMock,
        mock_yake_extractor: MagicMock,
        mock_keybert_extractor: MagicMock,
        mock_llm_labeler: MagicMock,
    ) -> None:
        output = build_tree_output(
            srt_path=SRT_FIXTURE,
            data_dir=SRT_FIXTURE.parent.parent,
            seg_cfg=seg_cfg,
            taxonomy_cfg=taxonomy_cfg,
            keyphrases_cfg=keyphrases_cfg,
            embedding_generator=mock_embedding_generator,
            embedding_model="test-model",
            matcher=mock_taxonomy_matcher,
            tfidf_extractor=mock_tfidf_extractor,
            yake_extractor=mock_yake_extractor,
            keybert_extractor=mock_keybert_extractor,
            llm_labeler=mock_llm_labeler,
        )

        for node in output.nodes:
            assert node.llm_label is not None
            assert node.llm_label.summary == "Discusses AI developments including language models and regulation."
            assert node.llm_label.topic_labels == ["ai", "regulation", "models"]

    def test_without_llm_labeler(
        self,
        seg_cfg: TopicDetectionHierarchicalSegmentationConfig,
        taxonomy_cfg: TopicDetectionTaxonomyConfig,
        keyphrases_cfg: TopicDetectionKeyphrasesConfig,
        mock_embedding_generator: MagicMock,
        mock_taxonomy_matcher: MagicMock,
        mock_tfidf_extractor: MagicMock,
        mock_yake_extractor: MagicMock,
        mock_keybert_extractor: MagicMock,
    ) -> None:
        output = build_tree_output(
            srt_path=SRT_FIXTURE,
            data_dir=SRT_FIXTURE.parent.parent,
            seg_cfg=seg_cfg,
            taxonomy_cfg=taxonomy_cfg,
            keyphrases_cfg=keyphrases_cfg,
            embedding_generator=mock_embedding_generator,
            embedding_model="test-model",
            matcher=mock_taxonomy_matcher,
            tfidf_extractor=mock_tfidf_extractor,
            yake_extractor=mock_yake_extractor,
            keybert_extractor=mock_keybert_extractor,
            llm_labeler=None,
        )

        for node in output.nodes:
            assert node.llm_label is None

    def test_without_taxonomy_matcher(
        self,
        seg_cfg: TopicDetectionHierarchicalSegmentationConfig,
        taxonomy_cfg: TopicDetectionTaxonomyConfig,
        keyphrases_cfg: TopicDetectionKeyphrasesConfig,
        mock_embedding_generator: MagicMock,
        mock_tfidf_extractor: MagicMock,
        mock_yake_extractor: MagicMock,
        mock_keybert_extractor: MagicMock,
        mock_llm_labeler: MagicMock,
    ) -> None:
        output = build_tree_output(
            srt_path=SRT_FIXTURE,
            data_dir=SRT_FIXTURE.parent.parent,
            seg_cfg=seg_cfg,
            taxonomy_cfg=taxonomy_cfg,
            keyphrases_cfg=keyphrases_cfg,
            embedding_generator=mock_embedding_generator,
            embedding_model="test-model",
            matcher=None,
            tfidf_extractor=mock_tfidf_extractor,
            yake_extractor=mock_yake_extractor,
            keybert_extractor=mock_keybert_extractor,
            llm_labeler=mock_llm_labeler,
        )

        for node in output.nodes:
            assert node.taxonomy_labels == []

    def test_output_serializes_to_json(
        self,
        tmp_path: Path,
        seg_cfg: TopicDetectionHierarchicalSegmentationConfig,
        taxonomy_cfg: TopicDetectionTaxonomyConfig,
        keyphrases_cfg: TopicDetectionKeyphrasesConfig,
        mock_embedding_generator: MagicMock,
        mock_taxonomy_matcher: MagicMock,
        mock_tfidf_extractor: MagicMock,
        mock_yake_extractor: MagicMock,
        mock_keybert_extractor: MagicMock,
        mock_llm_labeler: MagicMock,
    ) -> None:
        output = build_tree_output(
            srt_path=SRT_FIXTURE,
            data_dir=SRT_FIXTURE.parent.parent,
            seg_cfg=seg_cfg,
            taxonomy_cfg=taxonomy_cfg,
            keyphrases_cfg=keyphrases_cfg,
            embedding_generator=mock_embedding_generator,
            embedding_model="test-model",
            matcher=mock_taxonomy_matcher,
            tfidf_extractor=mock_tfidf_extractor,
            yake_extractor=mock_yake_extractor,
            keybert_extractor=mock_keybert_extractor,
            llm_labeler=mock_llm_labeler,
        )

        # Serialize to JSON file (same as the real pipeline does)
        output_path = tmp_path / "test_topic_tree.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, indent=2, ensure_ascii=False)

        # Read back and verify structure
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_nodes"] == output.total_nodes
        assert data["root_node_id"] == output.root_node_id
        assert len(data["nodes"]) == output.total_nodes

        # Verify llm_label serialized correctly
        for node_data in data["nodes"]:
            assert "llm_label" in node_data
            assert node_data["llm_label"]["topic_labels"] == ["ai", "regulation", "models"]

    def test_tree_has_hierarchy(
        self,
        seg_cfg: TopicDetectionHierarchicalSegmentationConfig,
        taxonomy_cfg: TopicDetectionTaxonomyConfig,
        keyphrases_cfg: TopicDetectionKeyphrasesConfig,
        mock_embedding_generator: MagicMock,
        mock_taxonomy_matcher: MagicMock,
        mock_tfidf_extractor: MagicMock,
        mock_yake_extractor: MagicMock,
        mock_keybert_extractor: MagicMock,
        mock_llm_labeler: MagicMock,
    ) -> None:
        output = build_tree_output(
            srt_path=SRT_FIXTURE,
            data_dir=SRT_FIXTURE.parent.parent,
            seg_cfg=seg_cfg,
            taxonomy_cfg=taxonomy_cfg,
            keyphrases_cfg=keyphrases_cfg,
            embedding_generator=mock_embedding_generator,
            embedding_model="test-model",
            matcher=mock_taxonomy_matcher,
            tfidf_extractor=mock_tfidf_extractor,
            yake_extractor=mock_yake_extractor,
            keybert_extractor=mock_keybert_extractor,
            llm_labeler=mock_llm_labeler,
        )

        # With 17 entries and max_depth=2, we should get a tree with multiple nodes
        assert output.total_nodes > 1

        # Root should have children
        root = output.nodes[0]
        assert len(root.children_ids) > 0

        # Leaf nodes should have no children
        leaves = [n for n in output.nodes if len(n.children_ids) == 0]
        assert len(leaves) >= 2
