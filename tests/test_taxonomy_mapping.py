"""Tests for deterministic taxonomy matching."""

import numpy as np

from src.config import TopicDetectionTaxonomyConfig
from src.topic_detection.hierarchy.labeling import TaxonomyMatcher
from src.topic_detection.taxonomy.data_types import TaxonomyConcept


def test_taxonomy_matcher_orders_ties_deterministically() -> None:
    """Ensure ties are resolved deterministically by stable sorting over sorted concept ids."""
    cfg = TopicDetectionTaxonomyConfig(
        enabled=True,
        taxonomy_name="acm_ccs_2012",
        acm_ccs_2012_xml_path="input/taxonomies/acm_ccs2012.xml",
        cache_dir="input/taxonomies/cache",
        top_k_per_node=3,
        min_similarity=0.0,
    )

    concepts = {
        "a": TaxonomyConcept(
            concept_id="a",
            pref_label="A",
            parents=["root"],
            level=2,
            path_labels=["Root", "A"],
        ),
        "b": TaxonomyConcept(
            concept_id="b",
            pref_label="B",
            parents=["root"],
            level=2,
            path_labels=["Root", "B"],
        ),
        "c": TaxonomyConcept(
            concept_id="c",
            pref_label="C",
            parents=["root"],
            level=2,
            path_labels=["Root", "C"],
        ),
    }

    concept_embeddings = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([1.0, 0.0], dtype=np.float32),
        "c": np.array([0.0, 1.0], dtype=np.float32),
    }

    matcher = TaxonomyMatcher(
        taxonomy_config=cfg,
        concepts=concepts,
        concept_embeddings=concept_embeddings,
        max_level=4,
    )

    node_embedding = np.array([1.0, 0.0], dtype=np.float32)

    matches = matcher.match(
        node_embedding=node_embedding,
        target_level=2,
        top_k=3,
        min_similarity=0.0,
    )

    assert [m.concept_id for m in matches] == ["a", "b", "c"]
    assert [m.label for m in matches] == ["A", "B", "C"]
    assert [m.path for m in matches] == [["Root", "A"], ["Root", "B"], ["Root", "C"]]
    assert [m.level for m in matches] == [2, 2, 2]
    assert [m.taxonomy for m in matches] == ["acm_ccs_2012", "acm_ccs_2012", "acm_ccs_2012"]

    filtered = matcher.match(
        node_embedding=node_embedding,
        target_level=2,
        top_k=3,
        min_similarity=0.5,
    )
    assert [m.concept_id for m in filtered] == ["a", "b"]
