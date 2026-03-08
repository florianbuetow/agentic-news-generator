"""Tests for ACM CCS 2012 SKOS XML loader."""

from pathlib import Path

from src.topic_detection.taxonomy.acm_ccs2012 import ACMCCS2012Loader


def test_load_minimal_fixture_assigns_levels_and_paths_deterministically() -> None:
    """Load a tiny SKOS fixture and assert deterministic levels + path selection."""
    xml_path = Path(__file__).parent / "fixtures" / "acm_ccs_minimal.xml"
    concepts = ACMCCS2012Loader().load(xml_path=xml_path)

    assert len(concepts) == 5

    # Multiple English labels: lexicographically smallest is chosen.
    assert concepts["urn:root_c"].pref_label == "C Root"
    assert concepts["urn:root_c"].level == 1
    assert concepts["urn:root_c"].path_labels == ["C Root"]

    child = concepts["urn:child_x"]
    assert child.parents == ["urn:root_a", "urn:root_b"]
    assert child.level == 2
    # Path selection: choose lexicographically smallest label path among shortest paths.
    assert child.path_labels == ["A Root", "Child"]

    grand = concepts["urn:grand_y"]
    assert grand.parents == ["urn:child_x"]
    assert grand.level == 3
    assert grand.path_labels == ["A Root", "Child", "Grand"]
