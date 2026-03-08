"""Taxonomy data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaxonomyConcept:
    """A taxonomy concept with derived level and deterministic path."""

    concept_id: str
    pref_label: str
    parents: list[str]
    level: int
    path_labels: list[str]
