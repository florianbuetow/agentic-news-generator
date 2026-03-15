"""ACM CCS 2012 taxonomy loader (SKOS XML).

The ACM CCS 2012 taxonomy is a poly-hierarchical ontology. This loader derives:
- `level`: minimum depth from any root (roots have no broader/parents)
- `path_labels`: deterministic root-to-node label path chosen by lexicographic ordering
  among shortest paths (by `level`).
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

from defusedxml import ElementTree

from src.topic_detection.taxonomy.data_types import TaxonomyConcept


class ACMCCS2012Loader:
    """Load and normalize the ACM CCS 2012 taxonomy."""

    _NS = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "xml": "http://www.w3.org/XML/1998/namespace",
    }

    def load(self, *, xml_path: Path) -> dict[str, TaxonomyConcept]:
        """Load ACM CCS 2012 SKOS XML.

        Args:
            xml_path: Path to the SKOS XML file.

        Returns:
            Mapping concept_id -> TaxonomyConcept with derived fields.

        Raises:
            FileNotFoundError: If xml_path does not exist.
            ValueError: If the file cannot be parsed or contains unsupported structures.
        """
        if not xml_path.exists():
            raise FileNotFoundError(f"ACM CCS 2012 taxonomy XML not found: {xml_path}")

        root = self._parse_xml(xml_path)
        raw_concepts = self._extract_raw_concepts(root)
        self._validate_parent_refs(raw_concepts)

        children = self._build_children(raw_concepts)
        levels, roots = self._compute_levels(raw_concepts, children)
        path_labels_by_id = self._compute_path_labels(raw_concepts, levels, roots)
        return self._materialize_concepts(raw_concepts, levels, path_labels_by_id)

    def _parse_xml(self, xml_path: Path) -> ElementTree.Element:
        """Parse XML and return the root element."""
        try:
            tree = ElementTree.parse(xml_path)
        except ElementTree.ParseError as e:
            raise ValueError(f"Failed to parse taxonomy XML: {e}") from e
        return tree.getroot()

    def _extract_raw_concepts(self, root: ElementTree.Element) -> dict[str, tuple[str, list[str]]]:
        """Extract raw concepts (label + parents) from XML."""
        raw_concepts: dict[str, tuple[str, list[str]]] = {}
        for concept_el in root.findall(".//skos:Concept", namespaces=self._NS):
            concept_id = concept_el.get(f"{{{self._NS['rdf']}}}about")
            if concept_id is None or concept_id.strip() == "":
                raise ValueError("Found skos:Concept without rdf:about")

            label = self._extract_english_pref_label(concept_el)
            parents = self._extract_parents(concept_el)
            raw_concepts[concept_id] = (label, parents)

        if not raw_concepts:
            raise ValueError("No skos:Concept elements found in taxonomy XML")
        return raw_concepts

    def _validate_parent_refs(self, raw_concepts: dict[str, tuple[str, list[str]]]) -> None:
        """Ensure all parent references exist."""
        for concept_id, (_, parents) in raw_concepts.items():
            for parent_id in parents:
                if parent_id not in raw_concepts:
                    raise ValueError(f"Concept '{concept_id}' references missing parent '{parent_id}'")

    def _build_children(self, raw_concepts: dict[str, tuple[str, list[str]]]) -> dict[str, list[str]]:
        """Build children adjacency list for BFS."""
        children: dict[str, list[str]] = defaultdict(list)
        for child_id, (_, parents) in raw_concepts.items():
            for parent_id in parents:
                children[parent_id].append(child_id)
        return children

    def _compute_levels(
        self,
        raw_concepts: dict[str, tuple[str, list[str]]],
        children: dict[str, list[str]],
    ) -> tuple[dict[str, int], list[str]]:
        """Compute minimum depth (level) from roots."""
        roots = sorted([cid for cid, (_, parents) in raw_concepts.items() if not parents])
        if not roots:
            raise ValueError("No root concepts found (expected some concepts with no skos:broader parents)")

        levels: dict[str, int] = {}
        q: deque[str] = deque()
        for cid in roots:
            levels[cid] = 1
            q.append(cid)

        while q:
            cid = q.popleft()
            parent_level = levels[cid]
            if cid not in children:
                continue
            for child_id in children[cid]:
                proposed = parent_level + 1
                existing = levels.get(child_id)
                if existing is None or proposed < existing:
                    levels[child_id] = proposed
                    q.append(child_id)

        missing_levels = sorted([cid for cid in raw_concepts if cid not in levels])
        if missing_levels:
            raise ValueError(f"Some concepts are unreachable from roots (cannot assign level): {missing_levels[:10]}")

        return levels, roots

    def _compute_path_labels(
        self,
        raw_concepts: dict[str, tuple[str, list[str]]],
        levels: dict[str, int],
        roots: list[str],
    ) -> dict[str, list[str]]:
        """Compute deterministic label paths among shortest paths (by level)."""
        concepts_by_level: dict[int, list[str]] = defaultdict(list)
        for cid, level in levels.items():
            concepts_by_level[level].append(cid)

        path_labels_by_id: dict[str, list[str]] = {}
        for cid in roots:
            label, _ = raw_concepts[cid]
            path_labels_by_id[cid] = [label]

        max_level = max(levels.values())
        for level in range(2, max_level + 1):
            if level not in concepts_by_level:
                continue
            for cid in sorted(concepts_by_level[level]):
                label, parents = raw_concepts[cid]
                eligible_parents = [p for p in parents if levels[p] == level - 1]
                if not eligible_parents:
                    raise ValueError(f"Concept '{cid}' has no eligible parent at level {level - 1} for shortest-path labeling")

                candidate_paths = [path_labels_by_id[p] + [label] for p in eligible_parents]
                path_labels_by_id[cid] = min(candidate_paths)

        return path_labels_by_id

    def _materialize_concepts(
        self,
        raw_concepts: dict[str, tuple[str, list[str]]],
        levels: dict[str, int],
        path_labels_by_id: dict[str, list[str]],
    ) -> dict[str, TaxonomyConcept]:
        """Materialize normalized TaxonomyConcept objects."""
        concepts: dict[str, TaxonomyConcept] = {}
        for cid, (label, parents) in raw_concepts.items():
            concepts[cid] = TaxonomyConcept(
                concept_id=cid,
                pref_label=label,
                parents=sorted(parents),
                level=levels[cid],
                path_labels=path_labels_by_id[cid],
            )
        return concepts

    def _extract_english_pref_label(self, concept_el: ElementTree.Element) -> str:
        """Extract the English skos:prefLabel from a concept element."""
        labels: list[str] = []
        for label_el in concept_el.findall("skos:prefLabel", namespaces=self._NS):
            xml_lang = label_el.get(f"{{{self._NS['xml']}}}lang")
            bare_lang = label_el.get("lang")
            if xml_lang is not None:
                lang = xml_lang
            elif bare_lang is not None:
                lang = bare_lang
            else:
                continue
            if lang == "en" and label_el.text is not None:
                labels.append(label_el.text.strip())

        labels = [label_item for label_item in labels if label_item != ""]
        if not labels:
            concept_id = concept_el.get(f"{{{self._NS['rdf']}}}about")
            if concept_id is None:
                raise ValueError("Concept without rdf:about is missing an English skos:prefLabel")
            raise ValueError(f"Concept '{concept_id}' is missing an English skos:prefLabel")

        if len(labels) > 1:
            # Deterministic selection when multiple English labels exist.
            return sorted(labels)[0]

        return labels[0]

    def _extract_parents(self, concept_el: ElementTree.Element) -> list[str]:
        """Extract skos:broader parent concept ids."""
        parents: list[str] = []
        for broader_el in concept_el.findall("skos:broader", namespaces=self._NS):
            parent_id = broader_el.get(f"{{{self._NS['rdf']}}}resource")
            if parent_id is None or parent_id.strip() == "":
                raise ValueError("Found skos:broader without rdf:resource")
            parents.append(parent_id.strip())
        return parents
