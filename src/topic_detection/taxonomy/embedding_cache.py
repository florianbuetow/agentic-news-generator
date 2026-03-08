"""Taxonomy embedding cache utilities."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from src.topic_detection.taxonomy.data_types import TaxonomyConcept


class EmbeddingGenerator(Protocol):
    """Minimal interface for embedding generators used by caching."""

    def generate(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for texts."""
        ...


class TaxonomyEmbeddingCacheFile(BaseModel):
    """On-disk cache of taxonomy concept embeddings."""

    taxonomy_name: str = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)
    generated_at: str
    concept_embeddings: dict[str, list[float]]

    model_config = ConfigDict(frozen=True, extra="forbid")


def sanitize_filename_component(value: str) -> str:
    """Sanitize a value for use in a filename."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    sanitized = sanitized.strip("._-")
    return sanitized if sanitized else "unnamed"


def get_cache_path(*, data_dir: Path, taxonomy_name: str, embedding_model: str, cache_dir: str) -> Path:
    """Compute the cache file path for a taxonomy+embedding model."""
    model_part = sanitize_filename_component(embedding_model)
    taxonomy_part = sanitize_filename_component(taxonomy_name)
    return data_dir / cache_dir / f"{taxonomy_part}.{model_part}.json"


def load_cache(*, cache_path: Path) -> TaxonomyEmbeddingCacheFile:
    """Load a taxonomy embedding cache from disk."""
    with open(cache_path, encoding="utf-8") as f:
        return TaxonomyEmbeddingCacheFile.model_validate(json.load(f))


def write_cache(*, cache_path: Path, cache: TaxonomyEmbeddingCacheFile) -> None:
    """Write a taxonomy embedding cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache.model_dump(), f, indent=2, sort_keys=True, ensure_ascii=False)


def build_cache(
    *,
    taxonomy_name: str,
    embedding_model: str,
    concepts: dict[str, TaxonomyConcept],
    embedding_generator: EmbeddingGenerator,
    max_level: int,
    batch_size: int,
) -> TaxonomyEmbeddingCacheFile:
    """Build a cache by embedding concept labels up to max_level."""
    if max_level <= 0:
        raise ValueError("max_level must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    concept_ids = sorted([cid for cid, c in concepts.items() if c.level <= max_level])
    if not concept_ids:
        raise ValueError(f"No concepts found at levels <= {max_level}")

    labels = [concepts[cid].pref_label for cid in concept_ids]

    vectors: list[NDArray[np.float32]] = []
    for i in range(0, len(labels), batch_size):
        batch = labels[i : i + batch_size]
        vectors.append(embedding_generator.generate(batch))

    all_vectors = np.vstack(vectors).astype(np.float32, copy=False)
    if all_vectors.shape[0] != len(concept_ids):
        raise ValueError("Embedding generator returned unexpected number of vectors")

    concept_embeddings: dict[str, list[float]] = {}
    for cid, vec in zip(concept_ids, all_vectors, strict=True):
        concept_embeddings[cid] = vec.tolist()

    return TaxonomyEmbeddingCacheFile(
        taxonomy_name=taxonomy_name,
        embedding_model=embedding_model,
        generated_at=datetime.now().isoformat(),
        concept_embeddings=concept_embeddings,
    )
