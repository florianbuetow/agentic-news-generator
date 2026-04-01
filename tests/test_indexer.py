"""Tests for the knowledge-base index builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import src.agents.article_generation.knowledge_base.indexer as indexer_module
from src.agents.article_generation.knowledge_base.indexer import KnowledgeBaseIndexer


class _FakeEmbeddingClient:
    call_count = 0

    def __init__(self, **kwargs: object) -> None:
        self.calls: list[dict[str, object]] = []

    def embed_text(self, *, text: str, timeout_seconds: int) -> npt.NDArray[np.float32]:
        _FakeEmbeddingClient.call_count += 1
        self.calls.append({"text": text, "timeout_seconds": timeout_seconds})
        return np.array([len(text), timeout_seconds], dtype=np.float32)


def _make_indexer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KnowledgeBaseIndexer:
    monkeypatch.setattr(indexer_module, "KnowledgeBaseEmbeddingClient", _FakeEmbeddingClient)
    return KnowledgeBaseIndexer(
        data_dir=tmp_path / "kb",
        index_dir=tmp_path / "index",
        chunk_size_tokens=64,
        chunk_overlap_tokens=8,
        embedding_provider="lmstudio",
        embedding_model_name="embed-model",
        embedding_api_base="http://localhost:1234/v1",
        embedding_api_key="secret",
        embedding_timeout_seconds=11,
        encoding_name="o200k_base",
    )


def test_init_rejects_overlap_not_smaller_than_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be smaller"):
        KnowledgeBaseIndexer(
            data_dir=tmp_path / "kb",
            index_dir=tmp_path / "index",
            chunk_size_tokens=32,
            chunk_overlap_tokens=32,
            embedding_provider="lmstudio",
            embedding_model_name="embed-model",
            embedding_api_base="http://localhost:1234/v1",
            embedding_api_key="secret",
            embedding_timeout_seconds=11,
            encoding_name="o200k_base",
        )


def test_ensure_index_builds_and_persists_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "alpha.txt").write_text("Alpha document.", encoding="utf-8")
    (kb_dir / "beta.md").write_text("Beta document.", encoding="utf-8")

    indexer = _make_indexer(tmp_path, monkeypatch)

    index_version = indexer.ensure_index()

    assert len(index_version) == 16
    manifest_path = tmp_path / "index" / "manifest.json"
    chunks_path = tmp_path / "index" / "chunks.json"
    embeddings_path = tmp_path / "index" / "embeddings.npy"
    assert manifest_path.exists()
    assert chunks_path.exists()
    assert embeddings_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path)

    assert manifest["index_version"] == index_version
    assert manifest["chunk_count"] == 2
    assert manifest["embedding_provider"] == "lmstudio"
    assert manifest["embedding_model_name"] == "embed-model"
    assert len(chunks) == 2
    assert embeddings.shape == (2, 2)


def test_ensure_index_reuses_current_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "alpha.txt").write_text("Alpha document.", encoding="utf-8")

    _FakeEmbeddingClient.call_count = 0
    indexer = _make_indexer(tmp_path, monkeypatch)
    first_version = indexer.ensure_index()
    first_call_count = _FakeEmbeddingClient.call_count

    second_indexer = _make_indexer(tmp_path, monkeypatch)
    second_version = second_indexer.ensure_index()

    assert second_version == first_version
    assert _FakeEmbeddingClient.call_count == first_call_count


def test_ensure_index_rebuilds_when_manifest_is_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "alpha.txt").write_text("Alpha document.", encoding="utf-8")

    indexer = _make_indexer(tmp_path, monkeypatch)
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "manifest.json").write_text("{bad-json", encoding="utf-8")

    index_version = indexer.ensure_index()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["index_version"] == index_version


def test_ensure_index_requires_existing_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    indexer = _make_indexer(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="data_dir does not exist"):
        indexer.ensure_index()


def test_build_embeddings_rejects_dimension_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _ChangingEmbeddingClient:
        def __init__(self, **kwargs: object) -> None:
            self.calls = 0

        def embed_text(self, *, text: str, timeout_seconds: int) -> npt.NDArray[np.float32]:
            self.calls += 1
            if self.calls == 1:
                return np.array([1.0, 2.0], dtype=np.float32)
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(indexer_module, "KnowledgeBaseEmbeddingClient", _ChangingEmbeddingClient)
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "alpha.txt").write_text("Alpha document.", encoding="utf-8")
    (kb_dir / "beta.txt").write_text("Beta document.", encoding="utf-8")
    indexer = KnowledgeBaseIndexer(
        data_dir=kb_dir,
        index_dir=tmp_path / "index",
        chunk_size_tokens=64,
        chunk_overlap_tokens=8,
        embedding_provider="lmstudio",
        embedding_model_name="embed-model",
        embedding_api_base="http://localhost:1234/v1",
        embedding_api_key="secret",
        embedding_timeout_seconds=11,
        encoding_name="o200k_base",
    )

    with pytest.raises(RuntimeError, match="dimension changed"):
        indexer.ensure_index()
