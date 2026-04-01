"""Tests for the local knowledge-base retriever."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import src.agents.article_generation.knowledge_base.retriever as retriever_module
from src.agents.article_generation.knowledge_base.retriever import HaystackKnowledgeBaseRetriever


class _FakeEmbeddingClient:
    last_instance: _FakeEmbeddingClient | None = None

    def __init__(self, **kwargs: object) -> None:
        self.calls: list[dict[str, object]] = []
        _FakeEmbeddingClient.last_instance = self

    def embed_text(self, *, text: str, timeout_seconds: int) -> npt.NDArray[np.float32]:
        self.calls.append({"text": text, "timeout_seconds": timeout_seconds})
        if text == "query":
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)


def _write_index(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "chunks.json").write_text(
        json.dumps(
            [
                {"source_path": "a.txt", "chunk_id": 0, "content": "alpha"},
                {"source_path": "b.txt", "chunk_id": 1, "content": "beta"},
            ]
        ),
        encoding="utf-8",
    )
    np.save(index_dir / "embeddings.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def _make_retriever(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HaystackKnowledgeBaseRetriever:
    monkeypatch.setattr(retriever_module, "KnowledgeBaseEmbeddingClient", _FakeEmbeddingClient)
    index_dir = tmp_path / "index"
    _write_index(index_dir)
    return HaystackKnowledgeBaseRetriever(
        index_dir=index_dir,
        embedding_provider="lmstudio",
        embedding_model_name="embed-model",
        embedding_api_base="http://localhost:1234/v1",
        embedding_api_key="secret",
        embedding_timeout_seconds=9,
    )


def test_search_returns_ranked_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = _make_retriever(tmp_path, monkeypatch)

    results = retriever.search(query="query", top_k=2, timeout_seconds=30)

    assert [result["source_path"] for result in results] == ["a.txt", "b.txt"]
    assert results[0]["chunk_id"] == "0"
    assert results[0]["snippet"] == "alpha"
    assert float(results[0]["score"]) > float(results[1]["score"])
    assert _FakeEmbeddingClient.last_instance is not None
    assert _FakeEmbeddingClient.last_instance.calls == [{"text": "query", "timeout_seconds": 9}]


def test_search_returns_empty_for_blank_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = _make_retriever(tmp_path, monkeypatch)

    assert retriever.search(query="   ", top_k=2, timeout_seconds=30) == []


def test_search_rejects_non_positive_top_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = _make_retriever(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="greater than 0"):
        retriever.search(query="query", top_k=0, timeout_seconds=30)


def test_search_rejects_dimension_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _WrongDimEmbeddingClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def embed_text(self, *, text: str, timeout_seconds: int) -> npt.NDArray[np.float32]:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(retriever_module, "KnowledgeBaseEmbeddingClient", _WrongDimEmbeddingClient)
    index_dir = tmp_path / "index"
    _write_index(index_dir)
    retriever = HaystackKnowledgeBaseRetriever(
        index_dir=index_dir,
        embedding_provider="lmstudio",
        embedding_model_name="embed-model",
        embedding_api_base="http://localhost:1234/v1",
        embedding_api_key="secret",
        embedding_timeout_seconds=9,
    )

    with pytest.raises(RuntimeError, match="dimension does not match"):
        retriever.search(query="query", top_k=2, timeout_seconds=30)


def test_retriever_requires_matching_chunk_and_embedding_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retriever_module, "KnowledgeBaseEmbeddingClient", _FakeEmbeddingClient)
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "chunks.json").write_text(
        json.dumps([{"source_path": "a.txt", "chunk_id": 0, "content": "alpha"}]),
        encoding="utf-8",
    )
    np.save(index_dir / "embeddings.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    with pytest.raises(RuntimeError, match="row count must match chunk count"):
        HaystackKnowledgeBaseRetriever(
            index_dir=index_dir,
            embedding_provider="lmstudio",
            embedding_model_name="embed-model",
            embedding_api_base="http://localhost:1234/v1",
            embedding_api_key="secret",
            embedding_timeout_seconds=9,
        )


def test_load_chunks_requires_structured_objects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retriever_module, "KnowledgeBaseEmbeddingClient", _FakeEmbeddingClient)
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "chunks.json").write_text(json.dumps(["bad-entry"]), encoding="utf-8")
    np.save(index_dir / "embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32))

    with pytest.raises(RuntimeError, match="entries must be objects"):
        HaystackKnowledgeBaseRetriever(
            index_dir=index_dir,
            embedding_provider="lmstudio",
            embedding_model_name="embed-model",
            embedding_api_base="http://localhost:1234/v1",
            embedding_api_key="secret",
            embedding_timeout_seconds=9,
        )
