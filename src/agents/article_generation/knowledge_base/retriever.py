"""Knowledge base retrieval implementation for fact-checking agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt

from src.agents.article_generation.knowledge_base.embedding_client import KnowledgeBaseEmbeddingClient

logger = logging.getLogger(__name__)


class IndexedChunk(TypedDict):
    """Serialized chunk entry loaded from the KB index."""

    source_path: str
    chunk_id: int
    content: str


class HaystackKnowledgeBaseRetriever:
    """Retrieves evidence snippets from a prebuilt local index."""

    _CHUNKS_FILE = "chunks.json"
    _EMBEDDINGS_FILE = "embeddings.npy"

    def __init__(
        self,
        *,
        index_dir: Path,
        embedding_provider: str,
        embedding_model_name: str,
        embedding_api_base: str | None,
        embedding_api_key: str,
        embedding_timeout_seconds: int,
    ) -> None:
        self._index_dir = index_dir
        self._embedding_timeout_seconds = embedding_timeout_seconds
        self._embedding_client = KnowledgeBaseEmbeddingClient(
            provider=embedding_provider,
            model_name=embedding_model_name,
            api_base=embedding_api_base,
            api_key=embedding_api_key,
        )
        self._chunks = self._load_chunks()
        self._embeddings = self._load_embeddings()

        if self._embeddings.ndim != 2:
            raise RuntimeError("Knowledge base embeddings file must contain a 2D array")
        if self._embeddings.shape[0] != len(self._chunks):
            raise RuntimeError("Knowledge base embeddings row count must match chunk count")

    def search(self, *, query: str, top_k: int, timeout_seconds: int) -> list[dict[str, str]]:
        """Search the index and return evidence snippets."""
        if query.strip() == "":
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self._embeddings.shape[0] == 0:
            return []

        timeout = min(timeout_seconds, self._embedding_timeout_seconds)
        query_vector = self._embedding_client.embed_text(text=query, timeout_seconds=timeout)

        if self._embeddings.shape[1] != int(query_vector.shape[0]):
            raise RuntimeError("Query embedding dimension does not match index embedding dimension")

        scores = self._cosine_similarity(self._embeddings, query_vector)
        candidate_count = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:candidate_count]

        results: list[dict[str, str]] = []
        for idx in top_indices:
            chunk = self._chunks[int(idx)]
            results.append(
                {
                    "source_path": chunk["source_path"],
                    "chunk_id": str(chunk["chunk_id"]),
                    "snippet": chunk["content"],
                    "score": f"{float(scores[int(idx)]):.6f}",
                }
            )
        logger.info("Knowledge base search returned %d results for query_chars=%d", len(results), len(query))
        return results

    def _load_chunks(self) -> list[IndexedChunk]:
        """Load chunk metadata from persisted JSON."""
        chunks_path = self._index_dir / self._CHUNKS_FILE
        if not chunks_path.exists():
            raise FileNotFoundError(f"Knowledge base chunks file does not exist: {chunks_path}")

        with open(chunks_path, encoding="utf-8") as chunks_file:
            raw_chunks = json.load(chunks_file)

        if not isinstance(raw_chunks, list):
            raise RuntimeError("Knowledge base chunks file must contain a list")

        chunks: list[IndexedChunk] = []
        typed_items = cast(list[object], raw_chunks)
        for item in typed_items:
            if not isinstance(item, dict):
                raise RuntimeError("Knowledge base chunk entries must be objects")
            typed_item = cast(dict[object, object], item)
            source_path = typed_item.get("source_path")
            chunk_id = typed_item.get("chunk_id")
            content = typed_item.get("content")
            if not isinstance(source_path, str):
                raise RuntimeError("Knowledge base chunk source_path must be a string")
            if not isinstance(chunk_id, int):
                raise RuntimeError("Knowledge base chunk chunk_id must be an int")
            if not isinstance(content, str):
                raise RuntimeError("Knowledge base chunk content must be a string")
            chunks.append(IndexedChunk(source_path=source_path, chunk_id=chunk_id, content=content))
        return chunks

    def _load_embeddings(self) -> npt.NDArray[np.float32]:
        """Load persisted embeddings matrix."""
        embeddings_path = self._index_dir / self._EMBEDDINGS_FILE
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Knowledge base embeddings file does not exist: {embeddings_path}")
        embeddings = np.load(embeddings_path)
        if not isinstance(embeddings, np.ndarray):
            raise RuntimeError("Knowledge base embeddings file did not load as ndarray")
        return embeddings.astype(np.float32)

    def _cosine_similarity(self, matrix: npt.NDArray[np.float32], query_vector: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Compute cosine similarity for each row in matrix against query vector."""
        matrix_norm = np.linalg.norm(matrix, axis=1)
        query_norm = float(np.linalg.norm(query_vector))
        denominator = matrix_norm * query_norm
        safe_denominator = np.where(denominator == 0.0, 1e-12, denominator)
        result: npt.NDArray[np.float32] = (matrix @ query_vector) / safe_denominator
        return result
