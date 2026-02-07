"""Knowledge base index builder for article-generation fact checking."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import tiktoken

from src.agents.article_generation.knowledge_base.embedding_client import KnowledgeBaseEmbeddingClient

logger = logging.getLogger(__name__)


class IndexedChunk(TypedDict):
    """Serialized chunk entry for persisted KB index."""

    source_path: str
    chunk_id: int
    content: str


class KnowledgeBaseIndexer:
    """Builds or loads a persisted knowledge-base index."""

    _MANIFEST_FILE = "manifest.json"
    _CHUNKS_FILE = "chunks.json"
    _EMBEDDINGS_FILE = "embeddings.npy"

    def __init__(
        self,
        *,
        data_dir: Path,
        index_dir: Path,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
        embedding_provider: str,
        embedding_model_name: str,
        embedding_api_base: str | None,
        embedding_api_key: str,
        embedding_timeout_seconds: int,
        encoding_name: str,
    ) -> None:
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("Knowledge base chunk_overlap_tokens must be smaller than chunk_size_tokens")

        self._data_dir = data_dir
        self._index_dir = index_dir
        self._chunk_size_tokens = chunk_size_tokens
        self._chunk_overlap_tokens = chunk_overlap_tokens
        self._embedding_provider = embedding_provider
        self._embedding_model_name = embedding_model_name
        self._embedding_api_base = embedding_api_base
        self._embedding_api_key = embedding_api_key
        self._embedding_timeout_seconds = embedding_timeout_seconds
        self._encoding_name = encoding_name

        self._embedding_client = KnowledgeBaseEmbeddingClient(
            provider=embedding_provider,
            model_name=embedding_model_name,
            api_base=embedding_api_base,
            api_key=embedding_api_key,
        )

    def ensure_index(self) -> str:
        """Ensure index exists and return the active index version."""
        settings_hash = self._settings_hash()
        docs_hash = self._documents_hash()
        active_index_version = self._index_version(settings_hash=settings_hash, docs_hash=docs_hash)

        manifest = self._load_manifest()
        if manifest is not None and self._index_is_current(manifest=manifest, settings_hash=settings_hash, docs_hash=docs_hash):
            logger.info("Knowledge base index is current: %s", active_index_version)
            return active_index_version

        logger.info("Building knowledge base index at %s", self._index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        chunks = self._build_chunks()
        embeddings = self._build_embeddings(chunks=chunks)

        chunks_path = self._index_dir / self._CHUNKS_FILE
        with open(chunks_path, "w", encoding="utf-8") as chunks_file:
            json.dump(chunks, chunks_file, ensure_ascii=False, indent=2)

        embeddings_path = self._index_dir / self._EMBEDDINGS_FILE
        np.save(embeddings_path, embeddings)

        manifest_payload = {
            "index_version": active_index_version,
            "settings_hash": settings_hash,
            "documents_hash": docs_hash,
            "chunk_count": len(chunks),
            "embedding_provider": self._embedding_provider,
            "embedding_model_name": self._embedding_model_name,
        }
        manifest_path = self._index_dir / self._MANIFEST_FILE
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest_payload, manifest_file, ensure_ascii=False, indent=2)

        logger.info("Knowledge base index built: version=%s chunks=%d", active_index_version, len(chunks))
        return active_index_version

    def _load_manifest(self) -> dict[str, object] | None:
        """Load persisted manifest if present and valid JSON."""
        manifest_path = self._index_dir / self._MANIFEST_FILE
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, encoding="utf-8") as manifest_file:
                data = json.load(manifest_file)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        typed_data = cast(dict[str, object], data)
        return {str(key): value for key, value in typed_data.items()}

    def _index_is_current(self, *, manifest: dict[str, object], settings_hash: str, docs_hash: str) -> bool:
        """Check whether existing index matches current settings/documents."""
        chunks_path = self._index_dir / self._CHUNKS_FILE
        embeddings_path = self._index_dir / self._EMBEDDINGS_FILE
        if not chunks_path.exists() or not embeddings_path.exists():
            return False
        manifest_settings_hash = manifest.get("settings_hash")
        manifest_documents_hash = manifest.get("documents_hash")
        return manifest_settings_hash == settings_hash and manifest_documents_hash == docs_hash

    def _settings_hash(self) -> str:
        """Compute hash of embedding and chunking settings."""
        payload = "|".join(
            [
                self._embedding_provider,
                self._embedding_model_name,
                str(self._embedding_api_base),
                str(self._chunk_size_tokens),
                str(self._chunk_overlap_tokens),
                self._encoding_name,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _documents_hash(self) -> str:
        """Compute hash of knowledge-base documents content."""
        digest = hashlib.sha256()
        for file_path in self._source_files():
            digest.update(str(file_path.relative_to(self._data_dir)).encode("utf-8"))
            digest.update(file_path.read_bytes())
        return digest.hexdigest()

    def _index_version(self, *, settings_hash: str, docs_hash: str) -> str:
        """Compute deterministic index version string."""
        payload = f"{settings_hash}|{docs_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _source_files(self) -> list[Path]:
        """Return sorted source files included in the KB."""
        txt_files = sorted(self._data_dir.glob("*.txt"))
        md_files = sorted(self._data_dir.glob("*.md"))
        return txt_files + md_files

    def _build_chunks(self) -> list[IndexedChunk]:
        """Build chunk list from knowledge-base source documents."""
        if not self._data_dir.exists():
            raise FileNotFoundError(f"Knowledge base data_dir does not exist: {self._data_dir}")

        encoding = tiktoken.get_encoding(self._encoding_name)
        stride = self._chunk_size_tokens - self._chunk_overlap_tokens

        chunks: list[IndexedChunk] = []
        for source_file in self._source_files():
            content = source_file.read_text(encoding="utf-8")
            if content.strip() == "":
                continue

            token_ids = encoding.encode(content)
            if len(token_ids) == 0:
                continue

            chunk_id = 0
            for start in range(0, len(token_ids), stride):
                end = min(start + self._chunk_size_tokens, len(token_ids))
                chunk_token_ids = token_ids[start:end]
                chunk_text = encoding.decode(chunk_token_ids).strip()
                if chunk_text == "":
                    continue
                chunks.append(
                    IndexedChunk(
                        source_path=str(source_file),
                        chunk_id=chunk_id,
                        content=chunk_text,
                    )
                )
                chunk_id += 1
                if end >= len(token_ids):
                    break
        return chunks

    def _build_embeddings(self, *, chunks: list[IndexedChunk]) -> npt.NDArray[np.float32]:
        """Build embedding matrix for all chunks."""
        if len(chunks) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        first_vector = self._embedding_client.embed_text(
            text=chunks[0]["content"],
            timeout_seconds=self._embedding_timeout_seconds,
        )
        expected_dim = int(first_vector.shape[0])
        vectors: list[npt.NDArray[np.float32]] = [first_vector]

        for chunk in chunks[1:]:
            vector = self._embedding_client.embed_text(
                text=chunk["content"],
                timeout_seconds=self._embedding_timeout_seconds,
            )
            if int(vector.shape[0]) != expected_dim:
                raise RuntimeError("Embedding dimension changed within a single index build")
            vectors.append(vector)

        stacked: npt.NDArray[np.float32] = np.stack(vectors).astype(np.float32)
        return stacked
