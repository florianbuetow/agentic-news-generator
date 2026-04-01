"""Tests for the knowledge-base embedding client."""

from __future__ import annotations

import json
from typing import Any, cast
from urllib.request import Request

import numpy as np
import pytest

import src.agents.article_generation.knowledge_base.embedding_client as embedding_client_module
from src.agents.article_generation.knowledge_base.embedding_client import KnowledgeBaseEmbeddingClient


class _FakeEmbeddingResponse:
    def __init__(self, *, body: str) -> None:
        self._body = body

    def __enter__(self) -> _FakeEmbeddingResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body.encode("utf-8")


def test_init_validates_provider_and_api_base() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        KnowledgeBaseEmbeddingClient(
            provider="openai",
            model_name="embed",
            api_base="http://localhost:1234/v1",
            api_key="secret",
        )

    with pytest.raises(ValueError, match="api_base must not be null"):
        KnowledgeBaseEmbeddingClient(
            provider="lmstudio",
            model_name="embed",
            api_base=None,
            api_key="secret",
        )


def test_embed_text_returns_float32_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: int) -> _FakeEmbeddingResponse:
        assert request.data is not None
        body_bytes = cast(bytes, request.data)
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(body_bytes.decode("utf-8"))
        captured["auth"] = request.headers["Authorization"]
        return _FakeEmbeddingResponse(body=json.dumps({"data": [{"embedding": [1, 2.5, 3]}]}))

    monkeypatch.setattr(embedding_client_module.urllib.request, "urlopen", fake_urlopen)

    client = KnowledgeBaseEmbeddingClient(
        provider="lmstudio",
        model_name="embed-model",
        api_base="http://localhost:1234/v1/",
        api_key="secret",
    )

    vector = client.embed_text(text="hello world", timeout_seconds=12)

    assert vector.dtype == np.float32
    assert vector.tolist() == [1.0, 2.5, 3.0]
    assert captured == {
        "url": "http://localhost:1234/v1/embeddings",
        "timeout": 12,
        "body": {"model": "embed-model", "input": ["hello world"]},
        "auth": "Bearer secret",
    }


def test_embed_text_requires_object_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int) -> _FakeEmbeddingResponse:
        return _FakeEmbeddingResponse(body='["not", "object"]')

    monkeypatch.setattr(embedding_client_module.urllib.request, "urlopen", fake_urlopen)
    client = KnowledgeBaseEmbeddingClient(
        provider="lmstudio",
        model_name="embed-model",
        api_base="http://localhost:1234/v1",
        api_key="secret",
    )

    with pytest.raises(RuntimeError, match="must be a JSON object"):
        client.embed_text(text="hello", timeout_seconds=5)


_INVALID_EMBEDDING_PAYLOADS: list[tuple[object, str]] = [
    ({"data": "bad"}, "missing data list"),
    ({"data": []}, "data list is empty"),
    ({"data": ["bad"]}, "data entry must be an object"),
    ({"data": [{"embedding": "bad"}]}, "missing embedding list"),
    ({"data": [{"embedding": [1, "bad"]}]}, "non-numeric values"),
    ({"data": [{"embedding": []}]}, "must not be empty"),
]


@pytest.mark.parametrize(("payload", "message"), _INVALID_EMBEDDING_PAYLOADS)
def test_embed_text_validates_embedding_payload(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    message: str,
) -> None:
    def fake_urlopen(request: Request, timeout: int) -> _FakeEmbeddingResponse:
        return _FakeEmbeddingResponse(body=json.dumps(payload))

    monkeypatch.setattr(embedding_client_module.urllib.request, "urlopen", fake_urlopen)
    client = KnowledgeBaseEmbeddingClient(
        provider="lmstudio",
        model_name="embed-model",
        api_base="http://localhost:1234/v1",
        api_key="secret",
    )

    with pytest.raises(RuntimeError, match=message):
        client.embed_text(text="hello", timeout_seconds=5)
