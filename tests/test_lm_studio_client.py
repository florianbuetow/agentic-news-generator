"""Tests for the LM Studio metadata client."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.summarize import lm_studio_client


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_get_lm_studio_models_url_uses_native_metadata_endpoint() -> None:
    """The OpenAI-compatible API path is replaced by LM Studio's metadata path."""
    assert lm_studio_client.get_lm_studio_models_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/api/v0/models"


def test_get_lm_studio_models_url_rejects_non_http_scheme() -> None:
    """Metadata requests are limited to HTTP(S) endpoints."""
    with pytest.raises(ValueError, match="Unsupported LM Studio api_base scheme"):
        lm_studio_client.get_lm_studio_models_url("ftp://127.0.0.1:1234/v1")


def test_get_model_id_candidates_includes_openai_prefixless_model() -> None:
    """LM Studio metadata may identify OpenAI-compatible models without the provider prefix."""
    assert lm_studio_client.get_model_id_candidates("openai/qwen/qwen3.6-35b-a3b") == [
        "openai/qwen/qwen3.6-35b-a3b",
        "qwen/qwen3.6-35b-a3b",
    ]


def test_get_loaded_context_length_reads_lm_studio_loaded_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client resolves the loaded context length from LM Studio model metadata."""

    def fake_urlopen(url: str, timeout: int) -> FakeResponse:
        assert url == "http://127.0.0.1:1234/api/v0/models"
        assert timeout == 10
        return FakeResponse(
            {
                "data": [
                    {
                        "id": "qwen/qwen3.6-35b-a3b",
                        "state": "loaded",
                        "max_context_length": 262144,
                        "loaded_context_length": 8618,
                    }
                ]
            }
        )

    monkeypatch.setattr(lm_studio_client, "urlopen", fake_urlopen)

    assert lm_studio_client.get_loaded_context_length("http://127.0.0.1:1234/v1", "openai/qwen/qwen3.6-35b-a3b") == 8618
