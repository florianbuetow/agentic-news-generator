"""Tests for the LM Studio metadata client."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from src.llm import lm_studio


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
    assert lm_studio.get_lm_studio_models_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/api/v0/models"


def test_get_lm_studio_models_url_rejects_non_http_scheme() -> None:
    """Metadata requests are limited to HTTP(S) endpoints."""
    with pytest.raises(ValueError, match="Unsupported LM Studio api_base scheme"):
        lm_studio.get_lm_studio_models_url("ftp://127.0.0.1:1234/v1")


def test_get_model_id_candidates_includes_openai_prefixless_model() -> None:
    """LM Studio metadata may identify OpenAI-compatible models without the provider prefix."""
    assert lm_studio.get_model_id_candidates("openai/qwen/qwen3.6-35b-a3b") == [
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

    monkeypatch.setattr(lm_studio, "urlopen", fake_urlopen)

    assert lm_studio.get_loaded_context_length("http://127.0.0.1:1234/v1", "openai/qwen/qwen3.6-35b-a3b") == 8618


def _registry(records: list[dict[str, Any]]) -> lm_studio.LMStudioRegistry:
    """Build a registry directly from raw /api/v0/models records."""
    return lm_studio.LMStudioRegistry(
        api_base="http://127.0.0.1:1234/v1",
        records={
            record["id"]: lm_studio.ModelRecord(
                lm_studio_id=record["id"],
                state=record["state"],
                loaded_context_length=record.get("loaded_context_length"),
                max_context_length=record.get("max_context_length"),
            )
            for record in records
        },
    )


_BASE_NOT_LOADED: dict[str, Any] = {
    "id": "qwen/qwen3.6-35b-a3b",
    "state": "not-loaded",
    "max_context_length": 262144,
}
_INSTANCE_LOADED: dict[str, Any] = {
    "id": "qwen/qwen3.6-35b-a3b:2",
    "state": "loaded",
    "max_context_length": 262144,
    "loaded_context_length": 32768,
}
_INCOMPLETE_NOT_LOADED: dict[str, Any] = {
    "id": "paddleocr-vl-1.6",
    "state": "not-loaded",
}


def test_resolve_first_loaded_skips_not_loaded_exact_match() -> None:
    """A present-but-not-loaded record never matches; the walk reaches the loaded ':2' instance."""
    registry = _registry([_BASE_NOT_LOADED, _INSTANCE_LOADED])

    loaded = registry.resolve_first_loaded(
        ["openai/qwen/qwen3.6-35b-a3b", "openai/qwen/qwen3.6-35b-a3b:2"],
        min_context_window=None,
    )

    assert loaded.configured_model == "openai/qwen/qwen3.6-35b-a3b:2"
    assert loaded.lm_studio_id == "qwen/qwen3.6-35b-a3b:2"
    assert loaded.loaded_context_length == 32768
    assert loaded.max_context_length == 262144


def test_resolve_first_loaded_honours_declared_order() -> None:
    """When two configured models are both loaded, the earlier one wins."""
    first = {"id": "model-a", "state": "loaded", "max_context_length": 8192, "loaded_context_length": 4096}
    second = {"id": "model-b", "state": "loaded", "max_context_length": 8192, "loaded_context_length": 8192}
    registry = _registry([first, second])

    loaded = registry.resolve_first_loaded(["model-a", "model-b"], min_context_window=None)

    assert loaded.lm_studio_id == "model-a"


def test_resolve_first_loaded_rejects_model_below_context_floor() -> None:
    """A loaded model with too small a window is passed over for the next configured entry."""
    small = {"id": "model-a", "state": "loaded", "max_context_length": 8192, "loaded_context_length": 4096}
    big = {"id": "model-b", "state": "loaded", "max_context_length": 32768, "loaded_context_length": 32768}
    registry = _registry([small, big])

    loaded = registry.resolve_first_loaded(["model-a", "model-b"], min_context_window=8192)

    assert loaded.lm_studio_id == "model-b"


def test_resolve_first_loaded_reports_context_shortfall_when_nothing_qualifies() -> None:
    """When every loaded model is too small, the error names each shortfall."""
    small = {"id": "model-a", "state": "loaded", "max_context_length": 8192, "loaded_context_length": 4096}
    registry = _registry([small])

    with pytest.raises(lm_studio.ModelNotLoadedError) as excinfo:
        registry.resolve_first_loaded(["model-a"], min_context_window=8192)

    message = str(excinfo.value)
    assert "model-a" in message
    assert "4,096" in message
    assert "8,192" in message


def test_resolve_first_loaded_raises_when_no_model_is_loaded() -> None:
    """The error names the configured models in order and the ids that are loaded."""
    other = {"id": "glm-ocr", "state": "loaded", "max_context_length": 131072, "loaded_context_length": 32678}
    registry = _registry([_BASE_NOT_LOADED, other])

    with pytest.raises(lm_studio.ModelNotLoadedError) as excinfo:
        registry.resolve_first_loaded(["openai/qwen/qwen3.6-35b-a3b"], min_context_window=None)

    message = str(excinfo.value)
    assert "openai/qwen/qwen3.6-35b-a3b" in message
    assert "glm-ocr" in message


def test_resolve_first_loaded_raises_when_loaded_record_lacks_context_length() -> None:
    """A loaded record with no loaded_context_length is a server bug, not a skip."""
    registry = _registry([{"id": "model-a", "state": "loaded", "max_context_length": 8192}])

    with pytest.raises(lm_studio.ModelNotLoadedError, match="loaded_context_length"):
        registry.resolve_first_loaded(["model-a"], min_context_window=None)


def test_resolve_first_loaded_raises_when_loaded_record_lacks_max_context_length() -> None:
    """A selected loaded model must report its supported maximum context length."""
    registry = _registry([{"id": "model-a", "state": "loaded", "loaded_context_length": 8192}])

    with pytest.raises(lm_studio.ModelNotLoadedError, match="max_context_length"):
        registry.resolve_first_loaded(["model-a"], min_context_window=None)


def test_loaded_ids_lists_only_loaded_records() -> None:
    """loaded_ids ignores downloaded-but-not-loaded models."""
    registry = _registry([_BASE_NOT_LOADED, _INSTANCE_LOADED])

    assert registry.loaded_ids() == ["qwen/qwen3.6-35b-a3b:2"]


def test_registry_fetch_parses_lm_studio_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch() reads /api/v0/models and tolerates records with no loaded_context_length."""

    def fake_urlopen(url: str, timeout: int) -> FakeResponse:
        assert url == "http://127.0.0.1:1234/api/v0/models"
        assert timeout == 10
        return FakeResponse({"data": [_BASE_NOT_LOADED, _INSTANCE_LOADED]})

    monkeypatch.setattr(lm_studio, "urlopen", fake_urlopen)

    registry = lm_studio.LMStudioRegistry.fetch("http://127.0.0.1:1234/v1")

    assert registry.records["qwen/qwen3.6-35b-a3b"].loaded_context_length is None
    assert registry.records["qwen/qwen3.6-35b-a3b:2"].loaded_context_length == 32768


def test_registry_fetch_tolerates_unloaded_record_without_max_context_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incomplete metadata for an unrelated unloaded model must not block resolution."""

    def fake_urlopen(url: str, timeout: int) -> FakeResponse:
        return FakeResponse({"data": [_INCOMPLETE_NOT_LOADED, _BASE_NOT_LOADED, _INSTANCE_LOADED]})

    monkeypatch.setattr(lm_studio, "urlopen", fake_urlopen)

    registry = lm_studio.LMStudioRegistry.fetch("http://127.0.0.1:1234/v1")
    loaded = registry.resolve_first_loaded(
        ["openai/qwen/qwen3.6-35b-a3b", "openai/qwen/qwen3.6-35b-a3b:2"],
        min_context_window=None,
    )

    assert registry.records["paddleocr-vl-1.6"].max_context_length is None
    assert loaded.lm_studio_id == "qwen/qwen3.6-35b-a3b:2"


def test_registry_fetch_requires_api_base() -> None:
    """A None api_base cannot be resolved against."""
    with pytest.raises(ValueError, match="api_base is required"):
        lm_studio.LMStudioRegistry.fetch(None)


def test_required_context_window_maps_auto_to_no_floor() -> None:
    """'auto' imposes no context floor; an explicit window is the floor."""
    assert lm_studio.required_context_window("auto") is None
    assert lm_studio.required_context_window(131072) == 131072


def test_resolve_first_loaded_logs_each_check_and_the_pick(caplog: pytest.LogCaptureFixture) -> None:
    """The walk narrates every candidate so the chosen model is never a silent decision."""
    caplog.set_level(logging.INFO)
    registry = _registry([_BASE_NOT_LOADED, _INSTANCE_LOADED])

    registry.resolve_first_loaded(
        ["openai/absent-model", "openai/qwen/qwen3.6-35b-a3b", "openai/qwen/qwen3.6-35b-a3b:2"],
        min_context_window=None,
    )

    text = caplog.text
    assert "[1/3] checking openai/absent-model: NOT FOUND on the server — skipping" in text
    assert "[2/3] checking openai/qwen/qwen3.6-35b-a3b: found as 'qwen/qwen3.6-35b-a3b' but state=not-loaded — skipping" in text
    assert "[3/3] checking openai/qwen/qwen3.6-35b-a3b:2: FOUND and LOADED (ctx=32,768 tokens)" in text
    assert "Picking model 'qwen/qwen3.6-35b-a3b:2'" in text
    assert "skipping 2 earlier candidate(s)" in text


def test_resolve_first_loaded_logs_a_context_window_skip(caplog: pytest.LogCaptureFixture) -> None:
    """A loaded-but-too-small model is reported as a warning, not passed over quietly."""
    caplog.set_level(logging.INFO)
    small = {"id": "model-a", "state": "loaded", "max_context_length": 8192, "loaded_context_length": 4096}
    big = {"id": "model-b", "state": "loaded", "max_context_length": 32768, "loaded_context_length": 32768}
    registry = _registry([small, big])

    registry.resolve_first_loaded(["model-a", "model-b"], min_context_window=8192)

    text = caplog.text
    assert "[1/2] checking model-a: LOADED but its context window is 4,096 tokens, below the configured 8,192 — skipping" in text
    assert "[2/2] checking model-b: FOUND and LOADED (ctx=32,768 tokens)" in text
    assert "Picking model 'model-b'" in text


def test_registry_fetch_logs_loaded_models(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """fetch() reports what the server actually has loaded, so 'NOT FOUND' lines are interpretable."""
    caplog.set_level(logging.INFO)

    def fake_urlopen(url: str, timeout: int) -> FakeResponse:
        return FakeResponse({"data": [_BASE_NOT_LOADED, _INSTANCE_LOADED]})

    monkeypatch.setattr(lm_studio, "urlopen", fake_urlopen)

    lm_studio.LMStudioRegistry.fetch("http://127.0.0.1:1234/v1")

    assert "reports 2 model(s); loaded right now: ['qwen/qwen3.6-35b-a3b:2']" in caplog.text


def test_find_matches_exact_id_and_strips_openai_prefix() -> None:
    """find() locates a record regardless of the litellm provider prefix, and ignores state."""
    registry = _registry([_BASE_NOT_LOADED, _INSTANCE_LOADED])

    prefixed = registry.find("openai/qwen/qwen3.6-35b-a3b")
    suffixed = registry.find("openai/qwen/qwen3.6-35b-a3b:2")

    assert prefixed is not None
    assert prefixed.state == "not-loaded"
    assert suffixed is not None
    assert suffixed.state == "loaded"
    assert registry.find("openai/nope") is None
