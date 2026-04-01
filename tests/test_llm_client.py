"""Tests for the LiteLLM-backed article-generation client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from urllib.request import Request

import pytest

import src.agents.article_generation.llm_client as llm_client_module
from src.agents.article_generation.llm_client import LiteLLMClient
from src.config import LLMConfig


def _make_llm_config(**overrides: Any) -> LLMConfig:
    config = LLMConfig(
        model="test-model",
        api_base="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        context_window=32768,
        max_tokens=256,
        temperature=0.3,
        context_window_threshold=90,
        max_retries=1,
        retry_delay=0.25,
        timeout_seconds=30,
    )
    return config.model_copy(update=overrides)


class _FakeURLResponse:
    def __init__(self, *, status: int) -> None:
        self.status = status

    def __enter__(self) -> _FakeURLResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


def test_check_connectivity_requests_models_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LiteLLMClient()
    captured: dict[str, object] = {}

    def fake_urlopen(request: Request, timeout: int) -> _FakeURLResponse:
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        return _FakeURLResponse(status=200)

    monkeypatch.setattr(llm_client_module.urllib.request, "urlopen", fake_urlopen)

    client.check_connectivity(api_base="http://localhost:1234/v1/", timeout_seconds=9)

    assert captured == {
        "url": "http://localhost:1234/v1/models",
        "method": "GET",
        "timeout": 9,
    }


def test_check_connectivity_wraps_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LiteLLMClient()

    def fake_urlopen(request: Request, timeout: int) -> _FakeURLResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr(llm_client_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ConnectionError, match="LLM API endpoint unreachable at http://localhost:1234/v1"):
        client.check_connectivity(api_base="http://localhost:1234/v1", timeout_seconds=3)


def test_complete_passes_expected_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LiteLLMClient()
    config = _make_llm_config()
    captured: dict[str, object] = {}

    def fake_completion(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="final response"),
                )
            ]
        )

    monkeypatch.setattr(llm_client_module, "completion", fake_completion)

    result = client.complete(
        llm_config=config,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result == "final response"
    assert captured["model"] == "test-model"
    assert captured["api_base"] == "http://127.0.0.1:1234/v1"
    assert captured["api_key"] == "lm-studio"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 256
    assert captured["timeout"] == 30
    assert captured["stream"] is False


def test_complete_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LiteLLMClient()
    config = _make_llm_config(max_retries=2, retry_delay=0.5)
    call_counter = {"count": 0}
    sleep_calls: list[float] = []

    def fake_completion(**kwargs: object) -> object:
        call_counter["count"] += 1
        if call_counter["count"] < 3:
            raise RuntimeError(f"boom-{call_counter['count']}")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="recovered"),
                )
            ]
        )

    monkeypatch.setattr(llm_client_module, "completion", fake_completion)

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(llm_client_module.time, "sleep", fake_sleep)

    result = client.complete(
        llm_config=config,
        messages=[{"role": "user", "content": "retry"}],
    )

    assert result == "recovered"
    assert call_counter["count"] == 3
    assert sleep_calls == [0.5, 0.5]


def test_complete_raises_after_retry_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LiteLLMClient()
    config = _make_llm_config(max_retries=1, retry_delay=0.25)
    sleep_calls: list[float] = []

    def fake_completion(**kwargs: object) -> object:
        raise ValueError("still broken")

    monkeypatch.setattr(llm_client_module, "completion", fake_completion)

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(llm_client_module.time, "sleep", fake_sleep)

    with pytest.raises(ValueError, match="still broken"):
        client.complete(
            llm_config=config,
            messages=[{"role": "user", "content": "fail"}],
        )

    assert sleep_calls == [0.25]


def test_complete_rejects_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LiteLLMClient()
    config = _make_llm_config(max_retries=0)

    def fake_completion(**kwargs: object) -> object:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None),
                )
            ]
        )

    monkeypatch.setattr(llm_client_module, "completion", fake_completion)

    with pytest.raises(ValueError, match="LLM returned empty content"):
        client.complete(
            llm_config=config,
            messages=[{"role": "user", "content": "empty"}],
        )
