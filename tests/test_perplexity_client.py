"""Tests for the Perplexity HTTP client wrapper."""

from __future__ import annotations

import json

import pytest

import src.agents.article_generation.perplexity_client as perplexity_client_module
from src.agents.article_generation.perplexity_client import PerplexityHTTPClient


class _FakeHTTPResponse:
    def __init__(self, *, status: int, body: str) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body.encode("utf-8")


class _RecordingConnection:
    def __init__(self, *, response: _FakeHTTPResponse) -> None:
        self.response = response
        self.closed = False
        self.requests: list[dict[str, object]] = []

    def request(self, *, method: str, url: str, body: str, headers: dict[str, str]) -> None:
        self.requests.append(
            {
                "method": method,
                "url": url,
                "body": body,
                "headers": headers,
            }
        )

    def getresponse(self) -> _FakeHTTPResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


def _connection_factory(connection: object) -> object:
    def factory(host: str, timeout: int) -> object:
        return connection

    return factory


def test_init_requires_https() -> None:
    with pytest.raises(ValueError, match="must use https"):
        PerplexityHTTPClient(api_base="http://api.perplexity.ai/v1", api_key="secret")


def test_init_requires_host() -> None:
    with pytest.raises(ValueError, match="must include host"):
        PerplexityHTTPClient(api_base="https:///v1", api_key="secret")


def test_search_posts_chat_completion_request(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeHTTPResponse(
        status=200,
        body=json.dumps({"id": "abc", "citations": ["https://example.com"]}),
    )
    connection = _RecordingConnection(response=response)
    captured_constructor_args: dict[str, object] = {}

    def fake_connection(host: str, timeout: int) -> _RecordingConnection:
        captured_constructor_args["host"] = host
        captured_constructor_args["timeout"] = timeout
        return connection

    monkeypatch.setattr(perplexity_client_module.http.client, "HTTPSConnection", fake_connection)

    client = PerplexityHTTPClient(api_base="https://api.perplexity.ai/v1/", api_key="secret")
    result = client.search(query="What is new?", model="sonar", timeout_seconds=15)

    assert result["citations"] == ["https://example.com"]
    assert captured_constructor_args == {"host": "api.perplexity.ai", "timeout": 15}
    assert connection.closed is True
    assert len(connection.requests) == 1

    request = connection.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "/v1/chat/completions"
    assert request["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer secret",
    }
    assert json.loads(str(request["body"])) == {
        "model": "sonar",
        "messages": [{"role": "user", "content": "What is new?"}],
    }


def test_search_normalizes_missing_citations_to_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeHTTPResponse(
        status=200,
        body=json.dumps({"id": "abc", "citations": "not-a-list"}),
    )
    connection = _RecordingConnection(response=response)
    monkeypatch.setattr(perplexity_client_module.http.client, "HTTPSConnection", _connection_factory(connection))

    client = PerplexityHTTPClient(api_base="https://api.perplexity.ai", api_key="secret")
    result = client.search(query="Q", model="sonar", timeout_seconds=7)

    assert result["citations"] == []


def test_search_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeHTTPResponse(status=429, body='{"error":"rate limit"}')
    connection = _RecordingConnection(response=response)
    monkeypatch.setattr(perplexity_client_module.http.client, "HTTPSConnection", _connection_factory(connection))

    client = PerplexityHTTPClient(api_base="https://api.perplexity.ai", api_key="secret")

    with pytest.raises(RuntimeError, match="status 429"):
        client.search(query="Q", model="sonar", timeout_seconds=7)

    assert connection.closed is True


def test_search_wraps_socket_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExplodingConnection:
        def __init__(self) -> None:
            self.closed = False

        def request(self, *, method: str, url: str, body: str, headers: dict[str, str]) -> None:
            raise OSError("network down")

        def close(self) -> None:
            self.closed = True

    exploding = _ExplodingConnection()
    monkeypatch.setattr(perplexity_client_module.http.client, "HTTPSConnection", _connection_factory(exploding))

    client = PerplexityHTTPClient(api_base="https://api.perplexity.ai", api_key="secret")

    with pytest.raises(RuntimeError, match="Perplexity request failed: network down"):
        client.search(query="Q", model="sonar", timeout_seconds=7)

    assert exploding.closed is True


def test_search_requires_json_object_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeHTTPResponse(status=200, body='["not", "an", "object"]')
    connection = _RecordingConnection(response=response)
    monkeypatch.setattr(perplexity_client_module.http.client, "HTTPSConnection", _connection_factory(connection))

    client = PerplexityHTTPClient(api_base="https://api.perplexity.ai", api_key="secret")

    with pytest.raises(RuntimeError, match="must be a JSON object"):
        client.search(query="Q", model="sonar", timeout_seconds=7)


def test_search_rejects_non_string_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeHTTPResponse(status=200, body='{"ok": true}')
    connection = _RecordingConnection(response=response)

    def fake_json_loads(raw: str) -> dict[int, str]:
        return {1: "bad"}

    monkeypatch.setattr(perplexity_client_module.http.client, "HTTPSConnection", _connection_factory(connection))
    monkeypatch.setattr(perplexity_client_module.json, "loads", fake_json_loads)

    client = PerplexityHTTPClient(api_base="https://api.perplexity.ai", api_key="secret")

    with pytest.raises(RuntimeError, match="non-string key"):
        client.search(query="Q", model="sonar", timeout_seconds=7)
