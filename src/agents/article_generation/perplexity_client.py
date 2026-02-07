"""Perplexity OpenAI-compatible HTTP client wrapper."""

from __future__ import annotations

import http.client
import json
import logging
import time
from typing import cast
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


class PerplexityHTTPClient:
    """Thin client for Perplexity's OpenAI-compatible API."""

    def __init__(self, *, api_base: str, api_key: str) -> None:
        parsed_api_base = urlsplit(api_base.rstrip("/"))
        if parsed_api_base.scheme != "https":
            raise ValueError("Perplexity api_base must use https")
        if parsed_api_base.netloc == "":
            raise ValueError("Perplexity api_base must include host")

        self._api_host = parsed_api_base.netloc
        self._api_base_path = parsed_api_base.path.rstrip("/")
        self._api_key = api_key

    def search(self, *, query: str, model: str, timeout_seconds: int) -> dict[str, object]:
        """Execute a search call and return parsed response payload."""
        logger.info("Perplexity search: model=%s host=%s timeout=%ds query_chars=%d", model, self._api_host, timeout_seconds, len(query))
        started_at = time.perf_counter()
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ],
        }
        request_path = f"{self._api_base_path}/chat/completions"
        request_body = json.dumps(payload)
        connection: http.client.HTTPSConnection | None = None
        try:
            connection = http.client.HTTPSConnection(self._api_host, timeout=timeout_seconds)
            connection.request(
                method="POST",
                url=request_path,
                body=request_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            response = connection.getresponse()
            raw_body = response.read().decode("utf-8")
            if response.status >= 400:
                raise RuntimeError(f"Perplexity request failed with status {response.status}: {raw_body}")
        except OSError as exc:
            elapsed = time.perf_counter() - started_at
            logger.error("Perplexity request failed after %.1fs: %s: %s", elapsed, type(exc).__name__, exc)
            raise RuntimeError(f"Perplexity request failed: {exc}") from exc
        finally:
            if connection is not None:
                connection.close()

        parsed_raw_unknown = json.loads(raw_body)
        if not isinstance(parsed_raw_unknown, dict):
            raise RuntimeError("Perplexity response payload must be a JSON object")
        parsed_raw = cast(dict[object, object], parsed_raw_unknown)

        parsed: dict[str, object] = {}
        for key, value in parsed_raw.items():
            if not isinstance(key, str):
                raise RuntimeError("Perplexity response payload contains non-string key")
            parsed[key] = value

        citations = parsed.get("citations")
        if not isinstance(citations, list):
            parsed["citations"] = []
            citation_count = 0
        else:
            citation_count = len(cast(list[object], citations))
        elapsed = time.perf_counter() - started_at
        logger.info("Perplexity search completed in %.1fs: %d citations, response_chars=%d", elapsed, citation_count, len(raw_body))
        return parsed
