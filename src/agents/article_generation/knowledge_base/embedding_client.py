"""OpenAI-compatible embedding client for article-generation knowledge base."""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import cast

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class KnowledgeBaseEmbeddingClient:
    """HTTP client for embedding generation against OpenAI-compatible APIs."""

    def __init__(
        self,
        *,
        provider: str,
        model_name: str,
        api_base: str | None,
        api_key: str,
    ) -> None:
        if provider != "lmstudio":
            raise ValueError(f"Unsupported embedding provider: {provider}")
        if api_base is None:
            raise ValueError("Embedding api_base must not be null")

        self._provider = provider
        self._model_name = model_name
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key

    def embed_text(self, *, text: str, timeout_seconds: int) -> npt.NDArray[np.float32]:
        """Embed a single text and return a float32 vector."""
        request_url = f"{self._api_base}/embeddings"
        payload = {"model": self._model_name, "input": [text]}
        request_body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            request_url,
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        logger.debug("Embedding request: provider=%s model=%s chars=%d", self._provider, self._model_name, len(text))
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
            raw_body = response.read().decode("utf-8")

        parsed_unknown = json.loads(raw_body)
        if not isinstance(parsed_unknown, dict):
            raise RuntimeError("Embedding response must be a JSON object")

        parsed = cast(dict[object, object], parsed_unknown)
        data_value = parsed.get("data")
        if not isinstance(data_value, list):
            raise RuntimeError("Embedding response missing data list")
        typed_data_value = cast(list[object], data_value)
        if len(typed_data_value) == 0:
            raise RuntimeError("Embedding response data list is empty")

        first_item: object = typed_data_value[0]
        if not isinstance(first_item, dict):
            raise RuntimeError("Embedding response data entry must be an object")

        first_item_dict = cast(dict[object, object], first_item)
        embedding_value = first_item_dict.get("embedding")
        if not isinstance(embedding_value, list):
            raise RuntimeError("Embedding response entry missing embedding list")

        typed_embedding_items = cast(list[object], embedding_value)
        vector: list[float] = []
        for item in typed_embedding_items:
            if not isinstance(item, (int, float)):
                raise RuntimeError("Embedding vector contains non-numeric values")
            vector.append(float(item))

        if len(vector) == 0:
            raise RuntimeError("Embedding vector must not be empty")
        return np.array(vector, dtype=np.float32)
