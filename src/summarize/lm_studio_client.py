"""LM Studio metadata client for transcript summarization."""

from __future__ import annotations

import json
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen


def get_lm_studio_models_url(api_base: str | None) -> str:
    """Return the native LM Studio models endpoint for an OpenAI-compatible API base."""
    models_path = "/api/v0/models"
    if api_base is None:
        raise ValueError("api_base is required to query LM Studio model metadata")

    parsed = urlparse(api_base)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid LM Studio api_base: {api_base}")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported LM Studio api_base scheme: {parsed.scheme}")

    return urlunparse((parsed.scheme, parsed.netloc, models_path, "", "", ""))


def get_model_id_candidates(model: str) -> list[str]:
    """Return model IDs to try against LM Studio metadata."""
    candidates = [model]
    if model.startswith("openai/"):
        candidates.append(model.removeprefix("openai/"))
    return candidates


def fetch_lm_studio_models(models_url: str) -> list[object]:
    """Fetch LM Studio model metadata."""
    try:
        with urlopen(models_url, timeout=10) as response:  # nosec B310
            payload = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
    except HTTPError as exc:
        raise RuntimeError(f"LM Studio metadata request failed with HTTP {exc.code}: {models_url}") from exc
    except URLError as exc:
        raise RuntimeError(f"LM Studio metadata endpoint is unreachable: {models_url} ({exc.reason})") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"LM Studio metadata request timed out: {models_url}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LM Studio metadata response was not valid JSON: {models_url}") from exc

    data_raw = payload.get("data")
    if not isinstance(data_raw, list):
        raise RuntimeError("LM Studio metadata response did not contain a data list")

    return cast(list[object], data_raw)


def extract_loaded_context_length(model_info: dict[str, Any], model: str) -> int:
    """Extract the loaded context length from one LM Studio model record."""
    if model_info.get("state") != "loaded":
        raise RuntimeError(f"Configured model is not loaded in LM Studio: {model}")
    loaded_context_length = model_info.get("loaded_context_length")
    if not isinstance(loaded_context_length, int) or loaded_context_length <= 0:
        raise RuntimeError(f"Loaded LM Studio model does not report loaded_context_length: {model}")
    return loaded_context_length


def get_loaded_context_length(api_base: str | None, model: str) -> int:
    """Fetch the loaded context length for the configured LM Studio model."""
    data = fetch_lm_studio_models(get_lm_studio_models_url(api_base))
    candidates = set(get_model_id_candidates(model))
    loaded_context_length: int | None = None
    for model_info_raw in data:
        if not isinstance(model_info_raw, dict):
            continue
        model_info = cast(dict[str, Any], model_info_raw)
        if model_info.get("id") not in candidates:
            continue
        loaded_context_length = extract_loaded_context_length(model_info, model)
        break

    if loaded_context_length is not None:
        return loaded_context_length

    raise RuntimeError(f"Configured model was not found in LM Studio metadata: {model}")
