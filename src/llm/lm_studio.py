"""LM Studio metadata client for transcript summarization."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

from src.util.log_util import get_logger

logger: logging.Logger = get_logger(__name__)

_MODEL_UNAVAILABLE_ERROR_FRAGMENTS = (
    "no models loaded",
    "model unloaded",
    "failed to load model",
)


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


def get_openai_models_url(api_base: str | None) -> str:
    """Return the OpenAI-compatible models endpoint for an API base."""
    if api_base is None:
        raise ValueError("api_base is required to query LM Studio model metadata")

    parsed = urlparse(api_base)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid LM Studio api_base: {api_base}")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported LM Studio api_base scheme: {parsed.scheme}")

    models_path = f"{parsed.path.rstrip('/')}/models"
    return urlunparse((parsed.scheme, parsed.netloc, models_path, "", "", ""))


def get_model_id_candidates(model: str) -> list[str]:
    """Return model IDs to try against LM Studio metadata."""
    candidates = [model]
    if model.startswith("openai/"):
        candidates.append(model.removeprefix("openai/"))
    return candidates


def is_model_unavailable_error(message: str) -> bool:
    """Return whether LM Studio reports that the requested model is unavailable."""
    normalized = message.lower()
    return any(fragment in normalized for fragment in _MODEL_UNAVAILABLE_ERROR_FRAGMENTS)


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


def fetch_model_ids(models_url: str) -> set[str]:
    """Fetch model identifiers from an OpenAI-compatible models endpoint."""
    model_ids: set[str] = set()
    for entry in fetch_lm_studio_models(models_url):
        if not isinstance(entry, dict):
            continue
        info = cast(dict[str, Any], entry)
        model_id = info.get("id")
        if isinstance(model_id, str) and model_id:
            model_ids.add(model_id)
    return model_ids


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


def required_context_window(context_window: int | Literal["auto"]) -> int | None:
    """Return the context floor a configured window imposes; 'auto' imposes none."""
    if context_window == "auto":
        return None
    return context_window


class ModelNotLoadedError(RuntimeError):
    """No configured model is loaded on the LM Studio server."""


class ModelAutoloadPolicyError(ModelNotLoadedError):
    """Server-side JIT policy cannot satisfy loaded-only inference."""


@dataclass(frozen=True)
class ModelRecord:
    """One record from LM Studio's /api/v0/models."""

    lm_studio_id: str
    state: str
    loaded_context_length: int | None
    max_context_length: int | None


@dataclass(frozen=True)
class LoadedModel:
    """A configured model confirmed loaded on the server."""

    configured_model: str
    lm_studio_id: str
    loaded_context_length: int
    max_context_length: int


@dataclass(frozen=True)
class LMStudioRegistry:
    """Immutable snapshot of /api/v0/models for one api_base."""

    api_base: str
    records: dict[str, ModelRecord]

    @classmethod
    def fetch(cls, api_base: str | None) -> LMStudioRegistry:
        """Snapshot every model record the server reports."""
        return cls._fetch(api_base, log_snapshot=True)

    @classmethod
    def fetch_quietly(cls, api_base: str | None) -> LMStudioRegistry:
        """Snapshot model records without emitting a per-request info message."""
        return cls._fetch(api_base, log_snapshot=False)

    @classmethod
    def _fetch(cls, api_base: str | None, *, log_snapshot: bool) -> LMStudioRegistry:
        """Build a registry snapshot with explicit logging behavior."""
        if api_base is None:
            raise ValueError("api_base is required to query LM Studio model metadata")

        records: dict[str, ModelRecord] = {}
        for entry in fetch_lm_studio_models(get_lm_studio_models_url(api_base)):
            if not isinstance(entry, dict):
                continue
            info = cast(dict[str, Any], entry)
            loaded_context_length = info.get("loaded_context_length")
            if not isinstance(loaded_context_length, int) or loaded_context_length <= 0:
                loaded_context_length = None
            max_context_length = info.get("max_context_length")
            if not isinstance(max_context_length, int) or max_context_length <= 0:
                max_context_length = None
            model_id = str(info["id"])
            records[model_id] = ModelRecord(
                lm_studio_id=model_id,
                state=str(info["state"]),
                loaded_context_length=loaded_context_length,
                max_context_length=max_context_length,
            )

        registry = cls(api_base=api_base, records=records)
        if log_snapshot:
            logger.info(
                "LM Studio at %s reports %d model(s); loaded right now: %s",
                api_base,
                len(records),
                registry.loaded_ids(),
            )
        return registry

    def loaded_ids(self) -> list[str]:
        """Return the ids of every model currently loaded, sorted."""
        return sorted(record.lm_studio_id for record in self.records.values() if record.state == "loaded")

    def find(self, model: str) -> ModelRecord | None:
        """Find a configured model's record by exact id, tolerating the litellm 'openai/' prefix."""
        for candidate in get_model_id_candidates(model):
            record = self.records.get(candidate)
            if record is not None:
                return record
        return None

    def require_loaded(self, model: str) -> ModelRecord:
        """Return an exact loaded model record or fail before an inference request is sent."""
        record = self.find(model)
        if record is None or record.state != "loaded":
            state = "not found" if record is None else record.state
            raise ModelNotLoadedError(
                f"Model autoload is disabled and {model!r} is not currently loaded in LM Studio "
                f"(state={state}). No inference request was sent."
            )
        return record

    def require_autoload_disabled(self, openai_model_ids: set[str]) -> None:
        """Fail unless /v1/models proves that LM Studio server-side JIT is disabled."""
        loaded_ids = set(self.loaded_ids())
        unloaded_ids = set(self.records) - loaded_ids
        if not unloaded_ids:
            raise ModelAutoloadPolicyError(
                "LM Studio JIT Model Loading state could not be verified because every discovered model is loaded. "
                "autoload_models=false requires JIT Model Loading to be disabled in LM Studio Server Settings."
            )

        exposed_unloaded_ids = sorted(openai_model_ids & unloaded_ids)
        if exposed_unloaded_ids:
            raise ModelAutoloadPolicyError(
                "autoload_models=false, but LM Studio JIT Model Loading is enabled: /v1/models exposes unloaded "
                f"model(s) {exposed_unloaded_ids}. Disable JIT Model Loading in LM Studio Server Settings."
            )

        if openai_model_ids != loaded_ids:
            raise ModelAutoloadPolicyError(
                "LM Studio JIT Model Loading state could not be verified because /v1/models did not exactly match "
                "the native loaded-model registry. No inference request was sent."
            )

    def resolve_first_loaded(self, models: Sequence[str], *, min_context_window: int | None) -> LoadedModel:
        """Return the first configured model that is loaded and meets the context floor.

        Matching is exact: a record that exists but is not loaded never matches, so a
        'not-loaded' base id is passed over in favour of a loaded ':N' instance. Each
        step of the walk is logged so the chosen model is never a silent decision.
        """
        logger.info("Resolving model from %d configured candidate(s), in order: %s", len(models), list(models))

        shortfalls: list[str] = []
        picked: LoadedModel | None = None
        for position, model in enumerate(models, start=1):
            prefix = f"[{position}/{len(models)}] checking {model}:"
            record = self.find(model)

            if record is None:
                logger.info("%s NOT FOUND on the server — skipping", prefix)
                continue

            if record.state != "loaded":
                logger.info("%s found as %r but state=%s — skipping", prefix, record.lm_studio_id, record.state)
                continue

            if record.loaded_context_length is None:
                raise ModelNotLoadedError(f"Loaded LM Studio model does not report loaded_context_length: {record.lm_studio_id}")

            max_context_length = record.max_context_length
            if max_context_length is None:
                raise ModelNotLoadedError(f"Loaded LM Studio model does not report max_context_length: {record.lm_studio_id}")

            if min_context_window is not None and record.loaded_context_length < min_context_window:
                logger.warning(
                    "%s LOADED but its context window is %s tokens, below the configured %s — skipping",
                    prefix,
                    f"{record.loaded_context_length:,}",
                    f"{min_context_window:,}",
                )
                shortfalls.append(
                    f"{record.lm_studio_id}: loaded with {record.loaded_context_length:,} tokens, "
                    f"configured context_window requires {min_context_window:,}"
                )
                continue

            logger.info("%s FOUND and LOADED (ctx=%s tokens)", prefix, f"{record.loaded_context_length:,}")
            picked = LoadedModel(
                configured_model=model,
                lm_studio_id=record.lm_studio_id,
                loaded_context_length=record.loaded_context_length,
                max_context_length=max_context_length,
            )
            break

        if picked is None:
            raise ModelNotLoadedError(_unresolved_message(list(models), self.loaded_ids(), shortfalls))

        skipped = list(models).index(picked.configured_model)
        logger.info(
            "Picking model %r (configured as %r), skipping %d earlier candidate(s)",
            picked.lm_studio_id,
            picked.configured_model,
            skipped,
        )
        return picked


def require_model_loaded(api_base: str | None, model: str) -> ModelRecord:
    """Prove server-side JIT is off and the exact model is resident before inference."""
    registry = LMStudioRegistry.fetch_quietly(api_base)
    record = registry.require_loaded(model)
    registry.require_autoload_disabled(fetch_model_ids(get_openai_models_url(api_base)))
    return record


def _unresolved_message(models: list[str], loaded_ids: list[str], shortfalls: list[str]) -> str:
    """Build the operator-facing error for a failed resolution."""
    lines = [
        "None of the configured models are loaded in LM Studio.",
        f"  configured (in order): {models}",
        f"  loaded on server:      {loaded_ids}",
    ]
    if shortfalls:
        lines.append("  rejected for context window:")
        lines.extend(f"    - {shortfall}" for shortfall in shortfalls)
    lines.append(f"  load one with: lms load {get_model_id_candidates(models[0])[-1]}")
    return "\n".join(lines)
