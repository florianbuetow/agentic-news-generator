"""LM Studio model discovery and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import requests


@dataclass(frozen=True)
class LMStudioModel:
    """A single model exposed by the LM Studio /api/v0/models endpoint."""

    model_id: str
    model_type: str
    state: str
    max_context_length: int
    loaded_context_length: int | None

    @property
    def is_loaded(self) -> bool:
        """True when LM Studio reports this model as loaded in memory."""
        return self.state == "loaded"

    @property
    def is_text_capable(self) -> bool:
        """True when the model can accept text chat completions (llm or vlm)."""
        return self.model_type in ("llm", "vlm")

    @property
    def effective_context_length(self) -> int:
        """Runtime ctx for loaded models; architectural max otherwise.

        LM Studio reports ``max_context_length`` as the model's architectural
        ceiling, but a loaded model runs with the (often smaller)
        ``loaded_context_length`` the user configured at load time. Requests
        exceeding the loaded value fail at runtime, so selection and
        per-file fit checks must use this.
        """
        if self.loaded_context_length is not None:
            return self.loaded_context_length
        return self.max_context_length


def fetch_lmstudio_models(*, models_endpoint: str, timeout: float) -> list[LMStudioModel]:
    """Call LM Studio /api/v0/models and return all reported models."""
    response = requests.get(models_endpoint, timeout=timeout)
    response.raise_for_status()
    payload = cast(dict[str, Any], response.json())

    data_raw = payload.get("data")
    if not isinstance(data_raw, list):
        raise ValueError(f"Unexpected /api/v0/models response shape: missing 'data' list. Got keys: {list(payload.keys())}")

    data = cast(list[dict[str, Any]], data_raw)
    models: list[LMStudioModel] = []
    for entry in data:
        try:
            loaded_raw = entry.get("loaded_context_length")
            loaded_ctx = int(loaded_raw) if loaded_raw is not None else None
            model = LMStudioModel(
                model_id=str(entry["id"]),
                model_type=str(entry["type"]),
                state=str(entry["state"]),
                max_context_length=int(entry["max_context_length"]),
                loaded_context_length=loaded_ctx,
            )
        except KeyError as e:
            raise ValueError(f"Model entry missing required field {e}: {entry!r}") from e

        models.append(model)

    return models


def select_best_model(
    *,
    models: list[LMStudioModel],
    required_tokens: int,
    prefer_loaded: bool,
) -> LMStudioModel:
    """Pick the best model for the task.

    Filters by text-capable type and sufficient context window, then ranks:
    loaded-first (if prefer_loaded), then by largest context window,
    then alphabetically by id for deterministic tie-break.
    """
    if required_tokens <= 0:
        raise ValueError(f"required_tokens must be > 0, got {required_tokens}")

    candidates = [m for m in models if m.is_text_capable and m.effective_context_length >= required_tokens]

    if not candidates:
        available = ", ".join(
            f"{m.model_id}({m.model_type},ctx={m.effective_context_length},max={m.max_context_length})" for m in models
        )
        raise ValueError(f"No LM Studio model can hold required_tokens={required_tokens}. Available (need type in llm/vlm): {available}")

    def rank_key(m: LMStudioModel) -> tuple[int, int, str]:
        loaded_bonus = 0 if (prefer_loaded and m.is_loaded) else 1
        return (loaded_bonus, -m.effective_context_length, m.model_id)

    candidates.sort(key=rank_key)
    return candidates[0]
