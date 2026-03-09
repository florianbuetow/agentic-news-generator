#!/usr/bin/env python3
"""Check if LM Studio is running and required models are loaded."""

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class RequiredModel:
    """A model required by a config section."""

    section: str
    model: str
    api_base: str


_Cfg = dict[str, Any]


def _sub(cfg: _Cfg, key: str) -> _Cfg:
    """Return a nested dict or empty dict."""
    val = cfg.get(key, {})
    if not isinstance(val, dict):
        return {}
    return cast(_Cfg, val)


def collect_required_models(config: _Cfg) -> list[RequiredModel]:
    """Walk config and collect all (section, model, api_base) tuples for LM Studio endpoints."""
    models: list[RequiredModel] = []

    # topic_segmentation.agent_llm / critic_llm
    ts = _sub(config, "topic_segmentation")
    for key in ("agent_llm", "critic_llm"):
        llm = _sub(ts, key)
        if llm.get("api_base"):
            models.append(
                RequiredModel(
                    section=f"topic_segmentation.{key}",
                    model=str(llm["model"]),
                    api_base=str(llm["api_base"]),
                )
            )

    # topic_detection.embedding (uses model_name instead of model)
    td = _sub(config, "topic_detection")
    emb = _sub(td, "embedding")
    if emb.get("api_base"):
        models.append(
            RequiredModel(
                section="topic_detection.embedding",
                model=str(emb["model_name"]),
                api_base=str(emb["api_base"]),
            )
        )

    # topic_detection.topic_detection_llm
    td_llm = _sub(td, "topic_detection_llm")
    if td_llm.get("api_base"):
        models.append(
            RequiredModel(
                section="topic_detection.topic_detection_llm",
                model=str(td_llm["model"]),
                api_base=str(td_llm["api_base"]),
            )
        )

    # topic_detection.llm_label.llm
    llm_label = _sub(td, "llm_label")
    llm_label_llm = _sub(llm_label, "llm")
    if llm_label_llm.get("api_base"):
        models.append(
            RequiredModel(
                section="topic_detection.llm_label.llm",
                model=str(llm_label_llm["model"]),
                api_base=str(llm_label_llm["api_base"]),
            )
        )

    return models


def _fetch_models_json(api_base: str) -> dict[str, Any] | None:
    """GET /v1/models and return parsed JSON, or None on failure."""
    try:
        req = urllib.request.Request(f"{api_base}/models")  # noqa: S310
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:  # noqa: S310
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def get_loaded_models(api_base: str) -> set[str] | None:
    """Fetch loaded model IDs. Returns None if server unreachable, empty set if no models."""
    data = _fetch_models_json(api_base)
    if data is None:
        return None
    return {m["id"] for m in data.get("data", [])}


def _is_model_loaded(model: str, loaded: set[str]) -> bool:
    """Check if a model is loaded, trying exact match then short-name match."""
    if model in loaded:
        return True
    model_short = model.split("/")[-1] if "/" in model else model
    return any(model_short in loaded_id for loaded_id in loaded)


def _check_base(api_base: str, entries: list[RequiredModel]) -> bool:
    """Check one api_base: server reachability and model availability. Returns True if all OK."""
    print(f"\n  LM Studio: {api_base}")

    loaded = get_loaded_models(api_base)
    if loaded is None:
        print(f"    FAIL  Server not reachable at {api_base}")
        for entry in entries:
            print(f"          - {entry.section} needs model: {entry.model}")
        return False

    print("    OK    Server is running")

    # Collect unique required model names for this base
    required_models: dict[str, list[str]] = {}
    for entry in entries:
        required_models.setdefault(entry.model, []).append(entry.section)

    all_ok = True
    for model, sections in sorted(required_models.items()):
        if _is_model_loaded(model, loaded):
            print(f"    OK    Model loaded: {model}")
        else:
            print(f"    FAIL  Model not loaded: {model}")
            for section in sections:
                print(f"          - required by: {section}")
            if loaded:
                print(f"          - loaded models: {', '.join(sorted(loaded))}")
            else:
                print("          - no models currently loaded")
            all_ok = False

    return all_ok


def main() -> int:
    """Check LM Studio status and loaded models."""
    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        return 1

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config: _Cfg = yaml.safe_load(f)

    required = collect_required_models(config)
    if not required:
        print("No LM Studio endpoints found in config.")
        return 0

    # Group by api_base
    bases: dict[str, list[RequiredModel]] = {}
    for rm in required:
        bases.setdefault(rm.api_base, []).append(rm)

    all_ok = all(_check_base(api_base, entries) for api_base, entries in sorted(bases.items()))

    print()
    if not all_ok:
        print("  Some checks failed. Start LM Studio and load the required models.")
        return 1

    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
