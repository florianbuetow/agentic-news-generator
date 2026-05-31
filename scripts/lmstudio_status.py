#!/usr/bin/env python3
"""Check if LM Studio is running and required models are loaded."""

import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

_TIMEOUT_SECONDS = 5

_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


@dataclass(frozen=True)
class RequiredModel:
    """A model required by a config section."""

    section: str
    model: str
    api_base: str


_Cfg = dict[str, Any]


def _sub(cfg: _Cfg, key: str) -> _Cfg:
    val = cfg.get(key, {})
    if not isinstance(val, dict):
        return {}
    return cast(_Cfg, val)


def collect_required_models(config: _Cfg) -> list[RequiredModel]:
    """Walk config and collect all (section, model, api_base) tuples for LM Studio endpoints."""
    models: list[RequiredModel] = []

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

    for top_key in ("summarize_transcripts", "topic_boundaries", "url_clean_content"):
        llm = _sub(_sub(config, top_key), "llm")
        if llm.get("api_base"):
            models.append(
                RequiredModel(
                    section=f"{top_key}.llm",
                    model=str(llm["model"]),
                    api_base=str(llm["api_base"]),
                )
            )

    for top_key in ("agentic_unit_test_reviews", "agentic_shell_script_reviews"):
        llm = _sub(_sub(config, top_key), "llm")
        api_base = llm.get("base_url") or llm.get("api_base")
        if api_base:
            models.append(
                RequiredModel(
                    section=f"{top_key}.llm",
                    model=str(llm["model"]),
                    api_base=str(api_base),
                )
            )

    return models


def _fetch_json(url: str) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(url)  # noqa: S310
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:  # noqa: S310
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _strip_v1(api_base: str) -> str:
    base = api_base.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _canonical_base(api_base: str) -> str:
    """Normalize URL so localhost and 127.0.0.1 at the same port group together."""
    return api_base.replace("localhost", "127.0.0.1")


@dataclass(frozen=True)
class ModelIndex:
    """Available and loaded model sets from /api/v0/models."""

    available: set[str]
    loaded: set[str]


def get_model_index(api_base: str) -> ModelIndex | None:
    """GET /api/v0/models — available = all downloaded, loaded = state=='loaded'. None if unreachable."""
    base = _strip_v1(api_base)
    data = _fetch_json(f"{base}/api/v0/models")
    if data is None:
        return None
    items = data.get("data", [])
    available = {m["id"] for m in items}
    loaded = {m["id"] for m in items if m.get("state") == "loaded"}
    return ModelIndex(available=available, loaded=loaded)


def _model_matches(model: str, model_set: set[str]) -> bool:
    if model in model_set:
        return True
    short = model.split("/")[-1] if "/" in model else model
    return any(short in mid for mid in model_set)


def _vlen(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _cell(s: str, width: int, align: str = "<") -> str:
    pad = width - _vlen(s)
    if align == ">":
        return " " * pad + s
    if align == "^":
        left = pad // 2
        right = pad - left
        return " " * left + s + " " * right
    return s + " " * pad


def _yn(flag: bool) -> str:
    return f"{_GREEN}yes{_RESET}" if flag else f"{_RED}-{_RESET}"


def _print_table(rows: list[tuple[str, str, bool, bool]]) -> None:
    """Print ASCII table. rows = (section, model, available, loaded)."""
    c0 = max((len(s) for s, _, _, _ in rows), default=10)
    c0 = max(c0, 10)  # "Config Key"
    c1 = max((len(m) for _, m, _, _ in rows), default=5)
    c1 = max(c1, 5)  # "Model"
    c2 = 9  # "Available"
    c3 = 6  # "Loaded"

    top = f"┌{'─' * (c0 + 2)}┬{'─' * (c1 + 2)}┬{'─' * (c2 + 2)}┬{'─' * (c3 + 2)}┐"
    sep = f"├{'─' * (c0 + 2)}┼{'─' * (c1 + 2)}┼{'─' * (c2 + 2)}┼{'─' * (c3 + 2)}┤"
    bottom = f"└{'─' * (c0 + 2)}┴{'─' * (c1 + 2)}┴{'─' * (c2 + 2)}┴{'─' * (c3 + 2)}┘"

    def row(sec: str, m: str, a: str, ld: str) -> str:
        return f"│ {_cell(sec, c0)} │ {_cell(m, c1)} │ {_cell(a, c2, '^')} │ {_cell(ld, c3, '^')} │"

    print(top)
    print(row(f"{_BOLD}Config Key{_RESET}", f"{_BOLD}Model{_RESET}", f"{_BOLD}Available{_RESET}", f"{_BOLD}Loaded{_RESET}"))
    print(sep)
    for section, model, avail, loaded in rows:
        print(row(section, model, _yn(avail), _yn(loaded)))
    print(bottom)


def _check_base(api_base: str, entries: list[RequiredModel]) -> tuple[bool, list[tuple[str, str, bool, bool]]]:
    print(f"\n  LM Studio: {api_base}")

    index = get_model_index(api_base)
    if index is None:
        print(f"  {_RED}FAIL{_RESET}  Server not reachable at {api_base}")
        return False, []

    print(f"  {_GREEN}OK{_RESET}    Server is running")

    rows: list[tuple[str, str, bool, bool]] = []
    for entry in sorted(entries, key=lambda e: e.section):
        avail = _model_matches(entry.model, index.available)
        loaded = _model_matches(entry.model, index.loaded)
        rows.append((entry.section, entry.model, avail, loaded))

    return all(avail for _, _, avail, _ in rows), rows


def main() -> int:
    """Check LM Studio status and model availability."""
    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        return 1

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config: _Cfg = yaml.safe_load(f)

    required = collect_required_models(config)
    if not required:
        print("No LM Studio endpoints found in config.")
        return 0

    bases: dict[str, list[RequiredModel]] = {}
    for rm in required:
        bases.setdefault(_canonical_base(rm.api_base), []).append(rm)

    all_ok = True
    all_rows: list[tuple[str, str, bool, bool]] = []
    for api_base, entries in sorted(bases.items()):
        ok, rows = _check_base(api_base, entries)
        if not ok:
            all_ok = False
        all_rows.extend(rows)

    if all_rows:
        print()
        _print_table(all_rows)

    print()
    if not all_ok:
        print("  Some checks failed. Load the required models in LM Studio.")
        return 1

    print(f"  {_GREEN}All checks passed.{_RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
