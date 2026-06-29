#!/usr/bin/env python3
"""Check if LM Studio is running and required models are loaded."""

import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, cast

import yaml

from src.config import Config

CONFIG_PATH = Config.repo_config_path()

_TIMEOUT_SECONDS = 5

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
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

    for top_key in ("summarize_transcripts", "url_clean_content"):
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
    """Per-instance state from /api/v0/models, keyed by full model id.

    The keys are the verbatim LM Studio ids, so duplicate instances of the same
    model (which LM Studio suffixes with ':N', e.g. 'qwen/qwen3.6-35b-a3b:2')
    are kept distinct rather than collapsed together.
    """

    states: dict[str, str]


def get_model_index(api_base: str) -> ModelIndex | None:
    """GET /api/v0/models — map each model id (incl. ':N' duplicate instances) to its state. None if unreachable."""
    base = _strip_v1(api_base)
    data = _fetch_json(f"{base}/api/v0/models")
    if data is None:
        return None
    items = data.get("data", [])
    states = {m["id"]: str(m.get("state", "unknown")) for m in items}
    return ModelIndex(states=states)


def _model_matches(model: str, model_set: set[str]) -> bool:
    if model in model_set:
        return True
    short = model.split("/")[-1] if "/" in model else model
    return any(short in mid for mid in model_set)


def _matching_instances(model: str, states: dict[str, str]) -> list[str]:
    """Every LM Studio instance id that corresponds to a configured model id.

    Reuses the prefix-tolerant matching of ``_model_matches`` (an exact id, or the
    model's short name as a substring — which is how the configured ``openai/``
    prefix is matched against LM Studio's prefix-free id), but returns each
    matching instance separately so duplicate ':N' instances are surfaced rather
    than collapsed into a single yes/no.
    """
    return sorted(mid for mid in states if _model_matches(model, {mid}))


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


def _state_cell(state: str | None) -> str:
    """Colourise an LM Studio per-instance state. None means no matching instance."""
    if state is None:
        return f"{_RED}-{_RESET}"
    if state == "loaded":
        return f"{_GREEN}loaded{_RESET}"
    if state == "loading":
        return f"{_YELLOW}loading{_RESET}"
    return f"{_RED}{state}{_RESET}"


def _print_table(rows: list[tuple[str, str, str, bool, str | None]]) -> None:
    """Print ASCII table. rows = (section, config model, instance id, available, state)."""
    c0 = max(max((len(s) for s, _, _, _, _ in rows), default=10), 10)  # "Config Key"
    c1 = max(max((len(m) for _, m, _, _, _ in rows), default=5), 5)  # "Model"
    c2 = max(max((len(i) for _, _, i, _, _ in rows), default=8), 8)  # "Instance"
    c3 = 9  # "Available"
    c4 = max(max((len(st) for _, _, _, _, st in rows if st), default=5), 5)  # "State"

    top = f"┌{'─' * (c0 + 2)}┬{'─' * (c1 + 2)}┬{'─' * (c2 + 2)}┬{'─' * (c3 + 2)}┬{'─' * (c4 + 2)}┐"
    sep = f"├{'─' * (c0 + 2)}┼{'─' * (c1 + 2)}┼{'─' * (c2 + 2)}┼{'─' * (c3 + 2)}┼{'─' * (c4 + 2)}┤"
    bottom = f"└{'─' * (c0 + 2)}┴{'─' * (c1 + 2)}┴{'─' * (c2 + 2)}┴{'─' * (c3 + 2)}┴{'─' * (c4 + 2)}┘"

    def row(sec: str, m: str, inst: str, a: str, st: str) -> str:
        return f"│ {_cell(sec, c0)} │ {_cell(m, c1)} │ {_cell(inst, c2)} │ {_cell(a, c3, '^')} │ {_cell(st, c4, '^')} │"

    print(top)
    print(
        row(
            f"{_BOLD}Config Key{_RESET}",
            f"{_BOLD}Model{_RESET}",
            f"{_BOLD}Instance{_RESET}",
            f"{_BOLD}Available{_RESET}",
            f"{_BOLD}State{_RESET}",
        )
    )
    print(sep)
    prev_key: tuple[str, str] | None = None
    for section, model, instance, avail, state in rows:
        key = (section, model)
        first = key != prev_key
        print(row(section if first else "", model if first else "", instance, _yn(avail), _state_cell(state)))
        prev_key = key
    print(bottom)


def _check_base(api_base: str, entries: list[RequiredModel]) -> tuple[bool, list[tuple[str, str, str, bool, str | None]]]:
    print(f"\n  LM Studio: {api_base}")

    index = get_model_index(api_base)
    if index is None:
        print(f"  {_RED}FAIL{_RESET}  Server not reachable at {api_base}")
        return False, []

    print(f"  {_GREEN}OK{_RESET}    Server is running")

    rows: list[tuple[str, str, str, bool, str | None]] = []
    all_available = True
    for entry in sorted(entries, key=lambda e: e.section):
        instances = _matching_instances(entry.model, index.states)
        if not instances:
            all_available = False
            rows.append((entry.section, entry.model, "-", False, None))
            continue
        rows.extend((entry.section, entry.model, instance, True, index.states[instance]) for instance in instances)

    return all_available, rows


def main() -> int:
    """Check LM Studio status and model availability."""
    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        return 0

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
    all_rows: list[tuple[str, str, str, bool, str | None]] = []
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
        return 0

    print(f"  {_GREEN}All checks passed.{_RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
