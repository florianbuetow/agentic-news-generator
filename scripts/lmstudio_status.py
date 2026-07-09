#!/usr/bin/env python3
"""Check if LM Studio is running and required models are loaded."""

import re
import sys
from dataclasses import dataclass
from typing import Any, cast

import yaml

from src.config import Config
from src.llm.lm_studio import LMStudioRegistry

CONFIG_PATH = Config.repo_config_path()

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


@dataclass(frozen=True)
class RequiredModels:
    """The ordered models a config section may use, and where they live."""

    section: str
    models: list[str]
    api_base: str


_Cfg = dict[str, Any]


def _sub(cfg: _Cfg, key: str) -> _Cfg:
    val = cfg.get(key, {})
    if not isinstance(val, dict):
        return {}
    return cast(_Cfg, val)


def collect_required_models(config: _Cfg) -> list[RequiredModels]:
    """Walk config and collect every LM Studio section's ordered models list."""
    models: list[RequiredModels] = []

    for top_key in ("summarize_transcripts", "url_clean_content"):
        llm = _sub(_sub(config, top_key), "llm")
        if llm.get("api_base"):
            models.append(
                RequiredModels(
                    section=f"{top_key}.llm",
                    models=[str(model) for model in llm["models"]],
                    api_base=str(llm["api_base"]),
                )
            )

    for top_key in ("agentic_unit_test_reviews", "agentic_shell_script_reviews"):
        llm = _sub(_sub(config, top_key), "llm")
        api_base = llm.get("base_url")
        if not api_base:
            api_base = llm.get("api_base")
        if api_base:
            models.append(
                RequiredModels(
                    section=f"{top_key}.llm",
                    models=[str(model) for model in llm["models"]],
                    api_base=str(api_base),
                )
            )

    return models


def _canonical_base(api_base: str) -> str:
    """Normalize URL so localhost and 127.0.0.1 at the same port group together."""
    return api_base.replace("localhost", "127.0.0.1")


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
    prev_section: str | None = None
    for section, model, instance, avail, state in rows:
        first = section != prev_section
        print(row(section if first else "", model, instance, _yn(avail), _state_cell(state)))
        prev_section = section
    print(bottom)


def _check_base(api_base: str, entries: list[RequiredModels]) -> tuple[bool, list[tuple[str, str, str, bool, str | None]]]:
    """Report each configured model's exact LM Studio state. Never raises.

    A section is available when at least one of its configured models is loaded —
    the same predicate ``LMStudioRegistry.resolve_first_loaded`` applies, so status
    and the pipeline can no longer disagree about the same server.
    """
    print(f"\n  LM Studio: {api_base}")

    try:
        registry = LMStudioRegistry.fetch(api_base)
    except (RuntimeError, ValueError) as exc:
        print(f"  {_RED}FAIL{_RESET}  Server not reachable at {api_base} ({exc})")
        return False, []

    print(f"  {_GREEN}OK{_RESET}    Server is running")

    rows: list[tuple[str, str, str, bool, str | None]] = []
    all_available = True
    for entry in sorted(entries, key=lambda e: e.section):
        section_has_loaded = False
        for model in entry.models:
            record = registry.find(model)
            if record is None:
                rows.append((entry.section, model, "-", False, None))
                continue
            is_loaded = record.state == "loaded"
            if is_loaded:
                section_has_loaded = True
            rows.append((entry.section, model, record.lm_studio_id, is_loaded, record.state))
        if not section_has_loaded:
            all_available = False

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

    bases: dict[str, list[RequiredModels]] = {}
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
