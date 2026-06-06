#!/usr/bin/env python3
"""Fetch all raindrop.io bookmarks labelled with their folder path.

Emits one line per bookmark in the form ``Category->Subcategory:url`` (deeper
nesting extends the chain, e.g. ``A->B->C:url``; unsorted bookmarks become
``Unsorted:url``). The Raindrop.io API token is read through the ``Config``
object from ``config/config.yaml`` (``integrations.raindrop_io.token``). Output
is written to the configured URL inbox as ``raindrop-<date>.txt``.

By default only bookmarks whose link was not emitted in a previous run are
written; pass ``--force`` to re-emit every bookmark. Emitted links are tracked
in ``raindrop-fetched-urls.json`` under the URL ingestion base directory.
"""

import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import cast

import requests

from src.config import Config

RAINDROP_API_BASE = "https://api.raindrop.io/rest/v1"
ALL_RAINDROPS_COLLECTION_ID = 0
PER_PAGE = 50
MAX_PAGES = 50
MAX_PATH_DEPTH = 20
REQUEST_TIMEOUT_SECONDS = 30
UNSORTED_COLLECTION_ID = -1
TRASH_COLLECTION_ID = -99
FETCHED_URLS_FILENAME = "raindrop-fetched-urls.json"


def fetch_collection_map(token: str) -> dict[int, tuple[str, int | None]]:
    """Return a ``collection_id -> (title, parent_id)`` map for all collections."""
    headers = {"Authorization": f"Bearer {token}"}
    collection_map: dict[int, tuple[str, int | None]] = {}

    for endpoint in (f"{RAINDROP_API_BASE}/collections", f"{RAINDROP_API_BASE}/collections/childrens"):
        print(f"Fetching collections endpoint: {endpoint}", flush=True)
        response = requests.get(endpoint, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        items = response.json().get("items", [])
        print(f"  found {len(items)} collection(s)", flush=True)
        for item in items:
            collection_id = item.get("_id")
            if collection_id is None:
                continue
            parent = item.get("parent")
            parent_id = parent.get("$id") if parent else None
            collection_map[collection_id] = (item.get("title", ""), parent_id)

    return collection_map


def fetch_all_raindrops(token: str) -> tuple[list[tuple[int, str]], list[str]]:
    """Page through every raindrop and return ``(collection_id, link)`` pairs plus failures."""
    headers = {"Authorization": f"Bearer {token}"}
    endpoint = f"{RAINDROP_API_BASE}/raindrops/{ALL_RAINDROPS_COLLECTION_ID}"

    raindrops: list[tuple[int, str]] = []
    failures: list[str] = []
    expected_count: int | None = None

    for page in range(MAX_PAGES):
        try:
            print(f"Fetching raindrops page {page + 1}...", flush=True)
            response = requests.get(
                endpoint,
                headers=headers,
                params={"page": page, "perpage": PER_PAGE},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            failures.append(f"page {page}: {exc}")
            continue

        if expected_count is None:
            expected_count = payload.get("count")
            if expected_count is not None:
                print(f"Raindrop reported {expected_count} bookmark(s)", flush=True)

        items = payload.get("items", [])
        print(f"  page {page + 1}: {len(items)} bookmark(s)", flush=True)
        if not items:
            break

        for item in items:
            link = item.get("link")
            collection_id = item.get("collectionId")
            if link and collection_id is not None:
                raindrops.append((collection_id, link))

        if expected_count is not None and len(raindrops) >= expected_count:
            break
    else:
        failures.append(f"reached MAX_PAGES={MAX_PAGES} backstop before exhausting raindrops")

    if expected_count is not None and len(raindrops) < expected_count:
        failures.append(f"fetched {len(raindrops)} raindrops but raindrop reported count={expected_count}")

    return raindrops, failures


def resolve_folder_path(collection_id: int, collection_map: dict[int, tuple[str, int | None]]) -> str:
    """Resolve a collection id to a ``->`` joined folder path from root to leaf."""
    if collection_id == UNSORTED_COLLECTION_ID:
        return "Unsorted"
    if collection_id == TRASH_COLLECTION_ID:
        return "Trash"

    titles: list[str] = []
    seen: set[int] = set()
    current: int | None = collection_id
    while current is not None and current in collection_map and current not in seen and len(titles) < MAX_PATH_DEPTH:
        title, parent_id = collection_map[current]
        titles.append(title)
        seen.add(current)
        current = parent_id

    if not titles:
        return f"Unknown({collection_id})"
    return "->".join(reversed(titles))


def resolve_output_path(output_dir: Path, today: date) -> Path:
    """Return a non-clobbering dated output path in the output directory."""
    base = output_dir / f"raindrop-{today.isoformat()}.txt"
    if not base.exists():
        return base
    chosen: Path | None = None
    for suffix in range(1, 1000):
        candidate = output_dir / f"raindrop-{today.isoformat()}-{suffix}.txt"
        if not candidate.exists():
            chosen = candidate
            break
    if chosen is None:
        raise FileExistsError(f"Could not find a free output filename in {output_dir}")
    return chosen


def load_fetched_urls(state_path: Path) -> set[str]:
    """Return the set of raindrop links already emitted to the inbox, or an empty set when no state exists."""
    if not state_path.is_file():
        return set()
    stored = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(stored, list):
        raise ValueError(f"expected a JSON array of links, got {type(stored).__name__}")
    return {str(link) for link in cast("list[object]", stored)}


def save_fetched_urls(state_path: Path, links: set[str]) -> None:
    """Atomically persist the emitted raindrop links as a sorted JSON array."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_name(f"{state_path.name}.tmp")
    temp_path.write_text(json.dumps(sorted(links), indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, state_path)


def select_unfetched_raindrops(
    raindrops: list[tuple[int, str]],
    previously_fetched: set[str],
    force: bool,
) -> list[tuple[int, str]]:
    """Return raindrops to emit: all when forcing, otherwise those whose link was not previously fetched."""
    if force:
        return list(raindrops)
    return [(collection_id, link) for collection_id, link in raindrops if link not in previously_fetched]


def emit_new_raindrops(
    *,
    raindrops: list[tuple[int, str]],
    collection_map: dict[int, tuple[str, int | None]],
    inbox_dir: Path,
    state_path: Path,
    previously_fetched: set[str],
    force: bool,
) -> int:
    """Write unfetched raindrops to a new inbox file, persist the seen-set, and print a summary."""
    selected = select_unfetched_raindrops(raindrops, previously_fetched, force)
    skipped_already_fetched = len(raindrops) - len(selected)

    if not selected:
        print("\nSummary:")
        print(f"raindrops_fetched: {len(raindrops)}")
        print(f"already_fetched_skipped: {skipped_already_fetched}")
        print("No new bookmarks to write into the inbox.")
        return 0

    lines = sorted(f"{resolve_folder_path(collection_id, collection_map)}:{link}" for collection_id, link in selected)

    output_path = resolve_output_path(inbox_dir, date.today())
    print(f"Writing inbox file: {output_path}", flush=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Persist the seen-set only after the inbox file is written so a crash never
    # records links as fetched that never reached the inbox.
    updated_fetched = previously_fetched | {link for _, link in selected}
    save_fetched_urls(state_path, updated_fetched)

    print("\nSummary:")
    if force:
        print("force: re-emitted every bookmark")
    print(f"raindrops_fetched: {len(raindrops)}")
    print(f"collections_mapped: {len(collection_map)}")
    print(f"already_fetched_skipped: {skipped_already_fetched}")
    print(f"lines_written: {len(lines)}")
    print(f"fetched_urls_tracked: {len(updated_fetched)}")
    print(f"output_file: {output_path}")
    return 0


def main() -> int:
    """Fetch raindrops, label each with its folder path, and write to the configured URL inbox."""
    arguments = sys.argv[1:]
    if arguments not in ([], ["--force"]):
        print("Usage: fetchurls-raindrop.py [--force]", file=sys.stderr)
        return 1
    force = arguments == ["--force"]

    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / "config.yaml"

    try:
        print(f"Loading config: {config_path}", flush=True)
        config = Config(config_path)
        token = config.get_raindrop_token()
        inbox_dir = config.get_url_inbox_dir()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Fail fast: confirm the configured inbox is present before making any API calls.
    if not inbox_dir.is_dir():
        print(f"Error: configured URL inbox folder does not exist: {inbox_dir}", file=sys.stderr)
        return 1
    print(f"URL inbox directory: {inbox_dir}", flush=True)

    try:
        print("Fetching Raindrop collection map...", flush=True)
        collection_map = fetch_collection_map(token)
    except requests.RequestException as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Mapped {len(collection_map)} collection(s)", flush=True)

    print("Fetching Raindrop bookmarks...", flush=True)
    raindrops, failures = fetch_all_raindrops(token)

    # Do not publish a partial file into the ingestion inbox when any page failed.
    if failures:
        print(f"raindrops_fetched: {len(raindrops)}")
        print("\n--- Failure Summary (output not written) ---")
        for failure in failures:
            print(f"❌ {failure}")
        return 1

    state_path = config.get_url_base_dir() / FETCHED_URLS_FILENAME
    try:
        previously_fetched = load_fetched_urls(state_path)
    except (OSError, ValueError) as exc:
        print(f"Error: could not read fetched-URL state {state_path}: {exc}", file=sys.stderr)
        return 1

    return emit_new_raindrops(
        raindrops=raindrops,
        collection_map=collection_map,
        inbox_dir=inbox_dir,
        state_path=state_path,
        previously_fetched=previously_fetched,
        force=force,
    )


if __name__ == "__main__":
    sys.exit(main())
