"""Bundle loader for article generation input bundles.

Reads manifest.json from a bundle directory, validates required fields,
and loads referenced files (transcript, topics) from the same directory.
"""

import json
import logging
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ManifestReference(BaseModel):
    """A single source reference entry in the manifest."""

    type: str
    title: str
    url: str
    author: str
    channel: str
    date: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class Manifest(BaseModel):
    """Parsed manifest.json from an article input bundle."""

    article_title: str = Field(..., min_length=1)
    slug: str = Field(..., min_length=1)
    publish_date: str = Field(..., min_length=1)
    source_text_file: str = Field(..., min_length=1)
    topics_file: str = Field(..., min_length=1)
    references: list[ManifestReference]

    model_config = ConfigDict(frozen=True, extra="forbid")


class LoadedBundle(BaseModel):
    """Fully loaded bundle with all file contents resolved."""

    manifest: Manifest
    source_text: str = Field(..., min_length=1)
    topics: list[dict[str, Any]]
    bundle_dir: str

    model_config = ConfigDict(frozen=True, extra="forbid")


def load_manifest(bundle_dir: Path) -> Manifest:
    """Load and validate manifest.json from a bundle directory.

    Args:
        bundle_dir: Path to the bundle directory containing manifest.json.

    Raises:
        FileNotFoundError: If manifest.json does not exist.
        ValueError: If manifest.json is invalid or missing required fields.
    """
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in bundle: {bundle_dir}")

    with open(manifest_path, encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"manifest.json root must be a JSON object: {manifest_path}")

    return Manifest.model_validate(raw)


def load_bundle(bundle_dir: Path) -> LoadedBundle:
    """Load a complete article input bundle.

    Reads manifest.json, then loads the referenced transcript and topics files.

    Args:
        bundle_dir: Path to the bundle directory.

    Raises:
        FileNotFoundError: If any required file is missing.
        ValueError: If manifest or file contents are invalid.
    """
    manifest = load_manifest(bundle_dir)

    source_text_path = bundle_dir / manifest.source_text_file
    if not source_text_path.exists():
        raise FileNotFoundError(f"Source text file '{manifest.source_text_file}' not found in bundle: {bundle_dir}")

    source_text = source_text_path.read_text(encoding="utf-8")
    if not source_text.strip():
        raise ValueError(f"Source text file '{manifest.source_text_file}' is empty in bundle: {bundle_dir}")

    topics_path = bundle_dir / manifest.topics_file
    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file '{manifest.topics_file}' not found in bundle: {bundle_dir}")

    with open(topics_path, encoding="utf-8") as handle:
        topics_raw = json.load(handle)

    if not isinstance(topics_raw, list):
        raise ValueError(f"Topics file must contain a JSON array: {topics_path}")

    topics = cast(list[dict[str, Any]], topics_raw)

    logger.info(
        "Loaded bundle: slug=%s title=%r source_chars=%d topics=%d",
        manifest.slug,
        manifest.article_title,
        len(source_text),
        len(topics),
    )

    return LoadedBundle(
        manifest=manifest,
        source_text=source_text,
        topics=topics,
        bundle_dir=str(bundle_dir),
    )


def bundle_to_source_metadata(bundle: LoadedBundle) -> dict[str, str | None]:
    """Convert a loaded bundle into the source_metadata dict expected by the orchestrator."""
    references_json = json.dumps(
        [ref.model_dump() for ref in bundle.manifest.references],
        ensure_ascii=False,
    )
    return {
        "source_file": bundle.manifest.source_text_file,
        "channel_name": _channel_from_references(bundle.manifest.references),
        "video_id": bundle.manifest.slug,
        "article_title": bundle.manifest.article_title,
        "slug": bundle.manifest.slug,
        "publish_date": bundle.manifest.publish_date,
        "references": references_json,
    }


def _channel_from_references(references: list[ManifestReference]) -> str:
    """Extract channel name from references list."""
    for ref in references:
        if ref.channel:
            return ref.channel
    raise ValueError("No channel name found in manifest references")
