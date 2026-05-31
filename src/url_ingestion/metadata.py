"""Metadata model and persistence helper for URL ingestion."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Metadata(BaseModel):
    """Metadata saved beside a downloaded URL raw source file."""

    source_url: str | None = Field(..., description="Original URL read from the inbox file")
    normalized_url: str | None = Field(..., description="URL after normalization")
    final_url: str | None = Field(..., description="Final URL after redirects when available")
    sanitized_url_stem: str = Field(..., description="Filesystem-safe stem derived from the normalized URL")
    classified_type: str = Field(..., description="URL class used for download routing")
    downloaded_at: str = Field(..., description="Download timestamp")
    http_status: int | None = Field(..., description="HTTP status or closest available navigation status")
    raw_path: str = Field(..., description="Path to the downloaded raw source file")
    metadata_path: str = Field(..., description="Path to this metadata JSON file")
    status: str = Field(..., description="Download status")
    source_kind: Literal["url_download", "manual_drop"] = Field(..., description="Source acquisition mode")

    model_config = ConfigDict(frozen=True, extra="forbid")


class MetadataHelper:
    """Own metadata field access plus load and save behavior."""

    def __init__(self, metadata: Metadata) -> None:
        """Initialize the helper with validated metadata."""
        self._metadata = metadata

    @property
    def metadata(self) -> Metadata:
        """Return the wrapped metadata model."""
        return self._metadata

    @property
    def source_url(self) -> str | None:
        """Return the original source URL."""
        return self._metadata.source_url

    @property
    def normalized_url(self) -> str | None:
        """Return the normalized URL."""
        return self._metadata.normalized_url

    @property
    def raw_path(self) -> Path:
        """Return the raw source path."""
        return Path(self._metadata.raw_path)

    @property
    def metadata_path(self) -> Path:
        """Return the metadata path."""
        return Path(self._metadata.metadata_path)

    @classmethod
    def load(cls, path: Path) -> "MetadataHelper":
        """Load metadata JSON from disk."""
        metadata = Metadata.model_validate_json(path.read_text(encoding="utf-8"))
        return cls(metadata)

    def save(self, path: Path) -> None:
        """Save metadata JSON to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
