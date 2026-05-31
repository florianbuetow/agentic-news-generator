"""Unit tests for URL ingestion metadata."""

from pathlib import Path

from src.url_ingestion.metadata import Metadata, MetadataHelper


def test_metadata_helper_saves_loads_and_exposes_typed_fields(tmp_path: Path) -> None:
    """Persist metadata JSON and load it back through MetadataHelper."""
    raw_path = tmp_path / "raw.pdf"
    metadata_path = tmp_path / "raw.metadata.json"
    metadata = Metadata(
        source_url="https://example.com/raw.pdf",
        normalized_url="https://example.com/raw.pdf",
        final_url="https://cdn.example.com/raw.pdf",
        sanitized_url_stem="https_example_com_raw_pdf",
        classified_type="pdf",
        downloaded_at="2026-05-31T00:00:00+00:00",
        http_status=200,
        raw_path=str(raw_path),
        metadata_path=str(metadata_path),
        status="downloaded",
        source_kind="url_download",
    )

    MetadataHelper(metadata).save(metadata_path)
    loaded = MetadataHelper.load(metadata_path)

    assert loaded.source_url == "https://example.com/raw.pdf"
    assert loaded.normalized_url == "https://example.com/raw.pdf"
    assert loaded.raw_path == raw_path
    assert loaded.metadata_path == metadata_path
    assert loaded.metadata == metadata
