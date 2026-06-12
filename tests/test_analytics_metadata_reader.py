"""Unit tests for the yt-dlp .info.json metadata reader."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.analytics.errors import MetadataError
from src.analytics.metadata_reader import load_video_metadata


def write_info_json(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """Write an info.json payload into tmp_path and return its path."""
    info_path = tmp_path / "Video [abc123XYZ09].info.json"
    info_path.write_text(json.dumps(payload), encoding="utf-8")
    return info_path


class TestLoadVideoMetadata:
    """Reading and normalizing required metadata fields."""

    def test_full_payload_loads(self, tmp_path: Path) -> None:
        """title, upload_date, and description are extracted."""
        info_path = write_info_json(
            tmp_path,
            {"title": "Building RAG Systems", "upload_date": "20251109", "description": "A talk."},
        )
        metadata = load_video_metadata(info_path)
        assert metadata.title == "Building RAG Systems"
        assert metadata.upload_date == "2025-11-09"
        assert metadata.description == "A talk."

    def test_compact_upload_date_normalized_to_iso(self, tmp_path: Path) -> None:
        """yt-dlp's compact YYYYMMDD format becomes ISO YYYY-MM-DD."""
        info_path = write_info_json(tmp_path, {"title": "T", "upload_date": "20240131"})
        assert load_video_metadata(info_path).upload_date == "2024-01-31"

    def test_iso_upload_date_rejected(self, tmp_path: Path) -> None:
        """Only yt-dlp's compact format is supported; ISO input is malformed."""
        info_path = write_info_json(tmp_path, {"title": "T", "upload_date": "2024-01-31"})
        with pytest.raises(MetadataError, match="upload_date"):
            load_video_metadata(info_path)

    def test_missing_upload_date_is_none(self, tmp_path: Path) -> None:
        """upload_date is optional at read time (timeline enforces later)."""
        info_path = write_info_json(tmp_path, {"title": "T"})
        assert load_video_metadata(info_path).upload_date is None

    def test_null_upload_date_is_none(self, tmp_path: Path) -> None:
        """An explicit null upload_date reads as None."""
        info_path = write_info_json(tmp_path, {"title": "T", "upload_date": None})
        assert load_video_metadata(info_path).upload_date is None

    def test_missing_description_is_none(self, tmp_path: Path) -> None:
        """description is optional."""
        info_path = write_info_json(tmp_path, {"title": "T", "upload_date": "20240101"})
        assert load_video_metadata(info_path).description is None


class TestVideoIdField:
    """The yt-dlp 'id' field (used for ID-less filename joins, Amendment 10)."""

    def test_id_field_loaded(self, tmp_path: Path) -> None:
        """A present 'id' field is exposed as video_id."""
        info_path = write_info_json(tmp_path, {"title": "T", "id": "abc123XYZ09"})
        assert load_video_metadata(info_path).video_id == "abc123XYZ09"

    def test_missing_id_is_none(self, tmp_path: Path) -> None:
        """Metadata without an 'id' field yields video_id None."""
        info_path = write_info_json(tmp_path, {"title": "T"})
        assert load_video_metadata(info_path).video_id is None

    def test_non_string_id_raises(self, tmp_path: Path) -> None:
        """'id' present but not a string is malformed metadata."""
        info_path = write_info_json(tmp_path, {"title": "T", "id": 42})
        with pytest.raises(MetadataError, match="'id'"):
            load_video_metadata(info_path)

    def test_empty_id_raises(self, tmp_path: Path) -> None:
        """'id' present but empty is malformed metadata."""
        info_path = write_info_json(tmp_path, {"title": "T", "id": ""})
        with pytest.raises(MetadataError, match="'id'"):
            load_video_metadata(info_path)


class TestLoadVideoMetadataFailures:
    """Fail-fast behavior on unreadable or incomplete metadata."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A nonexistent file is a MetadataError."""
        with pytest.raises(MetadataError, match="missing.info.json"):
            load_video_metadata(tmp_path / "missing.info.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        """Malformed JSON is a MetadataError including the path."""
        info_path = tmp_path / "broken.info.json"
        info_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(MetadataError, match="broken.info.json"):
            load_video_metadata(info_path)

    def test_non_object_json_raises(self, tmp_path: Path) -> None:
        """A top-level array is not a metadata object."""
        info_path = tmp_path / "array.info.json"
        info_path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(MetadataError, match="array.info.json"):
            load_video_metadata(info_path)

    def test_missing_title_raises(self, tmp_path: Path) -> None:
        """title is required for every report."""
        info_path = write_info_json(tmp_path, {"upload_date": "20240101"})
        with pytest.raises(MetadataError, match="title"):
            load_video_metadata(info_path)

    def test_empty_title_raises(self, tmp_path: Path) -> None:
        """An empty title is as unusable as a missing one."""
        info_path = write_info_json(tmp_path, {"title": "", "upload_date": "20240101"})
        with pytest.raises(MetadataError, match="title"):
            load_video_metadata(info_path)

    @pytest.mark.parametrize("bad_date", ["Nov 9 2025", "2025119", "20251332", "2025-13-32"])
    def test_malformed_upload_date_raises(self, tmp_path: Path, bad_date: str) -> None:
        """Dates that are not valid compact YYYYMMDD fail fast."""
        info_path = write_info_json(tmp_path, {"title": "T", "upload_date": bad_date})
        with pytest.raises(MetadataError, match="upload_date"):
            load_video_metadata(info_path)
