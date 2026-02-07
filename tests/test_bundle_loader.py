"""Tests for article generation bundle loader."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.article_generation.bundle_loader import (
    LoadedBundle,
    Manifest,
    ManifestReference,
    bundle_to_source_metadata,
    load_bundle,
    load_manifest,
)


def _write_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def _create_valid_manifest() -> dict[str, object]:
    return {
        "article_title": "Test Article Title",
        "slug": "TEST123",
        "publish_date": "2026-02-05",
        "source_text_file": "transcript.txt",
        "topics_file": "topics.json",
        "references": [
            {
                "type": "video",
                "title": "Test Article Title",
                "url": "https://www.youtube.com/watch?v=TEST123",
                "author": "Author Name",
                "channel": "TestChannel",
                "date": "2026-02-05",
            }
        ],
    }


def _create_valid_bundle(bundle_dir: Path) -> None:
    """Create a complete valid bundle in the given directory."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(bundle_dir / "manifest.json", _create_valid_manifest())
    (bundle_dir / "transcript.txt").write_text("This is the transcript content.", encoding="utf-8")
    _write_json(bundle_dir / "topics.json", [{"topic": "AI", "start": 0, "end": 100}])


class TestManifestModel:
    """Tests for the Manifest Pydantic model."""

    def test_valid_manifest_parses(self) -> None:
        manifest = Manifest.model_validate(_create_valid_manifest())
        assert manifest.article_title == "Test Article Title"
        assert manifest.slug == "TEST123"
        assert manifest.publish_date == "2026-02-05"
        assert manifest.source_text_file == "transcript.txt"
        assert manifest.topics_file == "topics.json"
        assert len(manifest.references) == 1
        assert manifest.references[0].channel == "TestChannel"

    def test_missing_article_title_fails(self) -> None:
        data = _create_valid_manifest()
        del data["article_title"]
        with pytest.raises(ValidationError):
            Manifest.model_validate(data)

    def test_missing_slug_fails(self) -> None:
        data = _create_valid_manifest()
        del data["slug"]
        with pytest.raises(ValidationError):
            Manifest.model_validate(data)

    def test_empty_article_title_fails(self) -> None:
        data = _create_valid_manifest()
        data["article_title"] = ""
        with pytest.raises(ValidationError):
            Manifest.model_validate(data)

    def test_empty_slug_fails(self) -> None:
        data = _create_valid_manifest()
        data["slug"] = ""
        with pytest.raises(ValidationError):
            Manifest.model_validate(data)

    def test_missing_source_text_file_fails(self) -> None:
        data = _create_valid_manifest()
        del data["source_text_file"]
        with pytest.raises(ValidationError):
            Manifest.model_validate(data)

    def test_missing_topics_file_fails(self) -> None:
        data = _create_valid_manifest()
        del data["topics_file"]
        with pytest.raises(ValidationError):
            Manifest.model_validate(data)

    def test_empty_references_list_is_valid(self) -> None:
        data = _create_valid_manifest()
        data["references"] = []
        manifest = Manifest.model_validate(data)
        assert len(manifest.references) == 0

    def test_extra_fields_forbidden(self) -> None:
        data = _create_valid_manifest()
        data["extra_field"] = "not allowed"
        with pytest.raises(ValidationError):
            Manifest.model_validate(data)


class TestManifestReferenceModel:
    """Tests for the ManifestReference Pydantic model."""

    def test_valid_reference(self) -> None:
        ref = ManifestReference(
            type="video",
            title="Test",
            url="https://example.com",
            author="Author",
            channel="Channel",
            date="2026-01-01",
        )
        assert ref.type == "video"
        assert ref.channel == "Channel"

    def test_missing_field_fails(self) -> None:
        with pytest.raises(ValidationError):
            ManifestReference(
                type="video",
                title="Test",
                # missing url, author, channel, date
            )  # type: ignore[call-arg]


class TestLoadManifest:
    """Tests for load_manifest function."""

    def test_load_valid_manifest(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "manifest.json", _create_valid_manifest())
        manifest = load_manifest(tmp_path)
        assert manifest.slug == "TEST123"

    def test_missing_manifest_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="manifest.json not found"):
            load_manifest(tmp_path)

    def test_invalid_json_root_type_raises(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "manifest.json", [1, 2, 3])
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_manifest(tmp_path)

    def test_invalid_manifest_fields_raises(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "manifest.json", {"slug": "x"})
        with pytest.raises(ValidationError):
            load_manifest(tmp_path)


class TestLoadBundle:
    """Tests for load_bundle function."""

    def test_load_valid_bundle(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "TEST123"
        _create_valid_bundle(bundle_dir)
        bundle = load_bundle(bundle_dir)
        assert bundle.manifest.slug == "TEST123"
        assert "transcript content" in bundle.source_text
        assert len(bundle.topics) == 1
        assert bundle.bundle_dir == str(bundle_dir)

    def test_missing_transcript_raises(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "TEST123"
        bundle_dir.mkdir(parents=True)
        _write_json(bundle_dir / "manifest.json", _create_valid_manifest())
        _write_json(bundle_dir / "topics.json", [])
        with pytest.raises(FileNotFoundError, match="transcript.txt.*not found"):
            load_bundle(bundle_dir)

    def test_empty_transcript_raises(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "TEST123"
        bundle_dir.mkdir(parents=True)
        _write_json(bundle_dir / "manifest.json", _create_valid_manifest())
        (bundle_dir / "transcript.txt").write_text("", encoding="utf-8")
        _write_json(bundle_dir / "topics.json", [])
        with pytest.raises(ValueError, match="empty"):
            load_bundle(bundle_dir)

    def test_whitespace_only_transcript_raises(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "TEST123"
        bundle_dir.mkdir(parents=True)
        _write_json(bundle_dir / "manifest.json", _create_valid_manifest())
        (bundle_dir / "transcript.txt").write_text("   \n\t  ", encoding="utf-8")
        _write_json(bundle_dir / "topics.json", [])
        with pytest.raises(ValueError, match="empty"):
            load_bundle(bundle_dir)

    def test_missing_topics_file_raises(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "TEST123"
        bundle_dir.mkdir(parents=True)
        _write_json(bundle_dir / "manifest.json", _create_valid_manifest())
        (bundle_dir / "transcript.txt").write_text("Some content.", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="topics.json.*not found"):
            load_bundle(bundle_dir)

    def test_topics_not_array_raises(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "TEST123"
        bundle_dir.mkdir(parents=True)
        _write_json(bundle_dir / "manifest.json", _create_valid_manifest())
        (bundle_dir / "transcript.txt").write_text("Some content.", encoding="utf-8")
        _write_json(bundle_dir / "topics.json", {"not": "an array"})
        with pytest.raises(ValueError, match="JSON array"):
            load_bundle(bundle_dir)

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "TEST123"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "transcript.txt").write_text("Some content.", encoding="utf-8")
        _write_json(bundle_dir / "topics.json", [])
        with pytest.raises(FileNotFoundError, match="manifest.json not found"):
            load_bundle(bundle_dir)

    def test_multiple_topics_loaded(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "TEST123"
        bundle_dir.mkdir(parents=True)
        _write_json(bundle_dir / "manifest.json", _create_valid_manifest())
        (bundle_dir / "transcript.txt").write_text("Content here.", encoding="utf-8")
        topics = [
            {"topic": "AI Safety", "start": 0, "end": 50},
            {"topic": "LLM Engineering", "start": 50, "end": 100},
            {"topic": "Agentic Systems", "start": 100, "end": 150},
        ]
        _write_json(bundle_dir / "topics.json", topics)
        bundle = load_bundle(bundle_dir)
        assert len(bundle.topics) == 3

    def test_empty_topics_array_valid(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "TEST123"
        bundle_dir.mkdir(parents=True)
        _write_json(bundle_dir / "manifest.json", _create_valid_manifest())
        (bundle_dir / "transcript.txt").write_text("Content here.", encoding="utf-8")
        _write_json(bundle_dir / "topics.json", [])
        bundle = load_bundle(bundle_dir)
        assert len(bundle.topics) == 0


class TestBundleToSourceMetadata:
    """Tests for bundle_to_source_metadata function."""

    def test_converts_bundle_to_metadata(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "TEST123"
        _create_valid_bundle(bundle_dir)
        bundle = load_bundle(bundle_dir)
        metadata = bundle_to_source_metadata(bundle)

        assert metadata["source_file"] == "transcript.txt"
        assert metadata["channel_name"] == "TestChannel"
        assert metadata["video_id"] == "TEST123"
        assert metadata["article_title"] == "Test Article Title"
        assert metadata["slug"] == "TEST123"
        assert metadata["publish_date"] == "2026-02-05"
        assert isinstance(metadata["references"], str)

        refs = json.loads(metadata["references"])
        assert len(refs) == 1
        assert refs[0]["channel"] == "TestChannel"

    def test_no_channel_in_references_raises(self) -> None:
        manifest = Manifest(
            article_title="Title",
            slug="SLUG",
            publish_date="2026-01-01",
            source_text_file="transcript.txt",
            topics_file="topics.json",
            references=[
                ManifestReference(
                    type="video",
                    title="Test",
                    url="https://example.com",
                    author="Author",
                    channel="",
                    date="2026-01-01",
                )
            ],
        )
        bundle = LoadedBundle(
            manifest=manifest,
            source_text="content",
            topics=[],
            bundle_dir="/tmp/test",
        )
        with pytest.raises(ValueError, match="No channel name"):
            bundle_to_source_metadata(bundle)

    def test_multiple_references_uses_first_channel(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "MULTI"
        bundle_dir.mkdir(parents=True)
        manifest_data = _create_valid_manifest()
        refs = [
            {
                "type": "video",
                "title": "First",
                "url": "https://example.com/1",
                "author": "Auth1",
                "channel": "FirstChannel",
                "date": "2026-01-01",
            },
            {
                "type": "article",
                "title": "Second",
                "url": "https://example.com/2",
                "author": "Auth2",
                "channel": "SecondChannel",
                "date": "2026-01-02",
            },
        ]
        manifest_data["references"] = refs
        _write_json(bundle_dir / "manifest.json", manifest_data)
        (bundle_dir / "transcript.txt").write_text("Content.", encoding="utf-8")
        _write_json(bundle_dir / "topics.json", [])

        bundle = load_bundle(bundle_dir)
        metadata = bundle_to_source_metadata(bundle)
        assert metadata["channel_name"] == "FirstChannel"
