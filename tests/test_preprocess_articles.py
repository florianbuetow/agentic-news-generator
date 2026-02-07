"""Tests for article preprocessor script functions."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Import the preprocessor script as a module
script_path = Path(__file__).parent.parent / "scripts" / "preprocess-articles.py"
spec = importlib.util.spec_from_file_location("preprocess_articles", script_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load script: {script_path}")
preprocess_module = importlib.util.module_from_spec(spec)
sys.modules["preprocess_articles"] = preprocess_module
spec.loader.exec_module(preprocess_module)

find_transcript = preprocess_module.find_transcript
find_metadata = preprocess_module.find_metadata
find_topics = preprocess_module.find_topics
extract_metadata_fields = preprocess_module.extract_metadata_fields
create_manifest = preprocess_module.create_manifest


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


class TestFindTranscript:
    """Tests for find_transcript function."""

    def test_finds_matching_transcript(self, tmp_path: Path) -> None:
        channel_dir = tmp_path / "TestChannel"
        channel_dir.mkdir()
        transcript = channel_dir / "Some Title [ABC123].txt"
        transcript.write_text("content", encoding="utf-8")

        result = find_transcript(tmp_path, "ABC123")
        assert result == transcript

    def test_finds_transcript_in_nested_directory(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "channel" / "subdir"
        nested_dir.mkdir(parents=True)
        transcript = nested_dir / "Video [XYZ789].txt"
        transcript.write_text("content", encoding="utf-8")

        result = find_transcript(tmp_path, "XYZ789")
        assert result == transcript

    def test_no_match_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No cleaned transcript found"):
            find_transcript(tmp_path, "NONEXISTENT")

    def test_multiple_matches_raises_value_error(self, tmp_path: Path) -> None:
        channel_dir = tmp_path / "channel"
        channel_dir.mkdir()
        (channel_dir / "First [DUP001].txt").write_text("a", encoding="utf-8")
        (channel_dir / "Second [DUP001].txt").write_text("b", encoding="utf-8")

        with pytest.raises(ValueError, match="Multiple transcripts found"):
            find_transcript(tmp_path, "DUP001")

    def test_non_txt_files_ignored(self, tmp_path: Path) -> None:
        channel_dir = tmp_path / "channel"
        channel_dir.mkdir()
        (channel_dir / "Video [VID001].json").write_text("{}", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="No cleaned transcript found"):
            find_transcript(tmp_path, "VID001")

    def test_empty_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No cleaned transcript found"):
            find_transcript(tmp_path, "ANY123")


class TestFindMetadata:
    """Tests for find_metadata function."""

    def test_finds_matching_metadata(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "TestChannel" / "video"
        video_dir.mkdir(parents=True)
        metadata = video_dir / "Some Title [ABC123].info.json"
        _write_json(metadata, {"title": "Test"})

        result = find_metadata(tmp_path, "TestChannel", "ABC123")
        assert result == metadata

    def test_missing_channel_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Video metadata directory not found"):
            find_metadata(tmp_path, "NonexistentChannel", "ABC123")

    def test_no_match_raises_file_not_found(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "TestChannel" / "video"
        video_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="No metadata found"):
            find_metadata(tmp_path, "TestChannel", "NONEXISTENT")

    def test_multiple_matches_raises_value_error(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "TestChannel" / "video"
        video_dir.mkdir(parents=True)
        _write_json(video_dir / "First [DUP001].info.json", {"title": "A"})
        _write_json(video_dir / "Second [DUP001].info.json", {"title": "B"})

        with pytest.raises(ValueError, match="Multiple metadata files found"):
            find_metadata(tmp_path, "TestChannel", "DUP001")


class TestFindTopics:
    """Tests for find_topics function."""

    def test_finds_topics_with_video_id_in_dict(self, tmp_path: Path) -> None:
        channel_dir = tmp_path / "TestChannel"
        channel_dir.mkdir()
        topics_file = channel_dir / "topics_001.json"
        _write_json(topics_file, {"video_id": "ABC123", "topics": []})

        result = find_topics(tmp_path, "TestChannel", "ABC123")
        assert result == topics_file

    def test_finds_topics_with_video_id_in_list(self, tmp_path: Path) -> None:
        channel_dir = tmp_path / "TestChannel"
        channel_dir.mkdir()
        topics_file = channel_dir / "topics_002.json"
        _write_json(topics_file, [{"video_id": "ABC123", "topic": "AI"}])

        result = find_topics(tmp_path, "TestChannel", "ABC123")
        assert result == topics_file

    def test_missing_channel_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Channel topics directory not found"):
            find_topics(tmp_path, "NonexistentChannel", "ABC123")

    def test_no_match_raises_file_not_found(self, tmp_path: Path) -> None:
        channel_dir = tmp_path / "TestChannel"
        channel_dir.mkdir()
        _write_json(channel_dir / "topics.json", {"video_id": "OTHER"})

        with pytest.raises(FileNotFoundError, match="No topics file found"):
            find_topics(tmp_path, "TestChannel", "ABC123")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        channel_dir = tmp_path / "TestChannel"
        channel_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No topics file found"):
            find_topics(tmp_path, "TestChannel", "ABC123")

    def test_non_json_files_ignored(self, tmp_path: Path) -> None:
        channel_dir = tmp_path / "TestChannel"
        channel_dir.mkdir()
        (channel_dir / "topics.txt").write_text("video_id: ABC123", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="No topics file found"):
            find_topics(tmp_path, "TestChannel", "ABC123")


class TestExtractMetadataFields:
    """Tests for extract_metadata_fields function."""

    def test_extracts_all_fields(self, tmp_path: Path) -> None:
        metadata_path = tmp_path / "video.info.json"
        _write_json(
            metadata_path,
            {
                "title": "Test Video Title",
                "upload_date": "20260205",
                "uploader": "Author Name",
                "channel": "TestChannel",
                "uploader_id": "test_id",
            },
        )

        result = extract_metadata_fields(metadata_path)
        assert result["title"] == "Test Video Title"
        assert result["publish_date"] == "2026-02-05"
        assert result["author"] == "Author Name"
        assert result["channel"] == "TestChannel"

    def test_date_formatting_yyyymmdd(self, tmp_path: Path) -> None:
        metadata_path = tmp_path / "video.info.json"
        _write_json(
            metadata_path,
            {
                "title": "Title",
                "upload_date": "20251231",
            },
        )

        result = extract_metadata_fields(metadata_path)
        assert result["publish_date"] == "2025-12-31"

    def test_non_8_char_date_kept_as_is(self, tmp_path: Path) -> None:
        metadata_path = tmp_path / "video.info.json"
        _write_json(
            metadata_path,
            {
                "title": "Title",
                "upload_date": "2025-01-01",
            },
        )

        result = extract_metadata_fields(metadata_path)
        assert result["publish_date"] == "2025-01-01"

    def test_empty_date_kept_as_is(self, tmp_path: Path) -> None:
        metadata_path = tmp_path / "video.info.json"
        _write_json(
            metadata_path,
            {
                "title": "Title",
                "upload_date": "",
            },
        )

        result = extract_metadata_fields(metadata_path)
        assert result["publish_date"] == ""

    def test_missing_title_raises(self, tmp_path: Path) -> None:
        metadata_path = tmp_path / "video.info.json"
        _write_json(metadata_path, {"upload_date": "20260101"})

        with pytest.raises(ValueError, match="Missing or empty 'title'"):
            extract_metadata_fields(metadata_path)

    def test_empty_title_raises(self, tmp_path: Path) -> None:
        metadata_path = tmp_path / "video.info.json"
        _write_json(metadata_path, {"title": "", "upload_date": "20260101"})

        with pytest.raises(ValueError, match="Missing or empty 'title'"):
            extract_metadata_fields(metadata_path)

    def test_non_dict_root_raises(self, tmp_path: Path) -> None:
        metadata_path = tmp_path / "video.info.json"
        _write_json(metadata_path, [1, 2, 3])

        with pytest.raises(ValueError, match="must be a JSON object"):
            extract_metadata_fields(metadata_path)

    def test_channel_falls_back_to_uploader_id(self, tmp_path: Path) -> None:
        metadata_path = tmp_path / "video.info.json"
        _write_json(
            metadata_path,
            {
                "title": "Title",
                "channel": "",
                "uploader_id": "fallback_id",
            },
        )

        result = extract_metadata_fields(metadata_path)
        assert result["channel"] == "fallback_id"

    def test_missing_optional_fields_returns_empty_strings(self, tmp_path: Path) -> None:
        metadata_path = tmp_path / "video.info.json"
        _write_json(metadata_path, {"title": "Title"})

        result = extract_metadata_fields(metadata_path)
        assert result["author"] == ""
        assert result["channel"] == ""
        assert result["publish_date"] == ""


class TestCreateManifest:
    """Tests for create_manifest function."""

    def test_creates_valid_manifest(self) -> None:
        metadata_fields = {
            "title": "Test Article",
            "publish_date": "2026-02-05",
            "author": "Author",
            "channel": "TestChannel",
        }

        result = create_manifest(
            video_id="VID001",
            metadata_fields=metadata_fields,
            video_url="https://www.youtube.com/watch?v=VID001",
        )

        assert result["article_title"] == "Test Article"
        assert result["slug"] == "VID001"
        assert result["publish_date"] == "2026-02-05"
        assert result["source_text_file"] == "transcript.txt"
        assert result["topics_file"] == "topics.json"

    def test_references_structure(self) -> None:
        metadata_fields = {
            "title": "My Video",
            "publish_date": "2026-01-01",
            "author": "Creator",
            "channel": "MyChannel",
        }

        result = create_manifest(
            video_id="ABC",
            metadata_fields=metadata_fields,
            video_url="https://www.youtube.com/watch?v=ABC",
        )

        refs_raw = result["references"]
        assert isinstance(refs_raw, list)
        assert len(refs_raw) == 1  # pyright: ignore[reportUnknownArgumentType]
        ref_raw = refs_raw[0]  # pyright: ignore[reportUnknownVariableType]
        assert isinstance(ref_raw, dict)
        ref: dict[str, str] = ref_raw  # pyright: ignore[reportUnknownVariableType]
        assert ref["type"] == "video"
        assert ref["title"] == "My Video"
        assert ref["url"] == "https://www.youtube.com/watch?v=ABC"
        assert ref["author"] == "Creator"
        assert ref["channel"] == "MyChannel"
        assert ref["date"] == "2026-01-01"

    def test_manifest_slug_matches_video_id(self) -> None:
        metadata_fields = {
            "title": "Title",
            "publish_date": "2026-01-01",
            "author": "",
            "channel": "Chan",
        }

        result = create_manifest(
            video_id="KX0GurmgAoo",
            metadata_fields=metadata_fields,
            video_url="https://www.youtube.com/watch?v=KX0GurmgAoo",
        )

        assert result["slug"] == "KX0GurmgAoo"
