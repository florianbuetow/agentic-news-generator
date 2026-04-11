import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import Config
from src.file_processing_filter import FileProcessingFilter


def get_valid_paths_config(tmp_path: Path) -> dict[str, str]:
    base_data_dir = tmp_path / "data"
    return {
        "data_dir": str(base_data_dir) + "/",
        "data_models_dir": str(base_data_dir / "models") + "/",
        "data_downloads_dir": str(base_data_dir / "downloads"),
        "data_downloads_videos_dir": str(base_data_dir / "downloads" / "videos") + "/",
        "data_downloads_transcripts_dir": str(base_data_dir / "downloads" / "transcripts"),
        "data_downloads_transcripts_hallucinations_dir": str(base_data_dir / "downloads" / "transcripts-hallucinations"),
        "data_downloads_transcripts_cleaned_dir": str(base_data_dir / "downloads" / "transcripts_cleaned"),
        "data_downloads_audio_dir": str(base_data_dir / "downloads" / "audio") + "/",
        "data_downloads_metadata_dir": str(base_data_dir / "downloads" / "metadata"),
        "data_output_dir": str(base_data_dir / "output") + "/",
        "data_input_dir": str(base_data_dir / "input") + "/",
        "data_temp_dir": str(base_data_dir / "temp"),
        "data_archive_dir": str(base_data_dir / "archive"),
        "data_archive_videos_dir": str(base_data_dir / "archive" / "videos"),
        "data_logs_dir": str(base_data_dir / "logs"),
        "reports_dir": str(tmp_path / "reports"),
    }


def write_config_file(tmp_path: Path, paths_config: dict[str, str] | None = None) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_data: dict[str, Any] = {
        "paths": paths_config if paths_config is not None else get_valid_paths_config(tmp_path),
        "channels": [],
    }

    config_path = config_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config_data, config_file, sort_keys=False)

    return config_path


def write_filter_file(config_path: Path, filter_data: dict[str, list[str]]) -> Path:
    filter_path = config_path.parent / "filefilter.json"
    with filter_path.open("w", encoding="utf-8") as filter_file:
        json.dump(filter_data, filter_file)
    return filter_path


def test_extract_config_paths(tmp_path: Path) -> None:
    config_path = write_config_file(tmp_path)
    config = Config(config_path)
    extracted_paths = FileProcessingFilter.extract_config_paths(config)

    expected: dict[str, Any] = config.get_paths_config().model_dump()
    assert extracted_paths == {str(k): str(v) for k, v in expected.items()}


def test_build_path_to_key_lookup(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    transcripts_dir = tmp_path / "transcripts"
    config_paths = {
        "data_downloads_audio_dir": str(audio_dir) + "/",
        "data_downloads_transcripts_dir": str(transcripts_dir),
    }

    path_lookup = FileProcessingFilter.build_path_to_key_lookup(config_paths)

    assert path_lookup[str(audio_dir.resolve())] == "data_downloads_audio_dir"
    assert path_lookup[str(transcripts_dir.resolve())] == "data_downloads_transcripts_dir"


def test_load_filter_file_missing(tmp_path: Path) -> None:
    missing_filter_path = tmp_path / "config" / "filefilter.json"

    with pytest.raises(FileNotFoundError, match="Filter file not found"):
        FileProcessingFilter.load_filter_file(missing_filter_path)


def test_load_filter_file_valid(tmp_path: Path) -> None:
    filter_path = tmp_path / "filefilter.json"
    with filter_path.open("w", encoding="utf-8") as filter_file:
        json.dump(
            {
                "data_downloads_audio_dir": ["channel_a/abc123", "channel_a/def456"],
                "data_downloads_transcripts_dir": [],
            },
            filter_file,
        )

    loaded_filter = FileProcessingFilter.load_filter_file(filter_path)

    assert loaded_filter == {
        "data_downloads_audio_dir": {"channel_a/abc123", "channel_a/def456"},
        "data_downloads_transcripts_dir": set(),
    }


def test_load_filter_file_invalid_keys(tmp_path: Path) -> None:
    config_path = write_config_file(tmp_path)
    write_filter_file(config_path, {"invalid_path_key": ["channel_a/abc123"]})
    config = Config(config_path)

    with pytest.raises(ValueError, match="invalid_path_key"):
        FileProcessingFilter(config)


def test_resolve_config_key_valid(tmp_path: Path) -> None:
    config_path = write_config_file(tmp_path)
    write_filter_file(config_path, {})
    config = Config(config_path)
    file_filter = FileProcessingFilter(config)
    audio_dir = config.get_paths_config().data_downloads_audio_dir
    file_path = Path(audio_dir) / "channel_a" / "Some Title [abc123].wav"

    assert file_filter.should_skip_file(str(file_path), audio_dir) is False


def test_resolve_config_key_invalid(tmp_path: Path) -> None:
    config_path = write_config_file(tmp_path)
    write_filter_file(config_path, {})
    config = Config(config_path)
    file_filter = FileProcessingFilter(config)
    unknown_dir = str(tmp_path / "unknown")
    fake_file = str(tmp_path / "unknown" / "file.wav")

    with pytest.raises(ValueError, match="Unknown base_dir"):
        file_filter.should_skip_file(fake_file, unknown_dir)


def test_should_skip_file_true(tmp_path: Path) -> None:
    config_path = write_config_file(tmp_path)
    write_filter_file(config_path, {"data_downloads_audio_dir": ["channel_a/abc123"]})
    config = Config(config_path)
    file_filter = FileProcessingFilter(config)
    base_dir = config.get_paths_config().data_downloads_audio_dir
    file_path = Path(base_dir) / "channel_a" / "Some Title [abc123].wav"

    assert file_filter.should_skip_file(str(file_path), base_dir) is True


def test_should_skip_file_false(tmp_path: Path) -> None:
    config_path = write_config_file(tmp_path)
    write_filter_file(config_path, {"data_downloads_audio_dir": ["channel_a/abc123"]})
    config = Config(config_path)
    file_filter = FileProcessingFilter(config)
    base_dir = config.get_paths_config().data_downloads_audio_dir
    file_path = Path(base_dir) / "channel_a" / "Other Title [xyz789].wav"

    assert file_filter.should_skip_file(str(file_path), base_dir) is False


def test_should_skip_file_no_entry_for_key(tmp_path: Path) -> None:
    config_path = write_config_file(tmp_path)
    write_filter_file(config_path, {"data_downloads_transcripts_dir": ["channel_a/abc123"]})
    config = Config(config_path)
    file_filter = FileProcessingFilter(config)
    base_dir = config.get_paths_config().data_downloads_audio_dir
    file_path = Path(base_dir) / "channel_a" / "Some Title [abc123].wav"

    assert file_filter.should_skip_file(str(file_path), base_dir) is False


def test_path_normalization_trailing_slash(tmp_path: Path) -> None:
    config_path = write_config_file(tmp_path)
    write_filter_file(config_path, {"data_downloads_audio_dir": ["channel_a/abc123"]})
    config = Config(config_path)
    file_filter = FileProcessingFilter(config)
    base_dir_with_slash = config.get_paths_config().data_downloads_audio_dir
    base_dir_resolved = str(Path(base_dir_with_slash).resolve())
    file_path = Path(base_dir_resolved) / "channel_a" / "Some Title [abc123].wav"

    assert file_filter.should_skip_file(str(file_path), base_dir_resolved) is True
    assert file_filter.should_skip_file(str(file_path), base_dir_with_slash) is True
