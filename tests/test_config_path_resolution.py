"""Tests for Config path resolution: absolute vs relative paths."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import Config


def _default_paths(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Build a valid paths dict with all required keys. Override specific keys."""
    base = {
        "data_dir": "data",
        "data_models_dir": "data/models",
        "data_downloads_dir": "data/downloads",
        "data_downloads_videos_dir": "data/downloads/videos",
        "data_downloads_transcripts_dir": "data/downloads/transcripts",
        "data_downloads_transcripts_hallucinations_dir": "data/downloads/transcripts-hallucinations",
        "data_downloads_transcripts_cleaned_dir": "data/downloads/transcripts_cleaned",
        "data_transcripts_topics_dir": "data/downloads/transcripts-topics",
        "data_downloads_audio_dir": "data/downloads/audio",
        "data_downloads_metadata_dir": "data/downloads/metadata",
        "data_output_dir": "data/output",
        "data_input_dir": "data/input",
        "data_temp_dir": "data/temp",
        "data_archive_dir": "data/archive",
        "data_archive_videos_dir": "data/archive/videos",
        "data_logs_dir": "data/logs",
        "data_output_articles_dir": "data/output/articles",
        "data_articles_input_dir": "data/articles/input",
        "reports_dir": "reports",
    }
    if overrides:
        base.update(overrides)
    return base


def _write_config(tmp_path: Path, path_overrides: dict[str, str] | None = None) -> Config:
    """Write a minimal config.yaml at tmp_path/config/config.yaml and return Config.

    Project root = tmp_path (since config is at tmp_path/config/config.yaml,
    project_root = config_path.parent.parent = tmp_path).
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "paths": _default_paths(path_overrides),
        "channels": [],
        "defaults": {
            "encoding_name": "o200k_base",
            "repetition_min_k": 1,
            "repetition_min_repetitions": 5,
            "detect_min_k": 3,
        },
    }

    config_path = config_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return Config(config_path)


# Registry: (getter_method_name, paths_key)
GETTER_REGISTRY: list[tuple[str, str]] = [
    ("getDataDir", "data_dir"),
    ("getDataModelsDir", "data_models_dir"),
    ("getDataDownloadsDir", "data_downloads_dir"),
    ("getDataDownloadsVideosDir", "data_downloads_videos_dir"),
    ("getDataDownloadsTranscriptsDir", "data_downloads_transcripts_dir"),
    ("getDataDownloadsTranscriptsHallucinationsDir", "data_downloads_transcripts_hallucinations_dir"),
    ("getDataDownloadsTranscriptsCleanedDir", "data_downloads_transcripts_cleaned_dir"),
    ("getDataTranscriptsTopicsDir", "data_transcripts_topics_dir"),
    ("getDataDownloadsAudioDir", "data_downloads_audio_dir"),
    ("getDataDownloadsMetadataDir", "data_downloads_metadata_dir"),
    ("getDataOutputDir", "data_output_dir"),
    ("getDataInputDir", "data_input_dir"),
    ("getDataTempDir", "data_temp_dir"),
    ("getDataArchiveDir", "data_archive_dir"),
    ("getDataArchiveVideosDir", "data_archive_videos_dir"),
    ("getDataLogsDir", "data_logs_dir"),
    ("getDataOutputArticlesDir", "data_output_articles_dir"),
    ("getDataArticlesInputDir", "data_articles_input_dir"),
    ("getReportsDir", "reports_dir"),
]


class TestAbsolutePathResolution:
    """TS-1, TS-4b, TS-10, TS-12: Absolute paths returned as-is."""

    def test_absolute_path_returned_as_is(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "/Volumes/data/project"})
        result = config.getDataDir()
        assert result == Path("/Volumes/data/project")
        assert result.is_absolute()

    def test_absolute_path_trailing_slash_stripped(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "/Volumes/data/"})
        result = config.getDataDir()
        assert result == Path("/Volumes/data")
        assert not str(result).endswith("/")

    def test_root_path_is_valid_absolute(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "/"})
        result = config.getDataDir()
        assert result == Path("/")
        assert result.is_absolute()

    def test_nonexistent_path_no_error_at_load(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "/nonexistent/path"})
        result = config.getDataDir()
        assert result == Path("/nonexistent/path")


class TestRelativePathResolution:
    """TS-2, TS-3, TS-4: Relative paths resolved via project root."""

    def test_relative_path_resolved_via_project_root(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "data"})
        result = config.getDataDir()
        assert result == tmp_path / "data"
        assert result.is_absolute()

    def test_project_root_derived_from_config_location(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "data/output"})
        result = config.getDataDir()
        assert result == tmp_path / "data" / "output"

    def test_relative_path_trailing_slash_stripped(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "data/output/"})
        result = config.getDataDir()
        assert result == tmp_path / "data" / "output"
        assert not str(result).endswith("/")


class TestProjectRootDerivation:
    """TS-6, TS-11: Project root from config file location."""

    def test_project_root_independent_of_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _write_config(tmp_path, {"data_dir": "data"})
        monkeypatch.chdir("/tmp")
        result = config.getDataDir()
        assert result == tmp_path / "data"

    def test_config_from_nonstandard_location(self, tmp_path: Path) -> None:
        """Config at tmp_path/test_config.yaml -> project_root = tmp_path.parent."""
        config_path = tmp_path / "test_config.yaml"
        config_data = {
            "paths": _default_paths({"data_dir": "mydata"}),
            "channels": [],
            "defaults": {
                "encoding_name": "o200k_base",
                "repetition_min_k": 1,
                "repetition_min_repetitions": 5,
                "detect_min_k": 3,
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        config = Config(config_path)
        # project_root = config_path.parent.parent = tmp_path.parent
        expected = tmp_path.parent / "mydata"
        assert config.getDataDir() == expected


class TestSchemaValidation:
    """TS-7, TS-9: Pydantic validation for paths."""

    def test_unknown_path_key_rejected(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        paths = _default_paths()
        paths["nonexistent_dir"] = "foo"
        config_data = {
            "paths": paths,
            "channels": [],
            "defaults": {
                "encoding_name": "o200k_base",
                "repetition_min_k": 1,
                "repetition_min_repetitions": 5,
                "detect_min_k": 3,
            },
        }
        config_path = config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        with pytest.raises(ValueError, match="nonexistent_dir|extra"):
            Config(config_path)

    def test_empty_string_path_rejected(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_data = {
            "paths": _default_paths({"data_dir": ""}),
            "channels": [],
            "defaults": {
                "encoding_name": "o200k_base",
                "repetition_min_k": 1,
                "repetition_min_repetitions": 5,
                "detect_min_k": 3,
            },
        }
        config_path = config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        with pytest.raises(ValueError):
            Config(config_path)


class TestEdgeCases:
    """TS-8, TS-13: Edge cases for path resolution."""

    def test_relative_path_with_parent_segments(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "../sibling/data"})
        result = config.getDataDir()
        assert result == tmp_path / ".." / "sibling" / "data"
        assert result.is_absolute()

    def test_symlinks_not_resolved(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "/some/symlinked/path"})
        result = config.getDataDir()
        assert str(result) == "/some/symlinked/path"


class TestExhaustiveGetterCoverage:
    """TS-18, TS-19, TS-20: Every getter works with absolute and relative values."""

    @pytest.mark.parametrize("getter_name,paths_key", GETTER_REGISTRY)
    def test_getter_returns_absolute_with_absolute_value(
        self, tmp_path: Path, getter_name: str, paths_key: str,
    ) -> None:
        config = _write_config(tmp_path, {paths_key: f"/abs/{paths_key}"})
        result = getattr(config, getter_name)()
        assert isinstance(result, Path)
        assert result.is_absolute()
        assert result == Path(f"/abs/{paths_key}")

    @pytest.mark.parametrize("getter_name,paths_key", GETTER_REGISTRY)
    def test_getter_returns_absolute_with_relative_value(
        self, tmp_path: Path, getter_name: str, paths_key: str,
    ) -> None:
        config = _write_config(tmp_path, {paths_key: f"rel/{paths_key}"})
        result = getattr(config, getter_name)()
        assert isinstance(result, Path)
        assert result.is_absolute()
        assert result == tmp_path / "rel" / paths_key

    def test_mixed_absolute_and_relative_in_same_config(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {
            "data_dir": "/absolute/data",
            "data_models_dir": "data/models",
            "data_output_dir": "/absolute/output",
            "reports_dir": "reports",
        })
        assert config.getDataDir() == Path("/absolute/data")
        assert config.getDataModelsDir() == tmp_path / "data" / "models"
        assert config.getDataOutputDir() == Path("/absolute/output")
        assert config.getReportsDir() == tmp_path / "reports"
