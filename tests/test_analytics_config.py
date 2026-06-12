"""Unit tests for AnalyticsConfig and the analytics-related Config getters.

All tests build synthetic configs in tmp_path; they never read the real
config/config.yaml or touch any real data directory.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from src.config import AnalyticsConfig, Config


def valid_paths_dict() -> dict[str, str]:
    """Return a valid paths block (without the optional analytics dir)."""
    return {
        "data_dir": "./data/",
        "data_models_dir": "./data/models/",
        "data_downloads_dir": "./data/downloads",
        "data_downloads_videos_dir": "./data/downloads/videos/",
        "data_downloads_transcripts_dir": "./data/downloads/transcripts",
        "data_downloads_transcripts_hallucinations_dir": "./data/downloads/transcripts-hallucinations",
        "data_downloads_transcripts_cleaned_dir": "./data/downloads/transcripts_cleaned",
        "data_downloads_transcripts_summaries_dir": "./data/downloads/transcripts_summaries",
        "data_downloads_audio_dir": "./data/downloads/audio",
        "data_downloads_metadata_dir": "./data/downloads/metadata",
        "data_output_dir": "./data/output/",
        "data_input_dir": "./data/input/",
        "data_temp_dir": "./data/temp",
        "data_archive_dir": "./data/archive",
        "data_archive_videos_dir": "./data/archive/videos",
        "data_logs_dir": "./data/logs",
        "reports_dir": "reports",
    }


def valid_channels_list() -> list[dict[str, Any]]:
    """Return a minimal valid channels block."""
    return [
        {
            "url": "https://www.youtube.com/@mock",
            "name": "Mock Channel",
            "category": "mock_category",
            "description": "Synthetic channel for tests",
            "download-limiter": 1,
            "transcription-limiter": 1,
            "language": "en",
        }
    ]


def valid_analytics_dict() -> dict[str, Any]:
    """Return a complete valid analytics block."""
    return {
        "lookback_days": 30,
        "timeline_bucket": "week",
        "channel_filter": None,
        "top_n_themes": 30,
        "top_n_terms": 50,
        "top_n_videos_per_theme": 10,
        "min_theme_document_frequency": 2,
        "tfidf_ngram_range_min": 1,
        "tfidf_ngram_range_max": 2,
        "include_cleaned_txt_in_tfidf": False,
        "previous_run_cache": "./.cache/analytics_previous.json",
    }


def write_config(tmp_path: Path, config_data: dict[str, Any]) -> Path:
    """Write a config dict as YAML into tmp_path and return its path."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return config_path


class TestAnalyticsConfigModel:
    """Validation behavior of the AnalyticsConfig pydantic model."""

    def test_valid_block_validates(self) -> None:
        """A complete analytics block validates and exposes typed fields."""
        model = AnalyticsConfig.model_validate(valid_analytics_dict())
        assert model.lookback_days == 30
        assert model.timeline_bucket == "week"
        assert model.channel_filter is None
        assert model.top_n_themes == 30
        assert model.top_n_terms == 50
        assert model.top_n_videos_per_theme == 10
        assert model.min_theme_document_frequency == 2
        assert model.tfidf_ngram_range_min == 1
        assert model.tfidf_ngram_range_max == 2
        assert model.include_cleaned_txt_in_tfidf is False
        assert model.previous_run_cache == Path("./.cache/analytics_previous.json")

    def test_channel_filter_accepts_channel_name(self) -> None:
        """channel_filter may be a channel name string."""
        data = valid_analytics_dict()
        data["channel_filter"] = "Mock Channel"
        model = AnalyticsConfig.model_validate(data)
        assert model.channel_filter == "Mock Channel"

    def test_month_bucket_accepted(self) -> None:
        """timeline_bucket accepts the documented 'month' alternative."""
        data = valid_analytics_dict()
        data["timeline_bucket"] = "month"
        model = AnalyticsConfig.model_validate(data)
        assert model.timeline_bucket == "month"

    def test_missing_channel_filter_rejected(self) -> None:
        """channel_filter is a required key (explicitly nullable, no default)."""
        data = valid_analytics_dict()
        del data["channel_filter"]
        with pytest.raises(ValidationError):
            AnalyticsConfig.model_validate(data)

    def test_missing_lookback_days_rejected(self) -> None:
        """lookback_days is required."""
        data = valid_analytics_dict()
        del data["lookback_days"]
        with pytest.raises(ValidationError):
            AnalyticsConfig.model_validate(data)

    def test_invalid_timeline_bucket_rejected(self) -> None:
        """Unknown timeline_bucket values are rejected at validation time."""
        data = valid_analytics_dict()
        data["timeline_bucket"] = "day"
        with pytest.raises(ValidationError):
            AnalyticsConfig.model_validate(data)

    def test_extra_field_rejected(self) -> None:
        """Unknown keys in the analytics block are rejected (extra='forbid')."""
        data = valid_analytics_dict()
        data["unknown_knob"] = 1
        with pytest.raises(ValidationError):
            AnalyticsConfig.model_validate(data)

    @pytest.mark.parametrize(
        "field",
        [
            "lookback_days",
            "top_n_themes",
            "top_n_terms",
            "top_n_videos_per_theme",
            "min_theme_document_frequency",
            "tfidf_ngram_range_min",
            "tfidf_ngram_range_max",
        ],
    )
    def test_non_positive_numbers_rejected(self, field: str) -> None:
        """All numeric knobs must be strictly positive."""
        data = valid_analytics_dict()
        data[field] = 0
        with pytest.raises(ValidationError):
            AnalyticsConfig.model_validate(data)

    def test_inverted_ngram_range_rejected(self) -> None:
        """tfidf_ngram_range_max below tfidf_ngram_range_min is a config error."""
        data = valid_analytics_dict()
        data["tfidf_ngram_range_min"] = 2
        data["tfidf_ngram_range_max"] = 1
        with pytest.raises(ValidationError, match="tfidf_ngram_range_max"):
            AnalyticsConfig.model_validate(data)

    def test_missing_include_cleaned_txt_rejected(self) -> None:
        """include_cleaned_txt_in_tfidf is a required key (no silent default)."""
        data = valid_analytics_dict()
        del data["include_cleaned_txt_in_tfidf"]
        with pytest.raises(ValidationError):
            AnalyticsConfig.model_validate(data)


class TestConfigAnalyticsGetters:
    """Config-level wiring: optional section, optional path, raising getters."""

    def test_full_config_exposes_analytics(self, tmp_path: Path) -> None:
        """With analytics block and path present, both getters return values."""
        paths = valid_paths_dict()
        paths["data_output_analytics_dir"] = "./data/output/analytics"
        config_path = write_config(
            tmp_path,
            {
                "paths": paths,
                "channels": valid_channels_list(),
                "analytics": valid_analytics_dict(),
            },
        )
        config = Config(config_path)
        analytics = config.get_analytics_config()
        assert analytics.lookback_days == 30
        assert analytics.channel_filter is None
        assert config.get_data_output_analytics_dir() == Path("./data/output/analytics")

    def test_config_without_analytics_still_loads(self, tmp_path: Path) -> None:
        """Configs predating the analytics feature load unchanged."""
        config_path = write_config(
            tmp_path,
            {"paths": valid_paths_dict(), "channels": valid_channels_list()},
        )
        config = Config(config_path)
        assert config.get_channels()[0].name == "Mock Channel"

    def test_missing_analytics_block_getter_raises(self, tmp_path: Path) -> None:
        """get_analytics_config raises KeyError when the block is absent."""
        config_path = write_config(
            tmp_path,
            {"paths": valid_paths_dict(), "channels": valid_channels_list()},
        )
        config = Config(config_path)
        with pytest.raises(KeyError, match="analytics"):
            config.get_analytics_config()

    def test_missing_analytics_dir_getter_raises(self, tmp_path: Path) -> None:
        """get_data_output_analytics_dir raises KeyError when the path is unset."""
        config_path = write_config(
            tmp_path,
            {
                "paths": valid_paths_dict(),
                "channels": valid_channels_list(),
                "analytics": valid_analytics_dict(),
            },
        )
        config = Config(config_path)
        with pytest.raises(KeyError, match="data_output_analytics_dir"):
            config.get_data_output_analytics_dir()

    def test_invalid_analytics_block_fails_load(self, tmp_path: Path) -> None:
        """A present-but-invalid analytics block aborts config loading."""
        analytics = valid_analytics_dict()
        analytics["lookback_days"] = "not-a-number"
        config_path = write_config(
            tmp_path,
            {
                "paths": valid_paths_dict(),
                "channels": valid_channels_list(),
                "analytics": analytics,
            },
        )
        with pytest.raises(ValueError, match="[Aa]nalytics"):
            Config(config_path)

    def test_empty_analytics_dir_rejected(self, tmp_path: Path) -> None:
        """An empty-string analytics dir fails paths validation (min_length=1)."""
        paths = valid_paths_dict()
        paths["data_output_analytics_dir"] = ""
        config_path = write_config(
            tmp_path,
            {"paths": paths, "channels": valid_channels_list()},
        )
        with pytest.raises(ValueError, match="[Pp]aths"):
            Config(config_path)
