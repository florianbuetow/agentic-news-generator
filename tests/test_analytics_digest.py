"""Tests for the research digest orchestration (artifacts, cache, emerging diff).

All corpora are synthetic and live in tmp_path; tests never touch the real
config or real data directories.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.analytics.digest import build_digest
from src.analytics.errors import AnalyticsError
from src.config import Config

REF_DATE = date(2024, 3, 1)

SRT_CONTENT = """1
00:00:00,000 --> 00:00:05,000
Hello world this is a test.

2
00:00:05,000 --> 00:10:00,000
More words follow here.
"""

ARTIFACT_NAMES = [
    "corpus_index.json",
    "themes.json",
    "themes_report.md",
    "timeline.json",
    "timeline_report.md",
    "research_digest.md",
]


def summary_md(body_terms: str) -> str:
    """Return a small markdown summary mentioning body_terms (opaque text)."""
    return f"# Notes\nA talk about {body_terms}.\n\n- Remember {body_terms}\n- {body_terms} matters\n"


def channel_entry(name: str) -> dict[str, Any]:
    """Return a valid channels[] entry for a synthetic channel."""
    return {
        "url": f"https://www.youtube.com/@{name.replace(' ', '')}",
        "name": name,
        "category": "synthetic_category",
        "description": f"Synthetic channel {name}",
        "download-limiter": 1,
        "transcription-limiter": 1,
        "language": "en",
    }


def make_config(tmp_path: Path, channel_names: list[str], **analytics_overrides: Any) -> Config:
    """Write a synthetic config.yaml under tmp_path and load it."""
    data = tmp_path / "data"
    paths = {
        "data_dir": str(data),
        "data_models_dir": str(data / "models"),
        "data_downloads_dir": str(data / "downloads"),
        "data_downloads_videos_dir": str(data / "downloads" / "videos"),
        "data_downloads_transcripts_dir": str(data / "downloads" / "transcripts"),
        "data_downloads_transcripts_hallucinations_dir": str(data / "downloads" / "transcripts-hallucinations"),
        "data_downloads_transcripts_cleaned_dir": str(data / "downloads" / "transcripts_cleaned"),
        "data_downloads_transcripts_summaries_dir": str(data / "downloads" / "transcripts_summaries"),
        "data_downloads_audio_dir": str(data / "downloads" / "audio"),
        "data_downloads_metadata_dir": str(data / "downloads" / "metadata"),
        "data_output_dir": str(data / "output"),
        "data_input_dir": str(data / "input"),
        "data_temp_dir": str(data / "temp"),
        "data_archive_dir": str(data / "archive"),
        "data_archive_videos_dir": str(data / "archive" / "videos"),
        "data_logs_dir": str(data / "logs"),
        "reports_dir": str(tmp_path / "reports"),
        "data_output_analytics_dir": str(data / "output" / "analytics"),
    }
    analytics: dict[str, Any] = {
        "lookback_days": 60,
        "timeline_bucket": "week",
        "channel_filter": None,
        "top_n_themes": 30,
        "top_n_terms": 50,
        "top_n_videos_per_theme": 10,
        "min_theme_document_frequency": 1,
        "tfidf_ngram_range_min": 1,
        "tfidf_ngram_range_max": 2,
        "include_cleaned_txt_in_tfidf": False,
        "previous_run_cache": str(tmp_path / ".cache" / "analytics_previous.json"),
    }
    analytics.update(analytics_overrides)
    config_data: dict[str, Any] = {
        "paths": paths,
        "channels": [channel_entry(name) for name in channel_names],
        "analytics": analytics,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return Config(config_path)


def add_video(
    tmp_path: Path,
    channel_dir: str,
    title: str,
    video_id: str,
    upload_date: str = "20240215",
    with_summary: bool = True,
    body_terms: str = "production pipelines",
) -> None:
    """Create the synthetic on-disk files for one video."""
    data = tmp_path / "data"
    stem = f"{title} [{video_id}]"

    cleaned = data / "downloads" / "transcripts_cleaned" / channel_dir
    cleaned.mkdir(parents=True, exist_ok=True)
    (cleaned / f"{stem}.txt").write_text("hello world transcript words here", encoding="utf-8")
    (cleaned / f"{stem}.srt").write_text(SRT_CONTENT, encoding="utf-8")

    if with_summary:
        summaries = data / "downloads" / "transcripts_summaries" / channel_dir
        summaries.mkdir(parents=True, exist_ok=True)
        (summaries / f"{stem}.md").write_text(summary_md(body_terms), encoding="utf-8")

    metadata = data / "downloads" / "metadata" / channel_dir / "video"
    metadata.mkdir(parents=True, exist_ok=True)
    payload = {"title": title, "upload_date": upload_date, "description": "Synthetic description."}
    (metadata / f"{stem}.info.json").write_text(json.dumps(payload), encoding="utf-8")


def analytics_dir(tmp_path: Path) -> Path:
    """The analytics output directory of the synthetic config."""
    return tmp_path / "data" / "output" / "analytics"


def cache_path(tmp_path: Path) -> Path:
    """The snapshot cache path of the synthetic config."""
    return tmp_path / ".cache" / "analytics_previous.json"


class TestDigestArtifacts:
    """The digest run produces the full artifact set."""

    def test_all_artifacts_written(self, tmp_path: Path) -> None:
        """One run writes index, themes, timeline, digest, and the cache."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")

        build_digest(config, REF_DATE)

        for name in ARTIFACT_NAMES:
            assert (analytics_dir(tmp_path) / name).is_file(), name
        assert cache_path(tmp_path).is_file()

    def test_corpus_snapshot_stats(self, tmp_path: Path) -> None:
        """The digest reports corpus counts, coverage, and date range."""
        config = make_config(tmp_path, ["Mock Channel", "Other Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", upload_date="20240115")
        add_video(tmp_path, "Mock_Channel", "Video B", "bbbbbbbbbb1", upload_date="20240220")
        add_video(tmp_path, "Other_Channel", "Video C", "cccccccccc1", upload_date="20240210", with_summary=False)

        build_digest(config, REF_DATE)

        digest = (analytics_dir(tmp_path) / "research_digest.md").read_text(encoding="utf-8")
        assert "Generated: 2024-03-01" in digest
        assert "- Videos indexed: 3" in digest
        assert "- With summaries: 2 (66.7%)" in digest
        assert "- Date range: 2024-01-15 — 2024-02-20" in digest
        assert "- Channels: 2" in digest

    def test_reading_list_lists_phrase_videos(self, tmp_path: Path) -> None:
        """The reading list links phrase-theme videos with channel, date, path."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", body_terms="retrieval pipelines")

        build_digest(config, REF_DATE)

        digest = (analytics_dir(tmp_path) / "research_digest.md").read_text(encoding="utf-8")
        assert "## Reading List" in digest
        assert "### retrieval pipelines" in digest
        assert "**Video A** — Mock_Channel, 2024-02-15" in digest
        assert "Video A [aaaaaaaaaa1].md" in digest


class TestEmergingDiff:
    """Cache snapshot lifecycle and week-over-week diffing."""

    def test_first_run_has_no_previous(self, tmp_path: Path) -> None:
        """The first run renders a first-run note and writes the snapshot."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")

        build_digest(config, REF_DATE)

        digest = (analytics_dir(tmp_path) / "research_digest.md").read_text(encoding="utf-8")
        assert "First run" in digest
        snapshot = json.loads(cache_path(tmp_path).read_text(encoding="utf-8"))
        assert snapshot["generated_on"] == "2024-03-01"
        assert isinstance(snapshot["top_terms"], list)
        assert isinstance(snapshot["top_phrases"], list)

    def test_second_run_reports_new_terms_and_phrases(self, tmp_path: Path) -> None:
        """New vocabulary and new phrases since the last run surface."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", body_terms="kubernetes clusters")
        build_digest(config, REF_DATE)

        add_video(
            tmp_path,
            "Mock_Channel",
            "Video B",
            "bbbbbbbbbb1",
            upload_date="20240225",
            body_terms="quantum entanglement",
        )
        build_digest(config, REF_DATE)

        digest = (analytics_dir(tmp_path) / "research_digest.md").read_text(encoding="utf-8")
        assert "First run" not in digest
        emerging = digest.split("## Emerging")[1].split("## Reading List")[0]
        assert "quantum" in emerging
        assert "quantum entanglement" in emerging

    def test_corrupt_cache_fails(self, tmp_path: Path) -> None:
        """A corrupt snapshot cache aborts the run before any artifact is written."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")
        cache_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        cache_path(tmp_path).write_text("{broken", encoding="utf-8")

        with pytest.raises(AnalyticsError, match="analytics_previous.json"):
            build_digest(config, REF_DATE)

        assert not analytics_dir(tmp_path).exists()
        assert cache_path(tmp_path).read_text(encoding="utf-8") == "{broken"

    def test_corrupt_cache_preserves_existing_artifacts(self, tmp_path: Path) -> None:
        """A corrupt cache on a later run leaves prior artifacts byte-identical."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")
        build_digest(config, REF_DATE)
        before = {name: (analytics_dir(tmp_path) / name).read_bytes() for name in ARTIFACT_NAMES}
        cache_path(tmp_path).write_text("{broken", encoding="utf-8")

        with pytest.raises(AnalyticsError, match="analytics_previous.json"):
            build_digest(config, REF_DATE)

        after = {name: (analytics_dir(tmp_path) / name).read_bytes() for name in ARTIFACT_NAMES}
        assert after == before
