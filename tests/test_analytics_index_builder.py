"""Unit tests for the analytics corpus index builder.

All corpora are synthetic and live in tmp_path; tests never touch the real
config or real data directories.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.analytics.errors import AnalyticsError, JoinError, MetadataError
from src.analytics.index_builder import build_index, write_corpus_index
from src.analytics.models import SummaryStats
from src.config import Config

SRT_CONTENT = """1
00:00:00,000 --> 00:00:05,000
Hello world this is a test.

2
00:00:05,000 --> 00:10:00,000
More words follow here.
"""


def minimal_summary_md(topic: str = "Main Topic") -> str:
    """Return a small markdown summary (opaque text; layout is irrelevant)."""
    return f"# Overview\nA short overview of a talk about {topic}.\n\n**Key points:** - First point\n- Second point\n"


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


def make_config(tmp_path: Path, channel_names: list[str]) -> Config:
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
    analytics = {
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
        "previous_run_cache": str(tmp_path / ".cache" / "analytics_previous.json"),
    }
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
    upload_date: str = "20240115",
    with_srt: bool = True,
    with_summary: bool = True,
    with_info: bool = True,
    summary_text: str | None = None,
    id_in_filename: bool = True,
    with_payload_id: bool = True,
) -> None:
    """Create the synthetic on-disk files for one video."""
    data = tmp_path / "data"
    stem = f"{title} [{video_id}]" if id_in_filename else title

    cleaned = data / "downloads" / "transcripts_cleaned" / channel_dir
    cleaned.mkdir(parents=True, exist_ok=True)
    (cleaned / f"{stem}.txt").write_text("hello world transcript words here", encoding="utf-8")
    if with_srt:
        (cleaned / f"{stem}.srt").write_text(SRT_CONTENT, encoding="utf-8")

    if with_summary:
        summaries = data / "downloads" / "transcripts_summaries" / channel_dir
        summaries.mkdir(parents=True, exist_ok=True)
        content = summary_text if summary_text is not None else minimal_summary_md()
        (summaries / f"{stem}.md").write_text(content, encoding="utf-8")

    if with_info:
        metadata = data / "downloads" / "metadata" / channel_dir / "video"
        metadata.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"title": title, "upload_date": upload_date, "description": "Synthetic description."}
        if with_payload_id:
            payload["id"] = video_id
        (metadata / f"{stem}.info.json").write_text(json.dumps(payload), encoding="utf-8")


class TestBuildIndexHappyPath:
    """Joining a complete synthetic corpus."""

    def test_two_channel_corpus_indexed(self, tmp_path: Path) -> None:
        """All videos across channels are joined into sorted records."""
        config = make_config(tmp_path, ["Mock Channel", "Other Channel"])
        add_video(tmp_path, "Mock_Channel", "Video B", "bbbbbbbbbb1", upload_date="20240220")
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", upload_date="20240115")
        add_video(tmp_path, "Other_Channel", "Video C", "cccccccccc1", upload_date="20240101")

        index = build_index(config)

        assert [r.video_id for r in index.records] == ["aaaaaaaaaa1", "bbbbbbbbbb1", "cccccccccc1"]
        record = index.records[0]
        assert record.channel == "Mock_Channel"
        assert record.title == "Video A"
        assert record.upload_date == "2024-01-15"
        assert record.duration_seconds == 600
        assert record.word_count == 5
        assert record.has_summary is True
        assert record.summary_stats is not None
        assert record.summary_stats.word_count > 0
        assert record.summary_stats.char_count > 0
        assert record.channel_meta.language == "en"
        assert record.channel_meta.category == "synthetic_category"
        assert record.channel_meta.description == "Synthetic channel Mock Channel"

    def test_record_paths_point_to_source_files(self, tmp_path: Path) -> None:
        """Each record carries the paths of its joined source files."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")

        record = build_index(config).records[0]
        assert record.paths.cleaned_txt.endswith("Video A [aaaaaaaaaa1].txt")
        assert record.paths.cleaned_srt is not None
        assert record.paths.cleaned_srt.endswith("Video A [aaaaaaaaaa1].srt")
        assert record.paths.summary_md is not None
        assert record.paths.summary_md.endswith("Video A [aaaaaaaaaa1].md")
        assert record.paths.metadata_json.endswith("Video A [aaaaaaaaaa1].info.json")

    def test_missing_summary_is_coverage_gap_not_error(self, tmp_path: Path) -> None:
        """A cleaned transcript without a summary is indexed, not fatal."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", with_summary=False)

        record = build_index(config).records[0]
        assert record.has_summary is False
        assert record.summary_stats is None
        assert record.paths.summary_md is None

    def test_summary_stats_counted_from_normalized_text(self, tmp_path: Path) -> None:
        """Stats reflect the normalized text: markup stripped, words kept."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", summary_text="# Title\nAlpha beta gamma.\n")

        record = build_index(config).records[0]
        assert record.summary_stats == SummaryStats(word_count=4, char_count=23)

    def test_missing_srt_gives_null_duration(self, tmp_path: Path) -> None:
        """A cleaned txt without its srt sibling still indexes, without duration."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", with_srt=False)

        record = build_index(config).records[0]
        assert record.duration_seconds is None
        assert record.paths.cleaned_srt is None

    def test_appledouble_files_ignored(self, tmp_path: Path) -> None:
        """macOS ._ metadata files in the cleaned tree are skipped."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")
        cleaned = tmp_path / "data" / "downloads" / "transcripts_cleaned" / "Mock_Channel"
        (cleaned / "._Video A [aaaaaaaaaa1].txt").write_text("junk", encoding="utf-8")

        assert len(build_index(config).records) == 1


class TestBuildIndexFailFast:
    """Every documented fail-fast condition aborts the run."""

    def test_empty_corpus_fails(self, tmp_path: Path) -> None:
        """An absent or empty cleaned corpus is an error, not an empty report."""
        config = make_config(tmp_path, ["Mock Channel"])
        with pytest.raises(AnalyticsError, match="transcripts_cleaned"):
            build_index(config)

    def test_missing_metadata_fails(self, tmp_path: Path) -> None:
        """A cleaned transcript without its info.json aborts the run."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", with_info=False)
        with pytest.raises(MetadataError, match="aaaaaaaaaa1"):
            build_index(config)

    def test_cleaned_file_without_video_id_or_stem_metadata_fails(self, tmp_path: Path) -> None:
        """No bracketed ID and no stem-matched info.json cannot be joined."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")
        cleaned = tmp_path / "data" / "downloads" / "transcripts_cleaned" / "Mock_Channel"
        (cleaned / "No id here.txt").write_text("words", encoding="utf-8")
        with pytest.raises(JoinError, match="No id here.txt"):
            build_index(config)

    def test_unmatched_channel_dir_fails(self, tmp_path: Path) -> None:
        """A cleaned channel dir with no config match cannot fill channel_meta."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Unknown_Channel", "Video A", "aaaaaaaaaa1")
        with pytest.raises(JoinError, match="Unknown_Channel"):
            build_index(config)

    def test_freeform_summary_is_valid(self, tmp_path: Path) -> None:
        """Any non-empty prose summary indexes; no template is enforced."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(
            tmp_path,
            "Mock_Channel",
            "Video A",
            "aaaaaaaaaa1",
            summary_text="Just some prose, not any template.\n",
        )
        record = build_index(config).records[0]
        assert record.summary_stats == SummaryStats(word_count=6, char_count=34)

    def test_empty_summary_is_coverage_gap(self, tmp_path: Path) -> None:
        """A summary that is empty after normalization counts as missing (Amendment 9)."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", summary_text="## \n\n- \n")
        record = build_index(config).records[0]
        assert record.has_summary is False
        assert record.summary_stats is None
        assert record.paths.summary_md is None

    def test_malformed_metadata_json_fails(self, tmp_path: Path) -> None:
        """Invalid JSON in info.json aborts the run."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", with_info=False)
        metadata = tmp_path / "data" / "downloads" / "metadata" / "Mock_Channel" / "video"
        metadata.mkdir(parents=True, exist_ok=True)
        (metadata / "Video A [aaaaaaaaaa1].info.json").write_text("{broken", encoding="utf-8")
        with pytest.raises(MetadataError, match="invalid JSON"):
            build_index(config)

    def test_duplicate_video_id_indexed_as_separate_records(self, tmp_path: Path) -> None:
        """The same video downloaded twice yields two records, each self-paired.

        The fixed corpus legitimately contains re-downloads of one video (same
        video_id, different title, e.g. after a channel rename). Both are indexed
        faithfully — neither is dropped or merged — and each record is paired to
        the metadata/summary whose filename stem matches its transcript.
        """
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Refactors with Agents, AllHands", "aaaaaaaaaa1")
        add_video(tmp_path, "Mock_Channel", "Refactors with Agents, OpenHands", "aaaaaaaaaa1")
        index = build_index(config)
        assert len(index.records) == 2
        assert {record.video_id for record in index.records} == {"aaaaaaaaaa1"}
        assert {record.title for record in index.records} == {
            "Refactors with Agents, AllHands",
            "Refactors with Agents, OpenHands",
        }
        for record in index.records:
            assert record.title in record.paths.cleaned_txt
            assert record.title in record.paths.metadata_json
            assert record.paths.summary_md is not None and record.title in record.paths.summary_md

    def test_sanitized_channel_name_collision_fails(self, tmp_path: Path) -> None:
        """Two config channels mapping to one directory cannot be joined."""
        config = make_config(tmp_path, ["AI Engineer", "AI-Engineer"])
        add_video(tmp_path, "AI_Engineer", "Video A", "aaaaaaaaaa1")
        with pytest.raises(JoinError, match="AI_Engineer"):
            build_index(config)

    def test_duplicate_metadata_disambiguated_by_stem(self, tmp_path: Path) -> None:
        """When two info.json share an ID, the transcript pairs with its own stem.

        The non-matching duplicate is never even parsed (kept deliberately
        invalid here to prove it), so each transcript reads its own metadata.
        """
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")
        metadata = tmp_path / "data" / "downloads" / "metadata" / "Mock_Channel" / "video"
        (metadata / "Video A copy [aaaaaaaaaa1].info.json").write_text('{"title": "x"}', encoding="utf-8")
        index = build_index(config)
        assert len(index.records) == 1
        assert index.records[0].title == "Video A"
        assert index.records[0].paths.metadata_json.endswith("Video A [aaaaaaaaaa1].info.json")

    def test_duplicate_summary_disambiguated_by_stem(self, tmp_path: Path) -> None:
        """When two summaries share an ID, the transcript pairs with its own stem."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")
        summaries = tmp_path / "data" / "downloads" / "transcripts_summaries" / "Mock_Channel"
        (summaries / "Video A copy [aaaaaaaaaa1].md").write_text(minimal_summary_md("Other Topic"), encoding="utf-8")
        index = build_index(config)
        assert len(index.records) == 1
        assert index.records[0].paths.summary_md is not None
        assert index.records[0].paths.summary_md.endswith("Video A [aaaaaaaaaa1].md")

    def test_ambiguous_duplicate_metadata_fails(self, tmp_path: Path) -> None:
        """Several same-ID info.json with no stem match cannot be paired (Amendment 8)."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", with_info=False)
        metadata = tmp_path / "data" / "downloads" / "metadata" / "Mock_Channel" / "video"
        metadata.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"title": "Video A", "upload_date": "20240115", "description": "x"})
        (metadata / "Other name [aaaaaaaaaa1].info.json").write_text(payload, encoding="utf-8")
        (metadata / "Another name [aaaaaaaaaa1].info.json").write_text(payload, encoding="utf-8")
        with pytest.raises(JoinError, match="cannot be paired"):
            build_index(config)

    def test_ambiguous_duplicate_summaries_become_coverage_gap(self, tmp_path: Path) -> None:
        """Several same-ID summaries with no stem match mean no summary (Amendment 8)."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1", with_summary=False)
        summaries = tmp_path / "data" / "downloads" / "transcripts_summaries" / "Mock_Channel"
        summaries.mkdir(parents=True, exist_ok=True)
        (summaries / "Other name [aaaaaaaaaa1].md").write_text(minimal_summary_md(), encoding="utf-8")
        (summaries / "Another name [aaaaaaaaaa1].md").write_text(minimal_summary_md(), encoding="utf-8")
        record = build_index(config).records[0]
        assert record.has_summary is False
        assert record.summary_stats is None
        assert record.paths.summary_md is None

    def test_undecodable_cleaned_txt_fails(self, tmp_path: Path) -> None:
        """A cleaned .txt that is not UTF-8 aborts with the file path."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")
        cleaned = tmp_path / "data" / "downloads" / "transcripts_cleaned" / "Mock_Channel"
        (cleaned / "Video A [aaaaaaaaaa1].txt").write_bytes(b"\xff\xfe\xfa")
        with pytest.raises(AnalyticsError, match="cannot be read"):
            build_index(config)

    def test_malformed_srt_fails(self, tmp_path: Path) -> None:
        """A cleaned SRT that cannot be parsed aborts with the file path."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")
        cleaned = tmp_path / "data" / "downloads" / "transcripts_cleaned" / "Mock_Channel"
        (cleaned / "Video A [aaaaaaaaaa1].srt").write_text("garbage\nnot an srt\n", encoding="utf-8")
        with pytest.raises(AnalyticsError, match="Video A"):
            build_index(config)


class TestIdlessFilenames:
    """Amendment 10: cleaned files without a bracketed ID join by exact stem."""

    def test_idless_transcript_joins_by_stem(self, tmp_path: Path) -> None:
        """The video_id comes from metadata; siblings pair by exact stem."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Old Video", "aaaaaaaaaa1", id_in_filename=False)

        record = build_index(config).records[0]
        assert record.video_id == "aaaaaaaaaa1"
        assert record.title == "Old Video"
        assert record.has_summary is True
        assert record.paths.cleaned_txt.endswith("Old Video.txt")
        assert record.paths.summary_md is not None
        assert record.paths.summary_md.endswith("Old Video.md")
        assert record.paths.metadata_json.endswith("Old Video.info.json")

    def test_idless_with_redownload_each_keeps_own_summary(self, tmp_path: Path) -> None:
        """An ID-less original and its tokened re-download never cross-join.

        The re-download's summary contains the bracketed ID; a token search
        from the ID-less transcript must not steal it (real-corpus case).
        """
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Old Title", "aaaaaaaaaa1", id_in_filename=False)
        add_video(tmp_path, "Mock_Channel", "New Title", "aaaaaaaaaa1")

        index = build_index(config)
        assert len(index.records) == 2
        assert {record.video_id for record in index.records} == {"aaaaaaaaaa1"}
        by_title = {record.title: record for record in index.records}
        old_summary = by_title["Old Title"].paths.summary_md
        new_summary = by_title["New Title"].paths.summary_md
        assert old_summary is not None and old_summary.endswith("Old Title.md")
        assert new_summary is not None and new_summary.endswith("New Title [aaaaaaaaaa1].md")

    def test_idless_without_stem_metadata_fails(self, tmp_path: Path) -> None:
        """No bracketed ID and no stem-matched info.json is a join failure."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Old Video", "aaaaaaaaaa1", id_in_filename=False, with_info=False)
        with pytest.raises(JoinError, match="no stem-matched info.json"):
            build_index(config)

    def test_idless_metadata_without_id_fails(self, tmp_path: Path) -> None:
        """Stem-matched metadata must carry the 'id' field for ID-less files."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Old Video", "aaaaaaaaaa1", id_in_filename=False, with_payload_id=False)
        with pytest.raises(MetadataError, match="'id'"):
            build_index(config)

    def test_idless_missing_summary_is_coverage_gap(self, tmp_path: Path) -> None:
        """An ID-less transcript without a stem-matched summary is a gap."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Old Video", "aaaaaaaaaa1", id_in_filename=False, with_summary=False)
        record = build_index(config).records[0]
        assert record.has_summary is False
        assert record.summary_stats is None


class TestWriteCorpusIndex:
    """Serialization of the index artifact."""

    def test_writes_corpus_index_json(self, tmp_path: Path) -> None:
        """The index serializes to corpus_index.json in the analytics dir."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")

        index = build_index(config)
        output_path = write_corpus_index(index, config.get_data_output_analytics_dir())

        assert output_path.name == "corpus_index.json"
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(payload["records"]) == 1
        assert payload["records"][0]["video_id"] == "aaaaaaaaaa1"

    def test_output_is_deterministic(self, tmp_path: Path) -> None:
        """Writing the same corpus twice yields byte-identical JSON."""
        config = make_config(tmp_path, ["Mock Channel"])
        add_video(tmp_path, "Mock_Channel", "Video A", "aaaaaaaaaa1")

        output_dir = config.get_data_output_analytics_dir()
        first = write_corpus_index(build_index(config), output_dir).read_bytes()
        second = write_corpus_index(build_index(config), output_dir).read_bytes()
        assert first == second
