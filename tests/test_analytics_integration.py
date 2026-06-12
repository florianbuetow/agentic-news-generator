"""End-to-end integration tests for the analytics feature.

A fully synthetic corpus (real on-disk layout) is built in tmp_path together
with a complete mock config.yaml. Summary fixtures are deliberately
heterogeneous — one template-shaped, one freeform prose — proving analytics
treats summaries as opaque text (Amendment 7). A real Config drives the full
chain build_index → rank_themes → build_timeline → build_digest, and
assertions run against the actual emitted artifacts. No test touches the real
config or any real data directory.
"""

import json
import py_compile
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.analytics.digest import build_digest
from src.analytics.errors import AnalyticsError, JoinError, MetadataError
from src.config import Config

REF_DATE = date(2024, 3, 1)

SRT_CONTENT = """1
00:00:00,000 --> 00:00:05,000
Hello world this is a test.

2
00:00:05,000 --> 00:10:00,000
More words follow here.
"""

TXT_CONTENT = "eight deterministic words are in this transcript file"


def template_shaped_summary(topic: str, terms: str) -> str:
    """Markdown that happens to look like the summarize template.

    Analytics must treat this as opaque text — no field is ever extracted.
    """
    return (
        "# Overview\n"
        f"This talk covers {topic}.\n"
        "\n"
        "# Index\n"
        f"1. {topic}\n"
        "\n"
        "# Section Summaries\n"
        "\n"
        f"## 1. {topic}\n"
        "\n"
        f"**Key points:** - First point about {terms}\n"
        f"- Second point about {terms}\n"
        "\n"
        "# Key Takeaways\n"
        f"- {terms} matter\n"
    )


def freeform_summary(terms: str) -> str:
    """Freeform prose markdown — equally valid analytics input."""
    return f"Rambling notes about {terms}, written without any template.\n\n> A quote about {terms}.\n\nClosing thought: {terms} again.\n"


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
    """Write a complete mock config.yaml under tmp_path and load it."""
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
        "min_theme_document_frequency": 2,
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
    upload_date: str,
    summary: str | None,
    id_in_filename: bool = True,
) -> None:
    """Create the full synthetic on-disk file set for one video."""
    data = tmp_path / "data"
    stem = f"{title} [{video_id}]" if id_in_filename else title

    cleaned = data / "downloads" / "transcripts_cleaned" / channel_dir
    cleaned.mkdir(parents=True, exist_ok=True)
    (cleaned / f"{stem}.txt").write_text(TXT_CONTENT, encoding="utf-8")
    (cleaned / f"{stem}.srt").write_text(SRT_CONTENT, encoding="utf-8")

    if summary is not None:
        summaries = data / "downloads" / "transcripts_summaries" / channel_dir
        summaries.mkdir(parents=True, exist_ok=True)
        (summaries / f"{stem}.md").write_text(summary, encoding="utf-8")

    metadata = data / "downloads" / "metadata" / channel_dir / "video"
    metadata.mkdir(parents=True, exist_ok=True)
    payload = {"title": title, "upload_date": upload_date, "description": "Synthetic description.", "id": video_id}
    (metadata / f"{stem}.info.json").write_text(json.dumps(payload), encoding="utf-8")


def build_demo_corpus(tmp_path: Path) -> Config:
    """Two channels, four videos: template-shaped and freeform summaries, one missing."""
    config = make_config(tmp_path, ["Alpha Channel", "Beta Channel"])
    add_video(
        tmp_path,
        "Alpha_Channel",
        "Evals Deep Dive",
        "aaaaaaaaaa1",
        "20240205",
        summary=template_shaped_summary("Production Evals", "production evals"),
    )
    add_video(
        tmp_path,
        "Alpha_Channel",
        "Evals at Scale",
        "bbbbbbbbbb1",
        "20240220",
        summary=template_shaped_summary("Production Evals", "production evals"),
    )
    add_video(
        tmp_path,
        "Beta_Channel",
        "Agent Talk",
        "cccccccccc1",
        "20240221",
        summary=freeform_summary("agent harness loops"),
    )
    add_video(
        tmp_path,
        "Beta_Channel",
        "No Summary Yet",
        "dddddddddd1",
        "20240222",
        summary=None,
    )
    return config


def analytics_dir(tmp_path: Path) -> Path:
    """The analytics output directory of the synthetic config."""
    return tmp_path / "data" / "output" / "analytics"


def read_artifact(tmp_path: Path, name: str) -> str:
    """Read one emitted artifact as text."""
    return (analytics_dir(tmp_path) / name).read_text(encoding="utf-8")


class TestFullChain:
    """The complete chain over a realistic synthetic corpus."""

    def test_corpus_index_artifact(self, tmp_path: Path) -> None:
        """corpus_index.json joins all four videos with summary stats only."""
        config = build_demo_corpus(tmp_path)
        build_digest(config, REF_DATE)

        payload = json.loads(read_artifact(tmp_path, "corpus_index.json"))
        records = payload["records"]
        assert [record["video_id"] for record in records] == [
            "aaaaaaaaaa1",
            "bbbbbbbbbb1",
            "cccccccccc1",
            "dddddddddd1",
        ]
        first = records[0]
        assert first["channel"] == "Alpha_Channel"
        assert first["title"] == "Evals Deep Dive"
        assert first["upload_date"] == "2024-02-05"
        assert first["duration_seconds"] == 600
        assert first["word_count"] == 8
        assert first["has_summary"] is True
        assert first["channel_meta"]["category"] == "synthetic_category"
        assert first["summary_stats"]["word_count"] > 0
        assert first["summary_stats"]["char_count"] > 0
        assert "summary_parsed" not in first
        freeform = records[2]
        assert freeform["has_summary"] is True
        assert freeform["summary_stats"]["word_count"] > 0
        missing = records[3]
        assert missing["has_summary"] is False
        assert missing["summary_stats"] is None

    def test_themes_artifact_terms_and_phrases(self, tmp_path: Path) -> None:
        """Shared phrases meet the document frequency; terms are extracted."""
        config = build_demo_corpus(tmp_path)
        build_digest(config, REF_DATE)

        payload = json.loads(read_artifact(tmp_path, "themes.json"))
        assert payload["video_count"] == 3
        assert "evals" in [term["term"] for term in payload["term_themes"]]
        phrases = {phrase["term"]: phrase for phrase in payload["phrase_themes"]}
        assert "production evals" in phrases
        production_evals = phrases["production evals"]
        assert production_evals["document_frequency"] == 2
        assert production_evals["channels"] == ["Alpha_Channel"]
        assert [video["video_id"] for video in production_evals["videos"]] == ["aaaaaaaaaa1", "bbbbbbbbbb1"]
        assert "agent harness" not in phrases  # document frequency 1 < threshold 2

    def test_timeline_artifact_buckets(self, tmp_path: Path) -> None:
        """Summarized videos land in their ISO weeks with channel breakdowns."""
        config = build_demo_corpus(tmp_path)
        build_digest(config, REF_DATE)

        payload = json.loads(read_artifact(tmp_path, "timeline.json"))
        assert payload["video_count"] == 3
        assert [bucket["bucket"] for bucket in payload["buckets"]] == ["2024-W06", "2024-W08"]
        last_bucket = payload["buckets"][1]
        assert last_bucket["channels"] == {"Alpha_Channel": 1, "Beta_Channel": 1}

    def test_research_digest_sections(self, tmp_path: Path) -> None:
        """The digest carries snapshot stats, themes, timeline, reading list."""
        config = build_demo_corpus(tmp_path)
        build_digest(config, REF_DATE)

        digest = read_artifact(tmp_path, "research_digest.md")
        assert "Generated: 2024-03-01" in digest
        assert "- Videos indexed: 4" in digest
        assert "- With summaries: 3 (75.0%)" in digest
        assert "- Date range: 2024-02-05 — 2024-02-22" in digest
        assert "- Channels: 2" in digest
        assert "### Week 2024-W06" in digest
        assert "## Reading List" in digest
        assert "### production evals" in digest
        assert "**Evals Deep Dive** — Alpha_Channel, 2024-02-05" in digest
        assert "**Evals at Scale** — Alpha_Channel, 2024-02-20" in digest

    def test_second_run_emerging_diff(self, tmp_path: Path) -> None:
        """New vocabulary and newly qualifying phrases surface as emerging."""
        config = build_demo_corpus(tmp_path)
        build_digest(config, REF_DATE)
        add_video(
            tmp_path,
            "Beta_Channel",
            "Quantum Talk",
            "eeeeeeeeee1",
            "20240226",
            summary=freeform_summary("quantum entanglement and agent harness loops"),
        )
        build_digest(config, REF_DATE)

        digest = read_artifact(tmp_path, "research_digest.md")
        emerging = digest.split("## Emerging")[1].split("## Reading List")[0]
        assert "Previous run: 2024-03-01" in emerging
        assert "quantum" in emerging
        assert "agent harness" in emerging  # now meets document frequency 2

    def test_idless_download_indexed_end_to_end(self, tmp_path: Path) -> None:
        """An early download without a bracketed ID (Amendment 10) flows through.

        Its video_id comes from the metadata 'id' field and its summary joins
        by exact stem, even though a re-download of the same video exists with
        the bracketed token in its filenames.
        """
        config = build_demo_corpus(tmp_path)
        add_video(
            tmp_path,
            "Beta_Channel",
            "Agent Talk Original",
            "cccccccccc1",
            "20240210",
            summary=freeform_summary("agent harness loops"),
            id_in_filename=False,
        )
        build_digest(config, REF_DATE)

        records = json.loads(read_artifact(tmp_path, "corpus_index.json"))["records"]
        assert len(records) == 5
        idless = next(record for record in records if record["title"] == "Agent Talk Original")
        assert idless["video_id"] == "cccccccccc1"
        assert idless["has_summary"] is True
        assert idless["paths"]["summary_md"].endswith("Agent Talk Original.md")
        assert idless["paths"]["metadata_json"].endswith("Agent Talk Original.info.json")
        tokened = next(record for record in records if record["title"] == "Agent Talk")
        assert tokened["paths"]["summary_md"].endswith("Agent Talk [cccccccccc1].md")

    def test_steady_state_is_deterministic(self, tmp_path: Path) -> None:
        """Re-running on an unchanged corpus reproduces every artifact byte."""
        config = build_demo_corpus(tmp_path)
        build_digest(config, REF_DATE)
        build_digest(config, REF_DATE)
        artifact_names = [
            "corpus_index.json",
            "themes.json",
            "themes_report.md",
            "timeline.json",
            "timeline_report.md",
            "research_digest.md",
        ]
        second = {name: (analytics_dir(tmp_path) / name).read_bytes() for name in artifact_names}
        build_digest(config, REF_DATE)
        third = {name: (analytics_dir(tmp_path) / name).read_bytes() for name in artifact_names}
        assert second == third

    def test_duplicate_download_indexed_end_to_end(self, tmp_path: Path) -> None:
        """A video downloaded twice (same id, renamed) is indexed twice, not merged.

        Mirrors the real corpus (Amendment 8): the analytics consumes the fixed
        data as-is, counting each cleaned transcript faithfully through the full
        digest chain rather than crashing or collapsing the re-download. Each
        copy is paired to the metadata/summary whose filename stem matches its
        transcript.
        """
        config = build_demo_corpus(tmp_path)
        add_video(
            tmp_path,
            "Alpha_Channel",
            "Evals Deep Dive REUPLOAD",
            "aaaaaaaaaa1",
            "20240205",
            summary=template_shaped_summary("Production Evals", "production evals"),
        )
        build_digest(config, REF_DATE)

        records = json.loads(read_artifact(tmp_path, "corpus_index.json"))["records"]
        assert len(records) == 5
        dupes = [record for record in records if record["video_id"] == "aaaaaaaaaa1"]
        assert {record["title"] for record in dupes} == {"Evals Deep Dive", "Evals Deep Dive REUPLOAD"}
        for record in dupes:
            assert record["title"] in record["paths"]["cleaned_txt"]
            assert record["title"] in record["paths"]["metadata_json"]
            assert record["paths"]["summary_md"] is not None
            assert record["title"] in record["paths"]["summary_md"]
            assert record["summary_stats"]["word_count"] > 0


class TestErrorPathsEndToEnd:
    """Every documented fail-fast condition aborts the full chain."""

    def test_missing_metadata_aborts(self, tmp_path: Path) -> None:
        """A transcript without info.json stops the digest."""
        config = build_demo_corpus(tmp_path)
        info = tmp_path / "data" / "downloads" / "metadata" / "Alpha_Channel" / "video" / "Evals Deep Dive [aaaaaaaaaa1].info.json"
        info.rename(info.with_suffix(".hidden"))
        with pytest.raises(MetadataError, match="aaaaaaaaaa1"):
            build_digest(config, REF_DATE)

    def test_empty_summary_is_coverage_gap_end_to_end(self, tmp_path: Path) -> None:
        """A 0-byte/markup-only summary degrades to a coverage gap (Amendment 9)."""
        config = build_demo_corpus(tmp_path)
        summary = tmp_path / "data" / "downloads" / "transcripts_summaries" / "Alpha_Channel" / "Evals Deep Dive [aaaaaaaaaa1].md"
        summary.write_text("", encoding="utf-8")

        build_digest(config, REF_DATE)

        records = json.loads(read_artifact(tmp_path, "corpus_index.json"))["records"]
        emptied = next(record for record in records if record["title"] == "Evals Deep Dive")
        assert emptied["has_summary"] is False
        assert emptied["summary_stats"] is None
        assert emptied["paths"]["summary_md"] is None
        digest = read_artifact(tmp_path, "research_digest.md")
        assert "- With summaries: 2 (50.0%)" in digest

    def test_unmatched_channel_dir_aborts(self, tmp_path: Path) -> None:
        """A cleaned channel directory unknown to config stops the digest."""
        config = build_demo_corpus(tmp_path)
        stray = tmp_path / "data" / "downloads" / "transcripts_cleaned" / "Ghost_Channel"
        stray.mkdir(parents=True)
        (stray / "Stray [zzzzzzzzzz1].txt").write_text("words", encoding="utf-8")
        with pytest.raises(JoinError, match="Ghost_Channel"):
            build_digest(config, REF_DATE)

    def test_missing_video_id_aborts(self, tmp_path: Path) -> None:
        """A cleaned file without a bracketed ID stops the digest."""
        config = build_demo_corpus(tmp_path)
        cleaned = tmp_path / "data" / "downloads" / "transcripts_cleaned" / "Alpha_Channel"
        (cleaned / "No id at all.txt").write_text("words", encoding="utf-8")
        with pytest.raises(JoinError, match="No id at all.txt"):
            build_digest(config, REF_DATE)

    def test_malformed_metadata_json_aborts(self, tmp_path: Path) -> None:
        """Invalid JSON metadata stops the digest."""
        config = build_demo_corpus(tmp_path)
        info = tmp_path / "data" / "downloads" / "metadata" / "Alpha_Channel" / "video" / "Evals Deep Dive [aaaaaaaaaa1].info.json"
        info.write_text("{broken", encoding="utf-8")
        with pytest.raises(MetadataError, match="invalid JSON"):
            build_digest(config, REF_DATE)

    def test_empty_corpus_aborts(self, tmp_path: Path) -> None:
        """An empty cleaned corpus stops the digest."""
        config = make_config(tmp_path, ["Alpha Channel"])
        with pytest.raises(AnalyticsError, match="transcripts_cleaned"):
            build_digest(config, REF_DATE)


class TestAnalyticsScriptsCompile:
    """Thin CLI entrypoints stay syntactically valid."""

    def test_analytics_scripts_compile(self) -> None:
        """Catch syntax errors in the analytics script entrypoints."""
        project_root = Path(__file__).parent.parent
        for relative_path in (
            "scripts/analytics/index.py",
            "scripts/analytics/themes.py",
            "scripts/analytics/timeline.py",
            "scripts/analytics/digest.py",
        ):
            py_compile.compile(str(project_root / relative_path), doraise=True)
