"""Unit tests for TF-IDF theme ranking over a synthetic corpus index.

Records reference real files in tmp_path because the ranker re-reads each
summary from ``paths.summary_md`` as opaque text (Amendment 7); nothing is
parsed from the summary layout.
"""

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.analytics.errors import AnalyticsError, EmptySummaryError, MetadataError
from src.analytics.models import ChannelMeta, CorpusIndex, CorpusRecord, RecordPaths, SummaryStats
from src.analytics.text_normalize import normalize_markdown, word_count
from src.analytics.theme_ranker import filter_candidate_records, rank_themes
from src.config import AnalyticsConfig

REF_DATE = date(2024, 3, 1)


def make_analytics_config(**overrides: Any) -> AnalyticsConfig:
    """Build an AnalyticsConfig with test-friendly values."""
    values: dict[str, Any] = {
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
        "previous_run_cache": "/tmp/unused_cache.json",
    }
    values.update(overrides)
    return AnalyticsConfig.model_validate(values)


def make_record(
    tmp_path: Path,
    video_id: str,
    channel: str = "Mock_Channel",
    upload_date: str | None = "2024-02-20",
    title: str = "A Video",
    summary_text: str | None = "generic summary text",
    txt_text: str = "transcript filler words",
) -> CorpusRecord:
    """Build a synthetic corpus record backed by real files in tmp_path."""
    channel_dir = tmp_path / channel
    channel_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{title} [{video_id}]"
    txt_path = channel_dir / f"{stem}.txt"
    txt_path.write_text(txt_text, encoding="utf-8")
    summary_path: Path | None = None
    summary_stats: SummaryStats | None = None
    if summary_text is not None:
        summary_path = channel_dir / f"{stem}.md"
        summary_path.write_text(summary_text, encoding="utf-8")
        normalized = normalize_markdown(summary_text)
        summary_stats = SummaryStats(word_count=word_count(normalized), char_count=len(normalized))
    return CorpusRecord(
        video_id=video_id,
        channel=channel,
        title=title,
        upload_date=upload_date,
        duration_seconds=600,
        word_count=word_count(txt_text),
        has_summary=summary_text is not None,
        paths=RecordPaths(
            cleaned_txt=str(txt_path),
            cleaned_srt=None,
            summary_md=str(summary_path) if summary_path is not None else None,
            metadata_json=str(channel_dir / f"{stem}.info.json"),
        ),
        channel_meta=ChannelMeta(language="en", category="synthetic", description="Synthetic channel"),
        summary_stats=summary_stats,
    )


class TestRecordFiltering:
    """Candidate selection: summaries only, lookback window, channel filter."""

    def test_records_without_summary_excluded(self, tmp_path: Path) -> None:
        """Only summarized videos are theme candidates."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1"),
                make_record(tmp_path, "bbbbbbbbbb1", summary_text=None),
            ]
        )
        candidates = filter_candidate_records(index, make_analytics_config(), REF_DATE)
        assert [r.video_id for r in candidates] == ["aaaaaaaaaa1"]

    def test_lookback_excludes_old_uploads(self, tmp_path: Path) -> None:
        """Uploads older than the lookback window are excluded."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", upload_date="2024-02-20"),
                make_record(tmp_path, "bbbbbbbbbb1", upload_date="2024-01-01"),
            ]
        )
        candidates = filter_candidate_records(index, make_analytics_config(lookback_days=30), REF_DATE)
        assert [r.video_id for r in candidates] == ["aaaaaaaaaa1"]

    def test_channel_filter_accepts_config_channel_name(self, tmp_path: Path) -> None:
        """channel_filter is a channels[].name value, matched after sanitization."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", channel="Mock_Channel"),
                make_record(tmp_path, "bbbbbbbbbb1", channel="Other_Channel"),
            ]
        )
        config = make_analytics_config(channel_filter="Mock Channel")
        candidates = filter_candidate_records(index, config, REF_DATE)
        assert [r.video_id for r in candidates] == ["aaaaaaaaaa1"]

    def test_unknown_channel_filter_raises(self, tmp_path: Path) -> None:
        """A channel_filter matching nothing in the corpus is a config error."""
        index = CorpusIndex(records=[make_record(tmp_path, "aaaaaaaaaa1")])
        config = make_analytics_config(channel_filter="Ghost Channel")
        with pytest.raises(AnalyticsError, match="Ghost Channel"):
            filter_candidate_records(index, config, REF_DATE)

    def test_candidate_without_upload_date_raises(self, tmp_path: Path) -> None:
        """Candidates need upload_date for lookback filtering (fail fast)."""
        index = CorpusIndex(records=[make_record(tmp_path, "aaaaaaaaaa1", upload_date=None)])
        with pytest.raises(MetadataError, match="aaaaaaaaaa1"):
            filter_candidate_records(index, make_analytics_config(), REF_DATE)


class TestTermThemes:
    """TF-IDF single-word term ranking over normalized summary text."""

    def test_distinctive_term_ranks(self, tmp_path: Path) -> None:
        """A term concentrated in the corpus appears among top terms."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", summary_text="Kubernetes operators reconcile cluster state"),
                make_record(tmp_path, "bbbbbbbbbb1", summary_text="Kubernetes networking with service meshes"),
            ]
        )
        report = rank_themes(index, make_analytics_config(), REF_DATE)
        assert "kubernetes" in [t.term for t in report.term_themes]

    def test_terms_capped_and_scores_descending(self, tmp_path: Path) -> None:
        """Terms are capped at top_n_terms and sorted by descending score."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", summary_text="alpha beta gamma delta epsilon words"),
                make_record(tmp_path, "bbbbbbbbbb1", summary_text="alpha beta gamma words appear again"),
            ]
        )
        report = rank_themes(index, make_analytics_config(top_n_terms=3), REF_DATE)
        assert len(report.term_themes) == 3
        scores = [t.score for t in report.term_themes]
        assert scores == sorted(scores, reverse=True)

    def test_document_frequency_counted_per_video(self, tmp_path: Path) -> None:
        """document_frequency counts contributing videos, not occurrences."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", summary_text="kubernetes kubernetes kubernetes"),
                make_record(tmp_path, "bbbbbbbbbb1", summary_text="kubernetes scheduling basics"),
                make_record(tmp_path, "cccccccccc1", summary_text="unrelated content entirely"),
            ]
        )
        report = rank_themes(index, make_analytics_config(), REF_DATE)
        kubernetes = next(t for t in report.term_themes if t.term == "kubernetes")
        assert kubernetes.document_frequency == 2

    def test_contributing_videos_ranked_by_weight_and_capped(self, tmp_path: Path) -> None:
        """Videos attach by descending TF-IDF weight, capped per theme."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", summary_text="kubernetes kubernetes kubernetes"),
                make_record(tmp_path, "bbbbbbbbbb1", summary_text="kubernetes scheduling matters here today"),
            ]
        )
        config = make_analytics_config(tfidf_ngram_range_max=1, top_n_videos_per_theme=1)
        report = rank_themes(index, config, REF_DATE)
        kubernetes = next(t for t in report.term_themes if t.term == "kubernetes")
        assert kubernetes.document_frequency == 2
        assert [v.video_id for v in kubernetes.videos] == ["aaaaaaaaaa1"]

    def test_channels_listed_sorted_unique(self, tmp_path: Path) -> None:
        """All contributing channels are listed once, sorted."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", channel="Zeta_Channel", summary_text="kubernetes basics"),
                make_record(tmp_path, "bbbbbbbbbb1", channel="Alpha_Channel", summary_text="kubernetes basics"),
            ]
        )
        report = rank_themes(index, make_analytics_config(), REF_DATE)
        kubernetes = next(t for t in report.term_themes if t.term == "kubernetes")
        assert kubernetes.channels == ["Alpha_Channel", "Zeta_Channel"]

    def test_stop_word_only_summaries_yield_no_themes(self, tmp_path: Path) -> None:
        """When no vocabulary survives stop-word removal, theme lists are empty."""
        index = CorpusIndex(records=[make_record(tmp_path, "aaaaaaaaaa1", summary_text="the of and is are")])
        report = rank_themes(index, make_analytics_config(), REF_DATE)
        assert report.term_themes == []
        assert report.phrase_themes == []

    def test_empty_window_yields_empty_report(self, tmp_path: Path) -> None:
        """No candidates in the window is a valid empty result, not an error."""
        index = CorpusIndex(records=[make_record(tmp_path, "aaaaaaaaaa1", upload_date="2020-01-01")])
        report = rank_themes(index, make_analytics_config(), REF_DATE)
        assert report.video_count == 0
        assert report.term_themes == []
        assert report.phrase_themes == []

    def test_summary_empty_at_ranking_time_raises(self, tmp_path: Path) -> None:
        """A summary that is empty when re-read aborts the ranking."""
        record = make_record(tmp_path, "aaaaaaaaaa1")
        assert record.paths.summary_md is not None
        Path(record.paths.summary_md).write_text("", encoding="utf-8")
        with pytest.raises(EmptySummaryError, match="aaaaaaaaaa1"):
            rank_themes(CorpusIndex(records=[record]), make_analytics_config(), REF_DATE)


class TestPhraseThemes:
    """Multi-word phrase ranking gated by document frequency."""

    def test_phrase_meets_document_frequency(self, tmp_path: Path) -> None:
        """A phrase shared by enough videos ranks; rarer phrases do not."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", summary_text="production evals ship quality"),
                make_record(tmp_path, "bbbbbbbbbb1", summary_text="production evals gate deployments"),
                make_record(tmp_path, "cccccccccc1", summary_text="agent harness design"),
            ]
        )
        report = rank_themes(index, make_analytics_config(min_theme_document_frequency=2), REF_DATE)
        phrases = [t.term for t in report.phrase_themes]
        assert "production evals" in phrases
        assert "agent harness" not in phrases
        production_evals = next(t for t in report.phrase_themes if t.term == "production evals")
        assert production_evals.document_frequency == 2

    def test_phrases_ordered_by_document_frequency(self, tmp_path: Path) -> None:
        """Phrases sort by descending document frequency first."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", summary_text="agent harness with production evals"),
                make_record(tmp_path, "bbbbbbbbbb1", summary_text="agent harness with production evals"),
                make_record(tmp_path, "cccccccccc1", summary_text="agent harness design"),
            ]
        )
        report = rank_themes(index, make_analytics_config(min_theme_document_frequency=2), REF_DATE)
        agent_rank = [t.term for t in report.phrase_themes].index("agent harness")
        evals_rank = [t.term for t in report.phrase_themes].index("production evals")
        assert agent_rank < evals_rank

    def test_phrases_capped_at_top_n_themes(self, tmp_path: Path) -> None:
        """At most top_n_themes phrases are reported."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", summary_text="production evals and agent harness"),
                make_record(tmp_path, "bbbbbbbbbb1", summary_text="production evals and agent harness"),
            ]
        )
        report = rank_themes(index, make_analytics_config(min_theme_document_frequency=2, top_n_themes=1), REF_DATE)
        assert len(report.phrase_themes) == 1

    def test_unigram_range_disables_phrases(self, tmp_path: Path) -> None:
        """With a (1, 1) n-gram range there are no multi-word features."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", summary_text="production evals everywhere"),
                make_record(tmp_path, "bbbbbbbbbb1", summary_text="production evals everywhere"),
            ]
        )
        config = make_analytics_config(tfidf_ngram_range_min=1, tfidf_ngram_range_max=1)
        report = rank_themes(index, config, REF_DATE)
        assert report.phrase_themes == []
        assert "production" in [t.term for t in report.term_themes]


class TestCleanedTxtSignal:
    """Optional cleaned-transcript contribution to TF-IDF documents."""

    def test_cleaned_txt_excluded_by_default(self, tmp_path: Path) -> None:
        """Transcript-only vocabulary stays out when the knob is off."""
        index = CorpusIndex(records=[make_record(tmp_path, "aaaaaaaaaa1", summary_text="summary words", txt_text="zirconium reactors")])
        report = rank_themes(index, make_analytics_config(), REF_DATE)
        assert "zirconium" not in [t.term for t in report.term_themes]

    def test_cleaned_txt_included_when_configured(self, tmp_path: Path) -> None:
        """Transcript vocabulary joins the document when the knob is on."""
        index = CorpusIndex(records=[make_record(tmp_path, "aaaaaaaaaa1", summary_text="summary words", txt_text="zirconium reactors")])
        report = rank_themes(index, make_analytics_config(include_cleaned_txt_in_tfidf=True), REF_DATE)
        assert "zirconium" in [t.term for t in report.term_themes]


class TestReportMetadata:
    """The report echoes its filter parameters."""

    def test_report_carries_filter_context(self, tmp_path: Path) -> None:
        """lookback_days, channel_filter, and video_count are recorded."""
        index = CorpusIndex(records=[make_record(tmp_path, "aaaaaaaaaa1")])
        report = rank_themes(index, make_analytics_config(lookback_days=30), REF_DATE)
        assert report.lookback_days == 30
        assert report.channel_filter is None
        assert report.video_count == 1
