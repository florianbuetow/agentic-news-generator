"""Unit tests for the timeline builder over a synthetic corpus index.

Records reference real summary files in tmp_path because per-bucket term and
phrase ranking re-reads each summary from disk as opaque text (Amendment 7).
"""

from datetime import date
from pathlib import Path
from typing import Any

from src.analytics.models import ChannelMeta, CorpusIndex, CorpusRecord, RecordPaths, SummaryStats
from src.analytics.text_normalize import normalize_markdown, word_count
from src.analytics.timeline_builder import build_timeline
from src.config import AnalyticsConfig

REF_DATE = date(2024, 3, 1)


def make_analytics_config(**overrides: Any) -> AnalyticsConfig:
    """Build an AnalyticsConfig with test-friendly values."""
    values: dict[str, Any] = {
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
        "previous_run_cache": "/tmp/unused_cache.json",
    }
    values.update(overrides)
    return AnalyticsConfig.model_validate(values)


def make_record(
    tmp_path: Path,
    video_id: str,
    upload_date: str,
    channel: str = "Mock_Channel",
    summary_text: str = "generic summary text",
) -> CorpusRecord:
    """Build a synthetic summarized corpus record backed by real files."""
    channel_dir = tmp_path / channel
    channel_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Video {video_id} [{video_id}]"
    txt_path = channel_dir / f"{stem}.txt"
    txt_path.write_text("transcript filler words", encoding="utf-8")
    summary_path = channel_dir / f"{stem}.md"
    summary_path.write_text(summary_text, encoding="utf-8")
    normalized = normalize_markdown(summary_text)
    return CorpusRecord(
        video_id=video_id,
        channel=channel,
        title=f"Video {video_id}",
        upload_date=upload_date,
        duration_seconds=600,
        word_count=3,
        has_summary=True,
        paths=RecordPaths(
            cleaned_txt=str(txt_path),
            cleaned_srt=None,
            summary_md=str(summary_path),
            metadata_json=str(channel_dir / f"{stem}.info.json"),
        ),
        channel_meta=ChannelMeta(language="en", category="synthetic", description="Synthetic channel"),
        summary_stats=SummaryStats(word_count=word_count(normalized), char_count=len(normalized)),
    )


class TestBucketAssignment:
    """Grouping records into ISO-week and calendar-month buckets."""

    def test_week_buckets_chronological(self, tmp_path: Path) -> None:
        """Records land in their ISO weeks, oldest bucket first."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", "2024-02-20"),
                make_record(tmp_path, "bbbbbbbbbb1", "2024-02-05"),
            ]
        )
        report = build_timeline(index, make_analytics_config(), REF_DATE)
        assert [b.bucket for b in report.buckets] == ["2024-W06", "2024-W08"]

    def test_month_buckets(self, tmp_path: Path) -> None:
        """Month bucketing groups by calendar month."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", "2024-01-31"),
                make_record(tmp_path, "bbbbbbbbbb1", "2024-02-01"),
            ]
        )
        report = build_timeline(index, make_analytics_config(timeline_bucket="month"), REF_DATE)
        assert [b.bucket for b in report.buckets] == ["2024-01", "2024-02"]

    def test_iso_year_boundary_week(self, tmp_path: Path) -> None:
        """Dec 30 2024 belongs to ISO week 2025-W01, not 2024."""
        index = CorpusIndex(records=[make_record(tmp_path, "aaaaaaaaaa1", "2024-12-30")])
        config = make_analytics_config(lookback_days=365)
        report = build_timeline(index, config, date(2025, 1, 15))
        assert [b.bucket for b in report.buckets] == ["2025-W01"]

    def test_lookback_excludes_old_uploads(self, tmp_path: Path) -> None:
        """Uploads outside the lookback window do not appear."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", "2024-02-20"),
                make_record(tmp_path, "bbbbbbbbbb1", "2023-06-01"),
            ]
        )
        report = build_timeline(index, make_analytics_config(lookback_days=60), REF_DATE)
        assert report.video_count == 1
        assert len(report.buckets) == 1


class TestBucketContents:
    """Per-bucket statistics."""

    def test_channel_breakdown_sorted(self, tmp_path: Path) -> None:
        """Each bucket counts videos per channel, channels sorted."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", "2024-02-20", channel="Zeta_Channel"),
                make_record(tmp_path, "bbbbbbbbbb1", "2024-02-21", channel="Alpha_Channel"),
                make_record(tmp_path, "cccccccccc1", "2024-02-22", channel="Alpha_Channel"),
            ]
        )
        report = build_timeline(index, make_analytics_config(), REF_DATE)
        assert report.buckets[0].channels == {"Alpha_Channel": 2, "Zeta_Channel": 1}

    def test_top_terms_local_to_bucket(self, tmp_path: Path) -> None:
        """TF-IDF terms are computed per bucket, not globally."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", "2024-02-05", summary_text="kubernetes operators reconcile clusters"),
                make_record(tmp_path, "bbbbbbbbbb1", "2024-02-20", summary_text="evaluation pipelines gate deployments"),
            ]
        )
        report = build_timeline(index, make_analytics_config(), REF_DATE)
        assert "kubernetes" in report.buckets[0].top_terms
        assert "kubernetes" not in report.buckets[1].top_terms

    def test_top_phrases_local_to_bucket(self, tmp_path: Path) -> None:
        """Phrases meeting the document-frequency threshold rank per bucket."""
        index = CorpusIndex(
            records=[
                make_record(tmp_path, "aaaaaaaaaa1", "2024-02-20", summary_text="production evals ship quality"),
                make_record(tmp_path, "bbbbbbbbbb1", "2024-02-21", summary_text="production evals gate deployments"),
            ]
        )
        report = build_timeline(index, make_analytics_config(min_theme_document_frequency=2), REF_DATE)
        assert "production evals" in report.buckets[0].top_phrases

    def test_empty_window_yields_empty_report(self, tmp_path: Path) -> None:
        """No candidates produce an empty bucket list, not an error."""
        index = CorpusIndex(records=[make_record(tmp_path, "aaaaaaaaaa1", "2020-01-01")])
        report = build_timeline(index, make_analytics_config(), REF_DATE)
        assert report.video_count == 0
        assert report.buckets == []


class TestReportMetadata:
    """The report echoes its parameters."""

    def test_report_carries_context(self, tmp_path: Path) -> None:
        """lookback, channel filter, bucket type, and counts are recorded."""
        index = CorpusIndex(records=[make_record(tmp_path, "aaaaaaaaaa1", "2024-02-20")])
        report = build_timeline(index, make_analytics_config(), REF_DATE)
        assert report.lookback_days == 60
        assert report.channel_filter is None
        assert report.bucket_type == "week"
        assert report.video_count == 1
