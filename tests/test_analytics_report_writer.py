"""Unit tests for analytics artifact writing and markdown rendering."""

import json
from pathlib import Path

import pytest

from src.analytics.errors import AnalyticsError
from src.analytics.models import ThemeEntry, ThemeReport, ThemeVideo, TimelineBucket, TimelineReport
from src.analytics.report_writer import (
    render_themes_markdown,
    render_timeline_markdown,
    write_model_json,
    write_text_file,
)


def make_theme_report(
    term_themes: list[ThemeEntry] | None = None,
    phrase_themes: list[ThemeEntry] | None = None,
) -> ThemeReport:
    """Build a small ThemeReport for rendering tests."""
    return ThemeReport(
        lookback_days=30,
        channel_filter=None,
        video_count=2,
        term_themes=term_themes if term_themes is not None else [],
        phrase_themes=phrase_themes if phrase_themes is not None else [],
    )


def sample_theme_entry(term: str, score: float = 0.41) -> ThemeEntry:
    """Build one theme entry with a single contributing video."""
    return ThemeEntry(
        term=term,
        score=score,
        document_frequency=2,
        channels=["Alpha_Channel", "Mock_Channel"],
        videos=[
            ThemeVideo(
                video_id="aaaaaaaaaa1",
                channel="Mock_Channel",
                title="A Video",
                upload_date="2024-02-20",
                summary_md="/synthetic/summary.md",
            )
        ],
    )


class TestWriteModelJson:
    """JSON artifact writing."""

    def test_writes_json_with_parent_creation(self, tmp_path: Path) -> None:
        """Parent directories are created and JSON round-trips."""
        output_path = tmp_path / "nested" / "themes.json"
        written = write_model_json(make_theme_report(), output_path)
        assert written == output_path
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["lookback_days"] == 30

    def test_output_is_deterministic_with_trailing_newline(self, tmp_path: Path) -> None:
        """Writing the same model twice is byte-identical and newline-terminated."""
        output_path = tmp_path / "themes.json"
        first = write_model_json(make_theme_report(), output_path).read_bytes()
        second = write_model_json(make_theme_report(), output_path).read_bytes()
        assert first == second
        assert first.endswith(b"\n")

    def test_unwritable_target_raises_analytics_error(self, tmp_path: Path) -> None:
        """An unwritable existing target fails fast with the destination path."""
        output_path = tmp_path / "themes.json"
        output_path.mkdir()

        with pytest.raises(AnalyticsError, match="themes.json"):
            write_model_json(make_theme_report(), output_path)


class TestWriteTextFile:
    """Text artifact writing."""

    def test_writes_text_with_parent_creation(self, tmp_path: Path) -> None:
        """Parent directories are created and content is preserved."""
        output_path = tmp_path / "nested" / "report.md"
        written = write_text_file("# Report\n", output_path)
        assert written == output_path
        assert output_path.read_text(encoding="utf-8") == "# Report\n"

    def test_unwritable_target_raises_analytics_error(self, tmp_path: Path) -> None:
        """An unwritable existing target fails fast with the destination path."""
        output_path = tmp_path / "report.md"
        output_path.mkdir()

        with pytest.raises(AnalyticsError, match="report.md"):
            write_text_file("# Report\n", output_path)


class TestRenderThemesMarkdown:
    """Markdown rendering of the themes report."""

    def test_header_carries_filter_context(self) -> None:
        """Lookback, channel filter, and video count are rendered."""
        markdown = render_themes_markdown(make_theme_report())
        assert "# Themes Report" in markdown
        assert "- Lookback days: 30" in markdown
        assert "- Channel filter: all channels" in markdown
        assert "- Videos considered: 2" in markdown

    def test_term_rows_rendered_with_scores_and_video_counts(self) -> None:
        """Each term becomes a ranked row with score and document frequency."""
        report = make_theme_report(term_themes=[sample_theme_entry("kubernetes")])
        markdown = render_themes_markdown(report)
        assert "| 1 | kubernetes | 0.410000 | 2 |" in markdown

    def test_phrase_rows_rendered_with_doc_frequency_and_channels(self) -> None:
        """Each phrase becomes a ranked row with doc frequency and channels."""
        report = make_theme_report(phrase_themes=[sample_theme_entry("production evals")])
        markdown = render_themes_markdown(report)
        assert "| 1 | production evals | 2 | Alpha_Channel, Mock_Channel |" in markdown

    def test_pipes_in_phrases_escaped(self) -> None:
        """Pipe characters cannot break the markdown table."""
        report = make_theme_report(phrase_themes=[sample_theme_entry("rag | pipes")])
        markdown = render_themes_markdown(report)
        assert "rag \\| pipes" in markdown

    def test_empty_report_renders_placeholders(self) -> None:
        """Empty result sets render explicit placeholder lines."""
        markdown = render_themes_markdown(make_theme_report())
        assert "No terms extracted" in markdown
        assert "No phrases met the document-frequency threshold" in markdown

    def test_named_channel_filter_rendered(self) -> None:
        """A configured channel filter is echoed in the header."""
        report = ThemeReport(
            lookback_days=30,
            channel_filter="Mock Channel",
            video_count=1,
            term_themes=[],
            phrase_themes=[],
        )
        markdown = render_themes_markdown(report)
        assert "- Channel filter: Mock Channel" in markdown


class TestRenderTimelineMarkdown:
    """Markdown rendering of the timeline report."""

    def make_timeline_report(self, buckets: list[TimelineBucket]) -> TimelineReport:
        """Build a small TimelineReport."""
        return TimelineReport(
            lookback_days=60,
            channel_filter=None,
            bucket_type="week",
            video_count=sum(bucket.video_count for bucket in buckets),
            buckets=buckets,
        )

    def test_header_and_bucket_sections(self) -> None:
        """Buckets render as sections with counts, channels, terms, phrases."""
        bucket = TimelineBucket(
            bucket="2024-W08",
            video_count=2,
            channels={"Alpha_Channel": 1, "Mock_Channel": 1},
            top_terms=["kubernetes", "evals"],
            top_phrases=["production evals"],
        )
        markdown = render_timeline_markdown(self.make_timeline_report([bucket]))
        assert "# Timeline Report" in markdown
        assert "- Bucket type: week" in markdown
        assert "## 2024-W08" in markdown
        assert "- Videos: 2" in markdown
        assert "- Channels: Alpha_Channel (1), Mock_Channel (1)" in markdown
        assert "- Top terms: kubernetes, evals" in markdown
        assert "- Top phrases: production evals" in markdown

    def test_bucket_without_phrases_renders_none(self) -> None:
        """An empty phrase list renders an explicit 'none'."""
        bucket = TimelineBucket(
            bucket="2024-W08",
            video_count=1,
            channels={"Mock_Channel": 1},
            top_terms=["kubernetes"],
            top_phrases=[],
        )
        markdown = render_timeline_markdown(self.make_timeline_report([bucket]))
        assert "- Top phrases: none" in markdown

    def test_empty_timeline_renders_placeholder(self) -> None:
        """An empty window renders an explicit placeholder."""
        markdown = render_timeline_markdown(self.make_timeline_report([]))
        assert "No videos in the lookback window" in markdown
