"""Writers and markdown renderers for analytics artifacts.

All artifacts are deterministic: the same inputs produce byte-identical
output. JSON mirrors the pydantic models; markdown is built from stable,
sorted report data.
"""

from datetime import date
from pathlib import Path

from pydantic import BaseModel

from src.analytics.errors import AnalyticsError
from src.analytics.models import CorpusIndex, EmergingDiff, ThemeReport, TimelineReport

SCORE_FORMAT: str = "{:.6f}"


def write_model_json(model: BaseModel, output_path: Path) -> Path:
    """Serialize a pydantic model as indented JSON.

    Args:
        model: Any analytics report model.
        output_path: Destination file; parent directories are created.

    Returns:
        The written path.

    Raises:
        AnalyticsError: If the parent directory cannot be created or the file
            cannot be written.
    """
    return _write_artifact(model.model_dump_json(indent=2) + "\n", output_path)


def write_text_file(content: str, output_path: Path) -> Path:
    """Write a text artifact (markdown report).

    Args:
        content: Full file content.
        output_path: Destination file; parent directories are created.

    Returns:
        The written path.

    Raises:
        AnalyticsError: If the parent directory cannot be created or the file
            cannot be written.
    """
    return _write_artifact(content, output_path)


def _write_artifact(content: str, output_path: Path) -> Path:
    """Write artifact content, failing fast with path context on I/O errors."""
    _ensure_parent_dir(output_path)
    try:
        output_path.write_text(content, encoding="utf-8")
    except OSError as e:
        raise AnalyticsError(f"Cannot write analytics artifact: {output_path}: {e}") from e
    print(f"Wrote {output_path}", flush=True)
    return output_path


def render_themes_markdown(report: ThemeReport) -> str:
    """Render the themes report as a human-readable markdown document."""
    lines: list[str] = [
        "# Themes Report",
        "",
        f"- Lookback days: {report.lookback_days}",
        f"- Channel filter: {_channel_filter_label(report.channel_filter)}",
        f"- Videos considered: {report.video_count}",
        "",
        "## Top terms (TF-IDF)",
        "",
        *_term_table_lines(report),
        "",
        "## Top phrases",
        "",
        *_phrase_table_lines(report),
        "",
    ]
    return "\n".join(lines)


def _term_table_lines(report: ThemeReport) -> list[str]:
    """Markdown table (or placeholder) for ranked single-word terms."""
    if not report.term_themes:
        return ["_No terms extracted._"]
    lines = ["| Rank | Term | Score | Videos |", "|------|------|-------|--------|"]
    lines.extend(
        f"| {rank} | {escape_table_cell(entry.term)} | {SCORE_FORMAT.format(entry.score)} | {entry.document_frequency} |"
        for rank, entry in enumerate(report.term_themes, start=1)
    )
    return lines


def _phrase_table_lines(report: ThemeReport) -> list[str]:
    """Markdown table (or placeholder) for ranked multi-word phrases."""
    if not report.phrase_themes:
        return ["_No phrases met the document-frequency threshold._"]
    lines = ["| Rank | Phrase | Doc frequency | Top channels |", "|------|--------|---------------|--------------|"]
    lines.extend(
        f"| {rank} | {escape_table_cell(entry.term)} | {entry.document_frequency} | {escape_table_cell(', '.join(entry.channels))} |"
        for rank, entry in enumerate(report.phrase_themes, start=1)
    )
    return lines


def render_digest_markdown(
    index: CorpusIndex,
    themes: ThemeReport,
    timeline: TimelineReport,
    emerging: EmergingDiff,
    reference_date: date,
) -> str:
    """Render the research digest combining corpus, themes, timeline, and diff."""
    records = index.records
    with_summary = sum(1 for record in records if record.has_summary)
    coverage_pct = with_summary / len(records) * 100
    upload_dates = sorted(record.upload_date for record in records if record.upload_date is not None)
    date_range = "unknown"
    if upload_dates:
        date_range = f"{upload_dates[0]} — {upload_dates[-1]}"

    lines: list[str] = [
        "# Research Digest",
        "",
        f"Generated: {reference_date.isoformat()}",
        "",
        "## Corpus Snapshot",
        "",
        f"- Videos indexed: {len(records)}",
        f"- With summaries: {with_summary} ({coverage_pct:.1f}%)",
        f"- Date range: {date_range}",
        f"- Channels: {len({record.channel for record in records})}",
        "",
        f"## Top Themes (last {themes.lookback_days} days)",
        "",
        f"- Channel filter: {_channel_filter_label(themes.channel_filter)}",
        f"- Videos considered: {themes.video_count}",
        "",
        "### Top terms (TF-IDF)",
        "",
        *_term_table_lines(themes),
        "",
        "### Top phrases",
        "",
        *_phrase_table_lines(themes),
        "",
        "## Timeline",
        "",
    ]
    if timeline.buckets:
        for bucket in timeline.buckets:
            lines.extend(
                [
                    f"### {timeline.bucket_type.capitalize()} {bucket.bucket}",
                    "",
                    f"- Videos: {bucket.video_count}",
                    f"- Top terms: {_listing_label(bucket.top_terms)}",
                    "",
                ]
            )
    else:
        lines.extend(["_No videos in the lookback window._", ""])

    lines.extend(["## Emerging (vs previous run)", ""])
    if emerging.previous_generated_on is None:
        lines.extend(["_First run — no previous snapshot._", ""])
    else:
        lines.extend(
            [
                f"- Previous run: {emerging.previous_generated_on}",
                f"- New terms: {_listing_label(emerging.new_terms)}",
                f"- New phrases: {_listing_label(emerging.new_phrases)}",
                "",
            ]
        )

    lines.extend(["## Reading List", ""])
    if themes.phrase_themes:
        for entry in themes.phrase_themes:
            lines.extend([f"### {entry.term}", ""])
            lines.extend(
                f"{position}. **{video.title}** — {video.channel}, {_date_label(video.upload_date)} — `{video.summary_md}`"
                for position, video in enumerate(entry.videos, start=1)
            )
            lines.append("")
    else:
        lines.extend(["_No phrases met the document-frequency threshold._", ""])
    return "\n".join(lines)


def render_timeline_markdown(report: TimelineReport) -> str:
    """Render the timeline report as a human-readable markdown document."""
    lines: list[str] = [
        "# Timeline Report",
        "",
        f"- Lookback days: {report.lookback_days}",
        f"- Channel filter: {_channel_filter_label(report.channel_filter)}",
        f"- Bucket type: {report.bucket_type}",
        f"- Videos considered: {report.video_count}",
        "",
    ]
    if not report.buckets:
        lines.extend(["_No videos in the lookback window._", ""])
        return "\n".join(lines)
    for bucket in report.buckets:
        channels = ", ".join(f"{channel} ({count})" for channel, count in bucket.channels.items())
        lines.extend(
            [
                f"## {bucket.bucket}",
                "",
                f"- Videos: {bucket.video_count}",
                f"- Channels: {channels}",
                f"- Top terms: {_listing_label(bucket.top_terms)}",
                f"- Top phrases: {_listing_label(bucket.top_phrases)}",
                "",
            ]
        )
    return "\n".join(lines)


def escape_table_cell(value: str) -> str:
    """Escape pipe characters so values cannot break markdown tables."""
    return value.replace("|", "\\|")


def _listing_label(values: list[str]) -> str:
    """Comma listing of values, or an explicit 'none'."""
    if not values:
        return "none"
    return ", ".join(values)


def _date_label(value: str | None) -> str:
    """Display label for an optional ISO date."""
    if value is None:
        return "unknown date"
    return value


def _channel_filter_label(channel_filter: str | None) -> str:
    """Human-readable channel filter description."""
    if channel_filter is None:
        return "all channels"
    return channel_filter


def _ensure_parent_dir(output_path: Path) -> None:
    """Create the artifact's parent directory, failing fast on errors."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise AnalyticsError(f"Cannot create analytics output directory: {output_path.parent}: {e}") from e
