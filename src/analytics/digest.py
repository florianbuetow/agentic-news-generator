"""Research digest orchestration: index → themes → timeline → digest + cache.

One call produces every analytics artifact and updates the run-over-run
snapshot used to surface emerging terms and phrases. The snapshot cache is
the only mutable state between runs; a corrupt cache fails fast (delete the
file to reset).
"""

import json
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from src.analytics.errors import AnalyticsError
from src.analytics.index_builder import build_index, write_corpus_index
from src.analytics.models import AnalyticsSnapshot, EmergingDiff, ThemeReport
from src.analytics.report_writer import (
    render_digest_markdown,
    render_themes_markdown,
    render_timeline_markdown,
    write_model_json,
    write_text_file,
)
from src.analytics.theme_ranker import rank_themes
from src.analytics.timeline_builder import build_timeline
from src.config import Config


def build_digest(config: Config, reference_date: date) -> None:
    """Run the full analytics chain and write all artifacts.

    Args:
        config: Loaded project configuration.
        reference_date: Date anchoring lookback windows and the digest header.
    """
    analytics_config = config.get_analytics_config()
    output_dir = config.get_data_output_analytics_dir()

    # Validate the cache before any artifact write: a corrupt cache must not
    # leave fresh subreports mixed with a stale research_digest.md.
    previous = _load_previous_snapshot(analytics_config.previous_run_cache)

    index = build_index(config)
    write_corpus_index(index, output_dir)

    themes = rank_themes(index, analytics_config, reference_date)
    write_model_json(themes, output_dir / "themes.json")
    write_text_file(render_themes_markdown(themes), output_dir / "themes_report.md")

    timeline = build_timeline(index, analytics_config, reference_date)
    write_model_json(timeline, output_dir / "timeline.json")
    write_text_file(render_timeline_markdown(timeline), output_dir / "timeline_report.md")

    emerging = _compute_emerging(themes, previous)
    write_text_file(
        render_digest_markdown(index, themes, timeline, emerging, reference_date),
        output_dir / "research_digest.md",
    )

    snapshot = AnalyticsSnapshot(
        generated_on=reference_date.isoformat(),
        top_terms=[entry.term for entry in themes.term_themes],
        top_phrases=[entry.term for entry in themes.phrase_themes],
    )
    write_model_json(snapshot, analytics_config.previous_run_cache)


def _load_previous_snapshot(cache_path: Path) -> AnalyticsSnapshot | None:
    """Load the previous run snapshot; absence means a first run."""
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise AnalyticsError(f"Analytics snapshot cache is unreadable: {cache_path}: {e} (delete the file to reset)") from e
    try:
        return AnalyticsSnapshot.model_validate(payload)
    except ValidationError as e:
        raise AnalyticsError(f"Analytics snapshot cache has an invalid structure: {cache_path} (delete the file to reset)") from e


def _compute_emerging(themes: ThemeReport, previous: AnalyticsSnapshot | None) -> EmergingDiff:
    """Diff the current top results against the previous snapshot."""
    if previous is None:
        return EmergingDiff(new_terms=[], new_phrases=[], previous_generated_on=None)
    return EmergingDiff(
        new_terms=sorted({entry.term for entry in themes.term_themes} - set(previous.top_terms)),
        new_phrases=sorted({entry.term for entry in themes.phrase_themes} - set(previous.top_phrases)),
        previous_generated_on=previous.generated_on,
    )
