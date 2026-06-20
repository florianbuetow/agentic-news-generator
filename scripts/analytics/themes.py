#!/usr/bin/env python3
"""Rank themes (TF-IDF terms and phrases) over the summarized corpus."""

from datetime import date

from src.analytics.index_builder import build_index
from src.analytics.report_writer import render_themes_markdown, write_model_json, write_text_file
from src.analytics.theme_ranker import rank_themes
from src.config import Config


def main() -> int:
    """Build the index in-process, rank themes, and write the artifacts."""
    config = Config.load_default()
    analytics_config = config.get_analytics_config()
    output_dir = config.get_data_output_analytics_dir()
    index = build_index(config)
    report = rank_themes(index, analytics_config, date.today())
    write_model_json(report, output_dir / "themes.json")
    write_text_file(render_themes_markdown(report), output_dir / "themes_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
