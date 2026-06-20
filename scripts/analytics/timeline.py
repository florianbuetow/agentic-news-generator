#!/usr/bin/env python3
"""Build the upload-date timeline report over the summarized corpus."""

from datetime import date

from src.analytics.index_builder import build_index
from src.analytics.report_writer import render_timeline_markdown, write_model_json, write_text_file
from src.analytics.timeline_builder import build_timeline
from src.config import Config


def main() -> int:
    """Build the index in-process, bucket the timeline, write the artifacts."""
    config = Config.load_default()
    analytics_config = config.get_analytics_config()
    output_dir = config.get_data_output_analytics_dir()
    index = build_index(config)
    report = build_timeline(index, analytics_config, date.today())
    write_model_json(report, output_dir / "timeline.json")
    write_text_file(render_timeline_markdown(report), output_dir / "timeline_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
