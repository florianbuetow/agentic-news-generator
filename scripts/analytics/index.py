#!/usr/bin/env python3
"""Build the analytics corpus index from cleaned transcripts, summaries, and metadata."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.analytics.index_builder import build_index, write_corpus_index
from src.config import Config


def main() -> int:
    """Build and write corpus_index.json; exceptions propagate (fail fast)."""
    config = Config(Path("config/config.yaml"))
    output_dir = config.get_data_output_analytics_dir()
    index = build_index(config)
    write_corpus_index(index, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
