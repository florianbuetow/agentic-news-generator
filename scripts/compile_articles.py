#!/usr/bin/env python3
"""Validate and stage newspaper articles for Nuxt."""

import logging
import sys
from pathlib import Path

from src.config import Config
from src.processing.newspaper_articles import NewspaperArticleCompilationError, compile_newspaper_articles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run newspaper article compilation."""
    config = Config.load_default()
    source_dir = config.get_data_input_dir() / "newspaper" / "articles"
    project_root = Path(__file__).resolve().parent.parent

    try:
        result = compile_newspaper_articles(source_dir=source_dir, project_root=project_root)
    except NewspaperArticleCompilationError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Generated %s Nuxt article files in %s", len(result.selected_slugs), result.article_output_dir)
    logger.info("Generated layout: %s", result.layout_output_path)


if __name__ == "__main__":
    main()
