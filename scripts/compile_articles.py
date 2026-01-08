#!/usr/bin/env python3
"""Compile markdown articles into articles.js."""

import json
import logging
import sys
from pathlib import Path

from src.config import Config
from src.processing.article_compiler import ArticleCompiler
from src.processing.article_parser import ArticleParser

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_javascript_output(compiled_data: dict) -> str:
    """Generate JavaScript ES6 module from compiled data."""
    template = """// Content data for The Artificial Intelligence Times
// Edit this file to update articles on the homepage

export const heroArticle = {hero}

export const featuredArticles = {featured}

export const secondaryMain = {secondary}

export const sidebarArticles = {sidebar}

export const briefsColumns = {briefs}
"""
    return template.format(
        hero=json.dumps(compiled_data["hero"], indent=2),
        featured=json.dumps(compiled_data["featured"], indent=2),
        secondary=json.dumps(compiled_data["secondary"], indent=2),
        sidebar=json.dumps(compiled_data["sidebar"], indent=2),
        briefs=json.dumps(compiled_data["briefs"], indent=2),
    )


def main() -> None:
    """Main entry point for article compilation."""
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        config = Config(config_path)
        compiler_config = config.get_article_compiler_config()

        logger.info("Starting article compilation")

        # Parse articles
        input_dir = Path(compiler_config.input_dir)
        parser = ArticleParser(compiler_config)
        articles = parser.parse_directory(input_dir)

        if not articles:
            logger.error("No articles parsed successfully")
            sys.exit(1)

        # Compile articles
        compiler = ArticleCompiler(compiler_config)
        compiled_data = compiler.compile(articles)

        # Generate JavaScript output
        js_output = generate_javascript_output(compiled_data)

        # Write output file
        output_path = Path(compiler_config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(js_output, encoding="utf-8")

        logger.info(f"Successfully compiled {len(articles)} articles to {output_path}")

    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
