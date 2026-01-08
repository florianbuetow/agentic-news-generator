#!/usr/bin/env python3
"""Preprocess markdown articles by extracting only YAML frontmatter.

This script reads markdown files from {data_input_dir}/newspaper/articles/ and
creates processed versions in frontend/newspaper/content/articles/ containing
only the YAML frontmatter (up to and including the second '---' delimiter).
All markdown content after the frontmatter is removed.
"""

import logging
import sys
from pathlib import Path

# Add src to path to import Config
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def preprocess_markdown(content: str) -> str:
    """Remove H1, H2, and separator line after YAML frontmatter.

    The file structure is:
    1. YAML frontmatter between first two '---' delimiters
    2. H1, H2, and a third '---' separator (to be removed)
    3. Article content (to be kept)

    Args:
        content: Full markdown content with YAML frontmatter

    Returns:
        YAML frontmatter + article content (without H1/H2/separator)

    Raises:
        ValueError: If frontmatter format is invalid
    """
    lines = content.split("\n")

    if not lines or lines[0].strip() != "---":
        raise ValueError("File must start with '---' delimiter")

    # Find all '---' delimiters
    delimiter_indices = []
    for i, line in enumerate(lines):
        if line.strip() == "---":
            delimiter_indices.append(i)

    if len(delimiter_indices) < 2:
        raise ValueError("Missing closing '---' delimiter for frontmatter")

    # Keep frontmatter (up to and including 2nd delimiter)
    frontmatter_lines = lines[: delimiter_indices[1] + 1]

    # Find the 3rd delimiter (after H1 and H2)
    # Skip everything until after the 3rd delimiter, or after frontmatter if no 3rd delimiter
    content_start_index = delimiter_indices[2] + 1 if len(delimiter_indices) >= 3 else delimiter_indices[1] + 1

    # Keep content after the 3rd delimiter (or after frontmatter if no 3rd delimiter)
    content_lines = lines[content_start_index:]

    # Combine frontmatter + content
    result_lines = frontmatter_lines + content_lines
    return "\n".join(result_lines)


def preprocess_article(input_path: Path, output_path: Path) -> None:
    """Preprocess a single article file.

    Args:
        input_path: Path to input markdown file
        output_path: Path to output processed file
    """
    try:
        content = input_path.read_text(encoding="utf-8")
        processed_content = preprocess_markdown(content)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write processed file
        output_path.write_text(processed_content, encoding="utf-8")

        logger.info(f"Processed: {input_path.name}")

    except Exception as e:
        logger.error(f"Failed to process {input_path.name}: {e}")
        raise


def main() -> None:
    """Main entry point for article preprocessing."""
    # Load config to get data directory paths (single source of truth)
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    config = Config(config_path=str(config_path))

    # Define paths from config
    input_dir = config.getDataInputDir() / "newspaper" / "articles"
    output_dir = project_root / "frontend" / "newspaper" / "content" / "articles"

    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Get all markdown files
    markdown_files = list(input_dir.glob("*.md"))

    if not markdown_files:
        logger.error(f"No markdown files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(markdown_files)} markdown files to process")

    # Process each file
    success_count = 0
    for input_path in markdown_files:
        try:
            output_path = output_dir / input_path.name
            preprocess_article(input_path, output_path)
            success_count += 1
        except Exception as e:
            logger.error(f"Skipping {input_path.name} due to error: {e}")
            continue

    # Report results
    if success_count == 0:
        logger.error("No articles were successfully processed")
        sys.exit(1)

    logger.info(f"Successfully processed {success_count}/{len(markdown_files)} articles")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
