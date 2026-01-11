#!/usr/bin/env python3
"""Validate that the articles directory exists and contains markdown files.

This script reads the path from config.yaml and validates the directory,
ensuring consistency with the single source of truth approach.

Exit codes:
    0: Articles directory exists and contains markdown files
    1: Error (directory doesn't exist or no markdown files found)
"""

import sys
from pathlib import Path

# Add src to path to import Config
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import Config


def main() -> None:
    """Validate articles directory from config."""
    # Load config
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    config = Config(config_path=str(config_path))

    # Get articles directory from config
    articles_dir = config.getDataInputDir() / "newspaper" / "articles"

    # Check if directory exists
    if not articles_dir.exists():
        print(f"✗ Error: Articles directory does not exist: {articles_dir}", file=sys.stderr)
        print("  Please generate the articles first", file=sys.stderr)
        sys.exit(1)

    # Check if directory contains markdown files
    markdown_files = list(articles_dir.glob("*.md"))
    if not markdown_files:
        print(f"✗ Error: No markdown articles found in {articles_dir}", file=sys.stderr)
        print("  Please generate the articles first", file=sys.stderr)
        sys.exit(1)

    # Success - directory exists and contains markdown files
    sys.exit(0)


if __name__ == "__main__":
    main()
