#!/usr/bin/env python3
"""Find and delete old-format *_topic_tree.json files.

Old format is identified by top-level `keyphrases` missing the nested
`tfidf`/`yake`/`keybert` structure that the current schema requires.

Usage:
    uv run python scripts/delete-old-topic-trees.py --dry-run
    uv run python scripts/delete-old-topic-trees.py --delete
"""

import argparse
import json
import sys
from pathlib import Path

from src.config import Config


def is_old_format(path: Path) -> bool:
    """Return True if the topic tree JSON uses the old keyphrase format."""
    data = json.loads(path.read_text(encoding="utf-8"))
    top_kp = data.get("keyphrases", {})
    if not isinstance(top_kp, dict):
        return True
    return "tfidf" not in top_kp


def _delete_old_files(old_files: list[Path], topics_dir: Path) -> int:
    """Delete old-format files and return exit code."""
    deleted = 0
    failed = 0
    for f in old_files:
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            print(f"  Failed to delete {f.relative_to(topics_dir)}: {e}")
            failed += 1
    print(f"Deleted {deleted} old-format files, {failed} failures")
    return 1 if failed > 0 else 0


def main() -> int:
    """Find and delete old-format topic tree files."""
    parser = argparse.ArgumentParser(description="Find and delete old-format topic tree JSON files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="List old-format files without deleting")
    group.add_argument("--delete", action="store_true", help="Delete old-format files")
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config(config_path)

    topics_dir = config.getTopicDetectionOutputDir()

    if not topics_dir.exists():
        print(f"Error: Topics directory not found: {topics_dir}", file=sys.stderr)
        return 1

    tree_files = sorted(topics_dir.rglob("*_topic_tree.json"))
    tree_files = [f for f in tree_files if not f.name.startswith("._")]

    print(f"Scanning {len(tree_files)} topic tree files in {topics_dir}")
    print()

    old_files: list[Path] = []
    new_files: list[Path] = []
    error_files: list[tuple[Path, str]] = []

    for f in tree_files:
        try:
            if is_old_format(f):
                old_files.append(f)
            else:
                new_files.append(f)
        except Exception as e:
            error_files.append((f, str(e)))

    print(f"Old format (to delete): {len(old_files)}")
    print(f"New format (keep):      {len(new_files)}")
    print(f"Read errors:            {len(error_files)}")
    print()

    if error_files:
        print("Files with read errors:")
        for f, err in error_files:
            print(f"  {f.relative_to(topics_dir)}: {err}")
        print()

    if not old_files:
        print("No old-format files found.")
        return 0

    if args.dry_run:
        print("Old-format files (dry run, not deleting):")
        for f in old_files:
            print(f"  {f.relative_to(topics_dir)}")
        return 0

    return _delete_old_files(old_files, topics_dir)


if __name__ == "__main__":
    sys.exit(main())
