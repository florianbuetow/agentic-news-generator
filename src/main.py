"""Agentic News Generator - Main Entry Point.

Note: Topic detection runs via the `just topics-tree` pipeline (deterministic
hierarchical topic trees). The main entry point is a placeholder.
"""

import sys


def main() -> None:
    """Main function placeholder.

    The old topic segmentation orchestrator has been replaced with the new
    topic detection pipeline. Use the topic tree CLI script instead:

        uv run python scripts/topic-tree.py --help
    """
    print("The main.py entry point is being updated.")
    print("Use the deterministic topic tree pipeline instead:")
    print("  just topics-tree")
    print("or:")
    print("  uv run python scripts/topic-tree.py --help")
    sys.exit(0)


if __name__ == "__main__":
    main()
