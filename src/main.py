"""Agentic News Generator - Main Entry Point.

Note: The topic segmentation pipeline is being migrated to the new
topic_detection module. Use scripts/topic-detection.py for topic detection.
"""

import sys


def main() -> None:
    """Main function placeholder.

    The old topic segmentation orchestrator has been replaced with the new
    topic detection pipeline. Use the CLI script instead:

        uv run scripts/topic-detection.py --help
    """
    print("The main.py entry point is being updated.")
    print("Use the topic detection CLI script instead:")
    print("  uv run scripts/topic-detection.py --help")
    sys.exit(0)


if __name__ == "__main__":
    main()
