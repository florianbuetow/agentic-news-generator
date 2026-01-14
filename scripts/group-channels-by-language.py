#!/usr/bin/env python3
"""Group channels by language from config.yaml.

This script reads all channels from config.yaml and outputs them grouped
by language as JSON. Used by transcribe_audio.sh to process channels in
language-based batches, minimizing model switching overhead.

Output format:
{
  "en": [
    {"name": "Channel Name", "sanitized_name": "Channel_Name", "language": "en"},
    ...
  ],
  "de": [
    {"name": "German Channel", "sanitized_name": "German_Channel", "language": "de"},
    ...
  ],
  ...
}
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config


def sanitize_channel_name(name: str) -> str:
    """Sanitize channel name for use as a directory name.

    Removes or replaces characters that are not safe for filesystem paths.

    Args:
        name: The channel name to sanitize.

    Returns:
        A sanitized version of the name safe for use in filesystem paths.
    """
    # Replace spaces and special characters with underscores
    # Keep alphanumeric characters, hyphens, and underscores
    sanitized = re.sub(r"[^\w\s-]", "", name)
    sanitized = re.sub(r"[-\s]+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def main() -> None:
    """Group channels by language and output as JSON."""
    try:
        config = Config("config/config.yaml")
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Group channels by language
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)

    for channel in config.get_channels():
        groups[channel.language].append(
            {
                "name": channel.name,
                "sanitized_name": sanitize_channel_name(channel.name),
                "language": channel.language,
            }
        )

    # Output as JSON
    # Format: {"en": [...], "de": [...], "ja": [...]}
    print(json.dumps(dict(groups)))


if __name__ == "__main__":
    main()
