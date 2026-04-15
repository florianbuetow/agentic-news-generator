"""Channel-name sanitization shared by the downloader and downstream scripts.

The downloader stores each channel's data under a filesystem-safe directory
derived from the raw channel name in ``config.yaml``. Any script that needs to
resolve disk paths from a ``ChannelConfig`` must apply the same transformation.
"""

from __future__ import annotations

import re


def sanitize_channel_name(name: str) -> str:
    """Sanitize a channel name for use as a directory name.

    Replaces whitespace and dashes with underscores and strips any character
    that is not alphanumeric, underscore, or dash.

    Args:
        name: The raw channel name from config.yaml.

    Returns:
        A filesystem-safe version of the name.
    """
    sanitized = re.sub(r"[^\w\s-]", "", name)
    sanitized = re.sub(r"[-\s]+", "_", sanitized)
    return sanitized.strip("_")
