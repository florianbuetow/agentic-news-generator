"""YouTube ID extraction for analytics file joins.

Deliberately duplicated from the pipeline scripts (which analytics must not
import) so the package stays import-isolated. The regex matches the
``Title [video_id].ext`` convention produced by the downloader, including
double extensions such as ``.info.json``.
"""

import re

YOUTUBE_ID_BEFORE_EXTENSION_RE: re.Pattern[str] = re.compile(r"\s*\[([A-Za-z0-9_-]+)\](?=\.[^/]+$)")


def extract_youtube_id(filename: str) -> str | None:
    """Return the bracketed YouTube ID immediately before the extension.

    Args:
        filename: File name or path following the ``Title [id].ext`` convention.

    Returns:
        The ID token, or None when the name carries no such token.
    """
    match = YOUTUBE_ID_BEFORE_EXTENSION_RE.search(filename)
    return match.group(1) if match is not None else None
