"""SRT file conversion utilities."""

from datetime import timedelta
from typing import cast

import srt


def timedelta_to_srt_timestamp(td: timedelta) -> str:
    """Convert timedelta to SRT timestamp format.

    Args:
        td: Timedelta object representing the timestamp.

    Returns:
        Timestamp in HH:MM:SS,mmm format.
    """
    total_seconds = int(td.total_seconds())
    milliseconds = int(td.total_seconds() * 1000) % 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def srt_to_simplified_format(srt_content: str) -> str:
    """Convert SRT to simplified format with timestamp prefixes.

    Format: [HH:MM:SS,mmm] text
    Uses double newlines between entries for readability.

    Args:
        srt_content: SRT format content.

    Returns:
        Simplified format with timestamps.

    Raises:
        ValueError: If SRT content cannot be parsed.
    """
    try:
        subs = list(srt.parse(srt_content))
    except Exception as e:
        raise ValueError(f"Failed to parse SRT content: {e}") from e

    if not subs:
        raise ValueError("No subtitles found in SRT content")

    timed_text: list[str] = []
    for sub in subs:
        # srt library returns timedelta but mypy doesn't have proper type stubs
        timestamp = timedelta_to_srt_timestamp(cast(timedelta, sub.start))
        text = sub.content.strip()
        if text:  # Skip empty subtitles
            timed_text.append(f"[{timestamp}] {text}")

    return "\n\n".join(timed_text)
