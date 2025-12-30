"""SRT file conversion utilities."""

import srt


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
        timestamp = sub.start.to_timecode()  # HH:MM:SS,mmm format
        text = sub.content.strip()
        if text:  # Skip empty subtitles
            timed_text.append(f"[{timestamp}] {text}")

    return "\n\n".join(timed_text)
