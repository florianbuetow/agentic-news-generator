"""SRT file conversion utilities."""

import srt


def srt_to_plain_text(srt_content: str) -> str:
    """Convert SRT to plain text, stripping all timestamps and formatting.

    Args:
        srt_content: SRT format content.

    Returns:
        Plain text with subtitle content only, one entry per line.

    Raises:
        ValueError: If SRT content cannot be parsed.
    """
    try:
        subs = list(srt.parse(srt_content))
    except Exception as e:
        raise ValueError(f"Failed to parse SRT content: {e}") from e

    if not subs:
        raise ValueError("No subtitles found in SRT content")

    lines: list[str] = []
    for sub in subs:
        text = sub.content.strip()
        if text:
            lines.append(text)

    return "\n".join(lines)
