#!/usr/bin/env python3
"""Parse video download log to extract members-only video information.

This script reads reports/video-download.log and extracts:
- YouTube channel URLs from yt-dlp extraction lines
- Video IDs from members-only error messages

Output is written to reports/video-download-membersonly.log with format:
    channel_url, video_id
"""

from pathlib import Path


def getYTChannelUrl(line: str) -> str | None:
    """Extract YouTube channel URL from yt-dlp extraction line.

    Example input:
        '[youtube:tab] Extracting URL: https://www.youtube.com/@WesRoth/videos'
    Example output:
        'https://www.youtube.com/@WesRoth/videos'

    Args:
        line: A log line to parse

    Returns:
        Channel URL if found, None otherwise
    """
    prefix = "[youtube:tab] Extracting URL:"
    if line.startswith(prefix):
        return line[len(prefix) :].strip()
    return None


def getYTVideoId(line: str) -> str | None:
    """Extract YouTube video ID from members-only error line.

    Example inputs:
    1) ERROR: [youtube] i6Wr1VBJ_bA: Join this channel to get access to members-only content...
    2) ERROR: [youtube] _LQgDzgLTD0: This video is available to this channel's members...
    3) ERROR: [youtube] ADWyWOlg8mg: This video is available to this channel's members...

    Example outputs:
     1) 'i6Wr1VBJ_bA'
     2) '_LQgDzgLTD0'
     3) 'ADWyWOlg8mg'

    Args:
        line: A log line to parse

    Returns:
        Video ID if found, None otherwise
    """
    prefix = "ERROR: [youtube] "
    if not line.startswith(prefix):
        return None

    # Extract everything after the prefix
    after_prefix = line[len(prefix) :]

    # Find the first colon - video ID is everything before it
    colon_index = after_prefix.find(":")
    if colon_index == -1:
        return None

    video_id = after_prefix[:colon_index].strip()
    return video_id if video_id else None


def main() -> None:
    """Parse video download log and extract members-only video information."""
    input_log_path = Path("reports/video-download.log")
    output_log_path = Path("reports/video-download-membersonly.log")

    if not input_log_path.exists():
        print(f"Error: {input_log_path} does not exist")
        return

    # Ensure output directory exists
    output_log_path.parent.mkdir(parents=True, exist_ok=True)

    channel_url: str | None = None
    members_only_count = 0

    with open(input_log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            # Check for channel URL
            url = getYTChannelUrl(line)
            if url:
                channel_url = url
                continue

            # Check for members-only video
            video_id = getYTVideoId(line)
            if video_id:
                with open(output_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{channel_url}, {video_id}\n")
                members_only_count += 1

    print(f"Found {members_only_count} members-only videos")
    print(f"Output written to {output_log_path}")


if __name__ == "__main__":
    main()
