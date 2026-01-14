#!/usr/bin/env python3
"""Parse video download log and archive members-only video IDs.

This script:
1. Parses reports/video-download.log to find members-only videos
2. Maps channel URLs to their download archive files
3. Appends video IDs to downloaded.txt to prevent re-download attempts
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path to import config module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ChannelConfig, Config


def parse_channel_url(line: str) -> str | None:
    """Extract YouTube channel URL from yt-dlp extraction line.

    Args:
        line: A log line to parse

    Returns:
        Channel URL if found, None otherwise
    """
    prefix = "[youtube:tab] Extracting URL:"
    if line.startswith(prefix):
        return line[len(prefix) :].strip()
    return None


def parse_video_id(line: str) -> str | None:
    """Extract YouTube video ID from members-only error line.

    Args:
        line: A log line to parse

    Returns:
        Video ID if found, None otherwise
    """
    prefix = "ERROR: [youtube] "
    if not line.startswith(prefix):
        return None

    after_prefix = line[len(prefix) :]
    colon_index = after_prefix.find(":")
    if colon_index == -1:
        return None

    video_id = after_prefix[:colon_index].strip()
    return video_id if video_id else None


def parse_download_log(log_path: Path) -> dict[str, list[str]]:
    """Parse download log and extract members-only videos by channel URL.

    Args:
        log_path: Path to the video download log file

    Returns:
        Dictionary mapping channel URL to list of video IDs
    """
    url_to_video_ids: dict[str, list[str]] = defaultdict(list)
    current_channel_url: str | None = None

    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            # Check for channel URL
            url = parse_channel_url(line)
            if url:
                current_channel_url = url
                continue

            # Check for members-only video
            video_id = parse_video_id(line)
            if video_id and current_channel_url:
                url_to_video_ids[current_channel_url].append(video_id)

    return dict(url_to_video_ids)


def sanitize_channel_name(name: str) -> str:
    """Sanitize channel name for use as a directory name.

    Args:
        name: The channel name to sanitize

    Returns:
        A sanitized version safe for filesystem paths
    """
    sanitized = re.sub(r"[^\w\s-]", "", name)
    sanitized = re.sub(r"[-\s]+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized


def load_existing_video_ids(archive_file: Path) -> set[str]:
    """Load existing video IDs from downloaded.txt file.

    Args:
        archive_file: Path to the downloaded.txt file

    Returns:
        Set of video IDs already in the archive
    """
    if not archive_file.exists():
        return set()

    video_ids: set[str] = set()
    with open(archive_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) == 2 and parts[0] == "youtube":
                video_ids.add(parts[1])

    return video_ids


def build_url_to_channel_map(channels: list[ChannelConfig]) -> dict[str, ChannelConfig]:
    """Build a mapping from channel URL to channel configuration.

    Args:
        channels: List of channel configurations

    Returns:
        Dictionary mapping URL to ChannelConfig
    """
    return {channel.url: channel for channel in channels}


def append_new_video_ids(archive_file: Path, video_ids: list[str]) -> None:
    """Append new video IDs to archive file.

    Args:
        archive_file: Path to the downloaded.txt file
        video_ids: List of video IDs to append
    """
    with open(archive_file, "a", encoding="utf-8") as f:
        for video_id in video_ids:
            f.write(f"youtube {video_id}\n")


def process_channel(
    channel: ChannelConfig, video_ids: list[str], videos_dir: Path
) -> tuple[int, int]:
    """Process members-only videos for a single channel.

    Args:
        channel: Channel configuration
        video_ids: List of video IDs to archive
        videos_dir: Base directory for video downloads

    Returns:
        Tuple of (archived_count, duplicate_count)
    """
    sanitized_name = sanitize_channel_name(channel.name)
    channel_dir = videos_dir / sanitized_name
    archive_file = channel_dir / "downloaded.txt"

    # Create directory if needed
    if not channel_dir.exists():
        print(f"⚠ Creating directory: {channel_dir}")
        channel_dir.mkdir(parents=True, exist_ok=True)

    # Load existing IDs and filter duplicates
    existing_ids = load_existing_video_ids(archive_file)
    new_video_ids = [vid for vid in video_ids if vid not in existing_ids]
    duplicate_count = len(video_ids) - len(new_video_ids)

    # Report and append
    if not new_video_ids:
        print(f"✓ {sanitized_name}: All {len(video_ids)} video ID(s) already archived")
        return 0, duplicate_count

    append_new_video_ids(archive_file, new_video_ids)

    if duplicate_count > 0:
        print(
            f"✓ {sanitized_name}: Appended {len(new_video_ids)} new video ID(s) "
            f"({duplicate_count} already archived)"
        )
    else:
        print(f"✓ {sanitized_name}: Appended {len(new_video_ids)} video ID(s)")

    return len(new_video_ids), duplicate_count


def main() -> None:
    """Parse download log and archive members-only video IDs."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    download_log_path = Path("reports/video-download.log")

    # Check if download log exists
    if not download_log_path.exists():
        print(f"No download log found at {download_log_path}")
        print("Nothing to do.")
        return

    # Load configuration
    try:
        config = Config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse download log
    url_to_video_ids = parse_download_log(download_log_path)

    if not url_to_video_ids:
        print("No members-only videos found in download log.")
        return

    total_videos = sum(len(ids) for ids in url_to_video_ids.values())
    print(f"Found {total_videos} members-only videos to archive")

    # Build URL mapping
    url_to_channel = build_url_to_channel_map(config.get_channels())
    videos_dir = config.getDataDownloadsVideosDir()

    # Process each channel
    total_archived = 0
    total_duplicates = 0
    unknown_urls: list[tuple[str, int]] = []

    for channel_url, video_ids in url_to_video_ids.items():
        channel = url_to_channel.get(channel_url)
        if not channel:
            unknown_urls.append((channel_url, len(video_ids)))
            continue

        try:
            archived, duplicates = process_channel(channel, video_ids, videos_dir)
            total_archived += archived
            total_duplicates += duplicates
        except Exception as e:
            print(f"✗ {channel.name}: Failed to process - {e}", file=sys.stderr)

    # Report unknown URLs
    if unknown_urls:
        print()
        for url, count in unknown_urls:
            print(f"⚠ {url}: Channel not found in config (skipping {count} video(s))")

    # Summary
    total_skipped = sum(count for _, count in unknown_urls)
    print()
    if total_duplicates > 0:
        print(f"Summary: {total_archived} new, {total_duplicates} already archived, {total_skipped} skipped")
    else:
        print(f"Summary: {total_archived}/{total_videos} video ID(s) successfully archived")
        if total_skipped > 0:
            print(f"         {total_skipped}/{total_videos} video ID(s) skipped (channel not in config)")


if __name__ == "__main__":
    main()
