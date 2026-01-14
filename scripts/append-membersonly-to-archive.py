#!/usr/bin/env python3
"""Append members-only video IDs to download archives.

This script reads reports/video-download-membersonly.log and appends the video IDs
to the appropriate downloaded.txt files so yt-dlp will skip them on future runs.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path to import config module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ChannelConfig, Config


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


def load_existing_video_ids(archive_file: Path) -> set[str]:
    """Load existing video IDs from downloaded.txt file.

    Args:
        archive_file: Path to the downloaded.txt file.

    Returns:
        Set of video IDs that are already in the archive.
    """
    if not archive_file.exists():
        return set()

    video_ids = set()
    with open(archive_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "youtube VIDEO_ID"
            parts = line.split()
            if len(parts) == 2 and parts[0] == "youtube":
                video_ids.add(parts[1])

    return video_ids


def load_config(config_path: Path) -> Config:
    """Load configuration from file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Loaded Config object
    """
    try:
        return Config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)


def parse_members_only_log(log_path: Path, url_to_channel: dict[str, ChannelConfig]) -> tuple[dict[str, list[str]], set[str], int]:
    """Parse members-only log and group by channel.

    Args:
        log_path: Path to members-only log file
        url_to_channel: Mapping of channel URL to config

    Returns:
        Tuple of (channel_video_ids, unknown_urls, total_videos)
    """
    channel_video_ids: dict[str, list[str]] = defaultdict(list)
    unknown_urls: set[str] = set()
    total_videos = 0

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(", ")
            if len(parts) != 2:
                print(f"⚠ Skipping malformed line: {line}", file=sys.stderr)
                continue

            channel_url, video_id = parts
            total_videos += 1

            channel = url_to_channel.get(channel_url)
            if not channel:
                unknown_urls.add(channel_url)
                continue

            channel_video_ids[channel.name].append(video_id)

    return dict(channel_video_ids), unknown_urls, total_videos


def process_channel_videos(channel_name: str, video_ids: list[str], videos_dir: Path) -> tuple[int, int, int]:
    """Process videos for a single channel.

    Args:
        channel_name: Name of the channel
        video_ids: List of video IDs to archive
        videos_dir: Base directory for video downloads

    Returns:
        Tuple of (archived_count, duplicate_count, skipped_count)
    """
    sanitized_name = sanitize_channel_name(channel_name)
    channel_dir = videos_dir / sanitized_name
    archive_file = channel_dir / "downloaded.txt"

    if not channel_dir.exists():
        print(f"⚠ Creating directory: {channel_dir}")
        channel_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_video_ids(archive_file)
    new_video_ids = [vid for vid in video_ids if vid not in existing_ids]
    duplicates = len(video_ids) - len(new_video_ids)

    if not new_video_ids:
        print(f"✓ {sanitized_name}: All {len(video_ids)} video ID(s) already archived")
        return 0, duplicates, 0

    try:
        with open(archive_file, "a", encoding="utf-8") as f:
            for video_id in new_video_ids:
                f.write(f"youtube {video_id}\n")

        if duplicates > 0:
            print(f"✓ {sanitized_name}: Appended {len(new_video_ids)} new video ID(s) ({duplicates} already archived)")
        else:
            print(f"✓ {sanitized_name}: Appended {len(new_video_ids)} video ID(s)")

        return len(new_video_ids), duplicates, 0
    except Exception as e:
        print(f"✗ {sanitized_name}: Failed to append - {e}", file=sys.stderr)
        return 0, duplicates, len(new_video_ids)


def print_summary_report(
    archived_count: int,
    duplicate_count: int,
    skipped_count: int,
    total_videos: int,
    unknown_urls: set[str],
    log_path: Path,
) -> None:
    """Print summary report.

    Args:
        archived_count: Number of archived videos
        duplicate_count: Number of duplicates
        skipped_count: Number of skipped videos
        total_videos: Total number of videos
        unknown_urls: Set of unknown channel URLs
        log_path: Path to members-only log
    """
    if unknown_urls:
        print()
        for url in sorted(unknown_urls):
            with open(log_path, encoding="utf-8") as f:
                video_count = sum(1 for line in f if url in line)
            print(f"⚠ {url}: Channel not found in config (skipping {video_count} video(s))")

    print()
    if duplicate_count > 0:
        print(f"Summary: {archived_count} new, {duplicate_count} already archived, {skipped_count} skipped")
    else:
        print(f"Summary: {archived_count}/{total_videos} video ID(s) successfully archived")
        if skipped_count > 0:
            print(f"         {skipped_count}/{total_videos} video ID(s) skipped (channel not in config)")


def main() -> None:
    """Append members-only video IDs to download archives."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    members_only_log_path = Path("reports/video-download-membersonly.log")

    if not members_only_log_path.exists():
        print(f"No members-only log found at {members_only_log_path}")
        print("Nothing to do.")
        return

    config = load_config(config_path)
    url_to_channel = {channel.url: channel for channel in config.get_channels()}

    channel_video_ids, unknown_urls, total_videos = parse_members_only_log(members_only_log_path, url_to_channel)

    if total_videos == 0:
        print("No members-only videos found in log.")
        return

    print(f"Found {total_videos} members-only videos to archive")

    videos_dir = config.getDataDownloadsVideosDir()
    archived_count = 0
    skipped_count = 0
    duplicate_count = 0

    for channel_name, video_ids in channel_video_ids.items():
        archived, duplicates, skipped = process_channel_videos(channel_name, video_ids, videos_dir)
        archived_count += archived
        duplicate_count += duplicates
        skipped_count += skipped

    print_summary_report(archived_count, duplicate_count, skipped_count, total_videos, unknown_urls, members_only_log_path)


if __name__ == "__main__":
    main()
