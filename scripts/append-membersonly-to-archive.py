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


def main() -> None:
    """Append members-only video IDs to download archives."""
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    members_only_log_path = Path("reports/video-download-membersonly.log")

    # Check if members-only log exists
    if not members_only_log_path.exists():
        print(f"No members-only log found at {members_only_log_path}")
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

    # Build URL → Channel mapping
    url_to_channel = {channel.url: channel for channel in config.get_channels()}

    # Read members-only log and group by channel
    channel_video_ids: dict[str, list[str]] = defaultdict(list)
    unknown_urls: set[str] = set()
    total_videos = 0

    with open(members_only_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse line: "channel_url, video_id"
            parts = line.split(", ")
            if len(parts) != 2:
                print(f"⚠ Skipping malformed line: {line}", file=sys.stderr)
                continue

            channel_url, video_id = parts
            total_videos += 1

            # Find channel by URL
            channel = url_to_channel.get(channel_url)
            if not channel:
                unknown_urls.add(channel_url)
                continue

            # Group video IDs by channel name
            channel_video_ids[channel.name].append(video_id)

    if total_videos == 0:
        print("No members-only videos found in log.")
        return

    print(f"Found {total_videos} members-only videos to archive")

    # Process each channel
    videos_dir = config.getDataDownloadsVideosDir()
    archived_count = 0
    skipped_count = 0
    duplicate_count = 0

    for channel_name, video_ids in channel_video_ids.items():
        # Sanitize channel name and build path
        sanitized_name = sanitize_channel_name(channel_name)
        channel_dir = videos_dir / sanitized_name
        archive_file = channel_dir / "downloaded.txt"

        # Create directory if it doesn't exist
        if not channel_dir.exists():
            print(f"⚠ Creating directory: {channel_dir}")
            channel_dir.mkdir(parents=True, exist_ok=True)

        # Load existing video IDs
        existing_ids = load_existing_video_ids(archive_file)

        # Filter out duplicates
        new_video_ids = [vid for vid in video_ids if vid not in existing_ids]
        duplicates = len(video_ids) - len(new_video_ids)

        if duplicates > 0:
            duplicate_count += duplicates

        # Skip if no new video IDs
        if not new_video_ids:
            print(f"✓ {sanitized_name}: All {len(video_ids)} video ID(s) already archived")
            continue

        # Append only new video IDs to archive file
        try:
            with open(archive_file, "a", encoding="utf-8") as f:
                for video_id in new_video_ids:
                    f.write(f"youtube {video_id}\n")

            if duplicates > 0:
                print(
                    f"✓ {sanitized_name}: Appended {len(new_video_ids)} new video ID(s) "
                    f"({duplicates} already archived)"
                )
            else:
                print(f"✓ {sanitized_name}: Appended {len(new_video_ids)} video ID(s)")

            archived_count += len(new_video_ids)
        except Exception as e:
            print(f"✗ {sanitized_name}: Failed to append - {e}", file=sys.stderr)
            skipped_count += len(new_video_ids)

    # Report unknown URLs
    if unknown_urls:
        print()
        for url in sorted(unknown_urls):
            video_count = sum(1 for line in open(members_only_log_path, encoding="utf-8") if url in line)
            print(f"⚠ {url}: Channel not found in config (skipping {video_count} video(s))")
            skipped_count += video_count

    # Summary
    print()
    if duplicate_count > 0:
        print(f"Summary: {archived_count} new, {duplicate_count} already archived, {skipped_count} skipped")
    else:
        print(f"Summary: {archived_count}/{total_videos} video ID(s) successfully archived")
        if skipped_count > 0:
            print(f"         {skipped_count}/{total_videos} video ID(s) skipped (channel not in config)")


if __name__ == "__main__":
    main()
