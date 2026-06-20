#!/usr/bin/env python3
"""Python wrapper for yt-downloader.sh that reads channels from config.yaml."""

import random
import shutil
import subprocess
import sys
from pathlib import Path

from src.config import ChannelConfig, Config
from src.util.channel_name import sanitize_channel_name
from src.util.log_util import configure_root_logger, get_logger

logger = get_logger(__name__)

# Stop before downloading the next channel once free space drops below this floor.
# Downloads are batched per channel, so this is checked between channels - mirroring
# the per-file disk guard in convert_to_audio.sh.
MIN_FREE_DISK_GB = 20
MIN_FREE_DISK_BYTES = MIN_FREE_DISK_GB * 1024**3


def stop_if_disk_low(videos_dir: Path) -> None:
    """Exit if free space on the videos volume has dropped below the floor."""
    free_bytes = shutil.disk_usage(videos_dir).free
    if free_bytes < MIN_FREE_DISK_BYTES:
        free_mb = free_bytes // (1024 * 1024)
        logger.error(f"🚨 Less than {MIN_FREE_DISK_GB} GB disk space remaining ({free_mb} MB available) - stopping")
        sys.exit(1)


EXIT_CODE_REASONS: dict[int, str] = {
    10: (
        "YouTube cookies are expired or invalid - re-export cookies from your browser "
        "(see reports/video-download.log for the yt-dlp warning that triggered this)"
    ),
    11: ("yt-dlp reported a fatal error (network, auth, parser, etc.) - see reports/video-download.log for details"),
}


def filter_channels_by_name(channels: list[ChannelConfig], channel_filter: str) -> list[ChannelConfig]:
    """Return channels whose name matches channel_filter.

    Matching accepts either the raw ``name`` from config.yaml or its sanitized
    directory form, so callers can pass the CLI-friendly directory name.
    """
    return [channel for channel in channels if channel.name == channel_filter or sanitize_channel_name(channel.name) == channel_filter]


def parse_channel_filter(argv: list[str]) -> str:
    """Parse the optional ``--channel`` argument from argv."""
    if len(argv) == 3 and argv[1] == "--channel":
        return argv[2].strip()
    if len(argv) != 1:
        print("Usage: yt-downloader.py [--channel CHANNEL]", file=sys.stderr)
        sys.exit(1)
    return ""


def resolve_channels(config: Config, channel_filter: str) -> list[ChannelConfig]:
    """Return the channels to process, applying the optional channel filter."""
    channels = config.get_channels()
    if not channels:
        logger.error("No channels found in config.yaml")
        sys.exit(1)
    if not channel_filter:
        return channels
    matched = filter_channels_by_name(channels, channel_filter)
    if not matched:
        available = ", ".join(sorted(sanitize_channel_name(c.name) for c in channels))
        logger.error(f"No channel matching '{channel_filter}' found in config.yaml. Available: {available}")
        sys.exit(1)
    logger.info(f"Filtered to {len(matched)} channel(s) matching '{channel_filter}'")
    return matched


def main() -> None:
    """Read config.yaml and invoke yt-downloader.sh for each channel."""
    channel_filter = parse_channel_filter(sys.argv)

    project_root = Path(__file__).parent.parent
    config_path = Config.repo_config_path()
    shell_script_path = project_root / "scripts" / "yt-downloader.sh"

    try:
        config = Config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    configure_root_logger(config.get_data_logs_dir())

    channels = resolve_channels(config, channel_filter)

    random.shuffle(channels)

    videos_dir = config.get_data_downloads_videos_dir()
    videos_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    failed_channels: list[tuple[str, str]] = []

    for channel in channels:
        logger.info("")
        logger.info(f"=== [{channel.name}] ===")
        logger.info("")

        stop_if_disk_low(videos_dir)

        if channel.download_limiter == 0:
            logger.info(f"Skipping {channel.name} (download-limiter: 0)")
            continue

        max_downloads = 99999 if channel.download_limiter == -1 else channel.download_limiter
        sanitized_name = sanitize_channel_name(channel.name)
        output_dir = config.get_data_downloads_videos_dir() / sanitized_name

        logger.info(f"Processing channel: {channel.name} ({channel.url})")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Download limit: {max_downloads}")

        try:
            result = subprocess.run(
                [str(shell_script_path), channel.url, str(output_dir), str(max_downloads)],
                check=False,
                capture_output=False,
            )
            if result.returncode == 0:
                logger.info(f"Successfully processed {channel.name}")
                success_count += 1
            else:
                reason = EXIT_CODE_REASONS.get(
                    result.returncode,
                    f"yt-downloader.sh exited with unmapped code {result.returncode}",
                )
                logger.error(f"Failed to process {channel.name}: {reason}")
                failure_count += 1
                failed_channels.append((channel.name, reason))
        except Exception as e:
            reason = str(e)
            logger.error(f"Error processing {channel.name}: {reason}")
            failure_count += 1
            failed_channels.append((channel.name, reason))

    logger.info(f"Summary: {success_count} succeeded, {failure_count} failed")

    if failure_count > 0:
        logger.error("Failed channels:")
        for name, reason in failed_channels:
            logger.error(f"  - {name}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
