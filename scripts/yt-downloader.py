#!/usr/bin/env python3
"""Python wrapper for yt-downloader.sh that reads channels from config.yaml."""

import random
import subprocess
import sys
from pathlib import Path

from src.config import Config
from src.util.channel_name import sanitize_channel_name
from src.util.log_util import configure_root_logger, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Read config.yaml and invoke yt-downloader.sh for each channel."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    shell_script_path = project_root / "scripts" / "yt-downloader.sh"

    try:
        config = Config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    configure_root_logger(config.getDataLogsDir())

    channels = config.get_channels()

    if not channels:
        logger.error("No channels found in config.yaml")
        sys.exit(1)

    random.shuffle(channels)

    success_count = 0
    failure_count = 0

    for channel in channels:
        logger.info("")
        logger.info(f"=== [{channel.name}] ===")
        logger.info("")

        if channel.download_limiter == 0:
            logger.info(f"Skipping {channel.name} (download-limiter: 0)")
            continue

        max_downloads = 99999 if channel.download_limiter == -1 else channel.download_limiter
        sanitized_name = sanitize_channel_name(channel.name)
        output_dir = config.getDataDownloadsVideosDir() / sanitized_name

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
                logger.error(f"Failed to process {channel.name} (exit code: {result.returncode})")
                failure_count += 1
        except Exception as e:
            logger.error(f"Error processing {channel.name}: {e}")
            failure_count += 1

    logger.info(f"Summary: {success_count} succeeded, {failure_count} failed")

    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
