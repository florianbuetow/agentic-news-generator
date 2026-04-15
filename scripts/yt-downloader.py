#!/usr/bin/env python3
"""Python wrapper for yt-downloader.sh that reads channels from config.yaml."""

import random
import subprocess
import sys
from pathlib import Path

from src.config import Config
from src.util.channel_name import sanitize_channel_name


def main() -> None:
    """Read config.yaml and invoke yt-downloader.sh for each channel."""
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    shell_script_path = project_root / "scripts" / "yt-downloader.sh"

    # Load configuration
    try:
        config = Config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Get all channels
    channels = config.get_channels()

    if not channels:
        print("No channels found in config.yaml", file=sys.stderr)
        sys.exit(1)

    # Randomize channel order to avoid predictable patterns
    random.shuffle(channels)

    # Track success/failure
    success_count = 0
    failure_count = 0

    # Process each channel
    for channel in channels:
        print()
        print(f"📺 === [{channel.name}] ===")
        print()

        # Check if downloads are disabled for this channel
        if channel.download_limiter == 0:
            print(f"⊘ Skipping {channel.name} (download-limiter: 0)")
            continue

        # Determine max downloads to pass to bash script
        max_downloads = 99999 if channel.download_limiter == -1 else channel.download_limiter

        # Sanitize channel name for directory use
        sanitized_name = sanitize_channel_name(channel.name)
        output_dir = config.getDataDownloadsVideosDir() / sanitized_name

        print(f"Processing channel: {channel.name} ({channel.url})")
        print(f"  Output directory: {output_dir}")
        print(f"  Download limit: {max_downloads}")

        try:
            # Invoke the shell script with the channel URL, output directory, and max downloads
            result = subprocess.run(
                [str(shell_script_path), channel.url, str(output_dir), str(max_downloads)],
                check=False,
                capture_output=False,
            )
            if result.returncode == 0:
                print(f"✓ Successfully processed {channel.name}")
                success_count += 1
            else:
                print(f"✗ Failed to process {channel.name} (exit code: {result.returncode})")
                failure_count += 1
        except Exception as e:
            print(f"✗ Error processing {channel.name}: {e}", file=sys.stderr)
            failure_count += 1

    # Report summary
    print(f"\nSummary: {success_count} succeeded, {failure_count} failed")

    # Exit with error if any failed
    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
