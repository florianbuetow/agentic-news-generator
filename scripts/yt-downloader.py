#!/usr/bin/env python3
"""Python wrapper for yt-downloader.sh that reads channels from config.yaml."""

import re
import subprocess
import sys
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

    # Track success/failure
    success_count = 0
    failure_count = 0

    # Process each channel
    for channel in channels:
        # Sanitize channel name for directory use
        sanitized_name = sanitize_channel_name(channel.name)
        output_dir = project_root / "data" / "downloads" / "video" / sanitized_name

        print(f"Processing channel: {channel.name} ({channel.url})")
        print(f"  Output directory: {output_dir}")

        try:
            # Invoke the shell script with the channel URL and output directory
            result = subprocess.run(
                [str(shell_script_path), channel.url, str(output_dir)],
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
