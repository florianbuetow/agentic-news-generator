"""Agentic News Generator - Main Entry Point."""

from pathlib import Path

from src.config import Config


def main() -> None:
    """Main function to run the news generator."""
    # Load configuration from config/config.yaml
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config(config_path)

    # Get all channels
    channels = config.get_channels()
    print(f"Loaded {len(channels)} channels from configuration")
    print("\nChannels:")
    for channel in channels:
        print(f"  - {channel.name} ({channel.category})")


if __name__ == "__main__":
    main()
