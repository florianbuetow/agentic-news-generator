"""Configuration loader for the Agentic News Generator."""

from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ParagraphConfig(BaseModel):
    """Configuration for paragraph extraction."""

    hero_count: int = Field(..., gt=0, description="Paragraphs for hero article")
    secondary_count: int = Field(..., gt=0, description="Paragraphs for secondary article")
    featured_count: int = Field(..., gt=0, description="Paragraphs for featured articles")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ImageConfig(BaseModel):
    """Configuration for image extraction."""

    extract_first: bool = Field(..., description="Extract first image from markdown")
    fallback_url: str | None = Field(None, description="Fallback URL if no image")

    model_config = ConfigDict(frozen=True, extra="forbid")


class LinkConfig(BaseModel):
    """Configuration for link generation."""

    base_path: str = Field(..., description="Base path for article links")
    slug_from_filename: bool = Field(..., description="Generate slug from filename")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleCompilerConfig(BaseModel):
    """Configuration for article compilation."""

    input_dir: str = Field(..., description="Input directory for markdown articles")
    output_file: str = Field(..., description="Output path for articles.js")
    min_articles: int = Field(..., gt=0, description="Minimum articles required")
    date_format: str = Field(..., description="Date format in YAML frontmatter")
    paragraphs: ParagraphConfig
    images: ImageConfig
    links: LinkConfig

    model_config = ConfigDict(frozen=True, extra="forbid")


class ChannelConfig(BaseModel):
    """Pydantic model for validating channel configurations.

    All fields are required to ensure complete channel configuration.
    """

    url: str = Field(..., description="URL of the channel")
    name: str = Field(..., description="Name of the channel")
    category: str = Field(..., description="Category classification of the channel")
    description: str = Field(..., description="Description of the channel content and style")

    model_config = ConfigDict(frozen=True, extra="forbid")


class Config:
    """Configuration class that loads and provides access to config.yaml."""

    def __init__(self, config_path: str | Path) -> None:
        """Initialize the Config by loading the YAML file.

        Args:
            config_path: Path to the config.yaml file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the YAML file is invalid.
            KeyError: If required keys are missing from the config.
            ValueError: If channel configurations are invalid or missing required fields.
        """
        self._load(config_path)

    def _load(self, config_path: str | Path) -> None:
        """Load the configuration from a YAML file.

        Args:
            config_path: Path to the config.yaml file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the YAML file is invalid.
            KeyError: If required keys are missing from the config.
            ValueError: If channel configurations are invalid or missing required fields.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

        # Handle empty or None YAML files
        if self._data is None:
            raise KeyError("Missing required key 'channels' in config file")

        # Explicitly check for required top-level keys - no defaults
        if "channels" not in self._data:
            raise KeyError("Missing required key 'channels' in config file")

        # Validate and parse all channel configurations using Pydantic
        self._channels = self._validate_channels()

    def _validate_channels(self) -> list[ChannelConfig]:
        """Validate and parse all channel configurations using Pydantic.

        Returns:
            List of validated ChannelConfig instances.

        Raises:
            ValueError: If any channel configuration is invalid or missing required fields.
        """
        # Access channels directly - we know it exists from _load validation
        channels_raw = self._data["channels"]
        if not isinstance(channels_raw, list):
            raise ValueError("'channels' must be a list")

        channels = cast(list[dict[str, Any]], channels_raw)

        validation_errors: list[str] = []
        validated_channels: list[ChannelConfig] = []
        for idx, channel_data in enumerate(channels):
            try:
                validated_channels.append(ChannelConfig.model_validate(channel_data))
            except ValidationError as e:
                # Try to get channel name for error reporting, fallback to index if missing
                try:
                    channel_name = channel_data["name"]
                except KeyError:
                    channel_name = f"channel at index {idx}"
                error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
                validation_errors.append(f"Channel '{channel_name}' (index {idx}): {error_messages}")

        if validation_errors:
            error_msg = "Channel validation failed:\n" + "\n".join(validation_errors)
            raise ValueError(error_msg)

        return validated_channels

    def get_channels(self) -> list[ChannelConfig]:
        """Get all channel configurations.

        Returns:
            List of ChannelConfig instances.
        """
        return self._channels

    def get_channel(self, index: int) -> ChannelConfig:
        """Get a specific channel by index.

        Args:
            index: Zero-based index of the channel.

        Returns:
            ChannelConfig instance.

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self._channels):
            raise IndexError(f"Channel index {index} out of range (0-{len(self._channels) - 1})")
        return self._channels[index]

    def get_channel_by_name(self, name: str) -> ChannelConfig:
        """Get a channel by its name.

        Args:
            name: The name of the channel.

        Returns:
            ChannelConfig instance.

        Raises:
            KeyError: If no channel with the given name exists.
        """
        for channel in self._channels:
            if channel.name == name:
                return channel
        raise KeyError(f"No channel found with name: {name}")

    def get_article_compiler_config(self) -> ArticleCompilerConfig:
        """Get article compiler configuration.

        Returns:
            ArticleCompilerConfig instance.

        Raises:
            KeyError: If article_compiler section is missing from config.
        """
        if "article_compiler" not in self._data:
            raise KeyError("Missing required key 'article_compiler' in config file")
        return ArticleCompilerConfig.model_validate(self._data["article_compiler"])
