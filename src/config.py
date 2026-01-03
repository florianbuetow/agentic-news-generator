"""Configuration loader for the Agentic News Generator."""

from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ChannelConfig(BaseModel):
    """Pydantic model for validating channel configurations.

    All fields are required to ensure complete channel configuration.
    """

    url: str = Field(..., description="URL of the channel")
    name: str = Field(..., description="Name of the channel")
    category: str = Field(..., description="Category classification of the channel")
    description: str = Field(..., description="Description of the channel content and style")

    model_config = ConfigDict(frozen=True, extra="forbid")


class LLMConfig(BaseModel):
    """Configuration for an individual LLM connection via litellm."""

    model: str = Field(
        ...,
        description="litellm model string (e.g., 'openai/gpt-4', 'anthropic/claude-3-5-sonnet', 'openai/local-model' for LM Studio)",
    )
    api_base: str | None = Field(..., description="API base URL (for LM Studio or custom endpoints, None for standard providers)")
    api_key_env: str = Field(..., description="Environment variable name for API key")
    context_window: int = Field(..., description="Model context window size in tokens")
    max_tokens: int = Field(..., description="Maximum tokens for response/completion")
    temperature: float = Field(..., description="Sampling temperature (0.0-1.0)")
    context_window_threshold: int = Field(
        ...,
        description="Percentage threshold (0-100) for context window usage before raising an error",
        ge=0,
        le=100,
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicSegmentationConfig(BaseModel):
    """Configuration for topic segmentation agent system."""

    agent_llm: LLMConfig = Field(..., description="LLM config for segmentation agent")
    critic_llm: LLMConfig = Field(..., description="LLM config for critic agent")
    retry_limit: int = Field(..., description="Maximum retry attempts")

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

        # Validate topic_segmentation section if present
        if "topic_segmentation" in self._data:
            self._topic_segmentation = self._validate_topic_segmentation()

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

    def _validate_topic_segmentation(self) -> TopicSegmentationConfig:
        """Validate topic segmentation configuration.

        Returns:
            Validated TopicSegmentationConfig instance.

        Raises:
            ValueError: If topic segmentation configuration is invalid.
        """
        try:
            return TopicSegmentationConfig.model_validate(self._data["topic_segmentation"])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"Topic segmentation configuration validation failed: {error_messages}") from e

    def get_topic_segmentation_config(self) -> TopicSegmentationConfig:
        """Get topic segmentation configuration.

        Returns:
            TopicSegmentationConfig instance.

        Raises:
            KeyError: If topic_segmentation section is not configured.
        """
        if not hasattr(self, "_topic_segmentation"):
            raise KeyError("Missing required key 'topic_segmentation' in config file")
        return self._topic_segmentation
