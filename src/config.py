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
    download_limiter: int = Field(
        ...,
        alias="download-limiter",
        description="Max videos to download: 0=skip, -1=unlimited, >0=exact limit",
    )

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


class PathsConfig(BaseModel):
    """Configuration for all project directory paths."""

    data_dir: str = Field(..., description="Root data directory path", min_length=1)
    data_downloads_dir: str = Field(..., description="Downloads directory path", min_length=1)
    data_downloads_videos_dir: str = Field(..., description="Downloaded videos directory path", min_length=1)
    data_downloads_transcripts_dir: str = Field(..., description="Transcripts directory path", min_length=1)
    data_downloads_transcripts_hallucinations_dir: str = Field(
        ..., description="Transcript hallucinations analysis directory path", min_length=1
    )
    data_downloads_transcripts_cleaned_dir: str = Field(..., description="Cleaned transcripts directory path", min_length=1)
    data_transcripts_topics_dir: str = Field(..., description="Transcripts split by topics directory path", min_length=1)
    data_downloads_audio_dir: str = Field(..., description="Audio files directory path", min_length=1)
    data_downloads_metadata_dir: str = Field(..., description="Metadata files directory path", min_length=1)
    data_output_dir: str = Field(..., description="Output directory path", min_length=1)
    data_input_dir: str = Field(..., description="Input directory path", min_length=1)
    data_temp_dir: str = Field(..., description="Temporary files directory path", min_length=1)
    data_archive_dir: str = Field(..., description="Archive directory path", min_length=1)
    data_archive_videos_dir: str = Field(..., description="Archived videos directory path", min_length=1)
    data_logs_dir: str = Field(..., description="Logs directory path", min_length=1)

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
            ValueError: If channel configurations or paths are invalid or missing required fields.
        """
        self.config_path = Path(config_path)
        self._load(config_path)

        # Validate paths section (required)
        self._paths = self._validate_paths()

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

    def _validate_paths(self) -> PathsConfig:
        """Validate paths configuration.

        Returns:
            Validated PathsConfig instance.

        Raises:
            KeyError: If paths section is missing.
            ValueError: If paths configuration is invalid or contains empty paths.
        """
        if "paths" not in self._data:
            raise KeyError("Missing required key 'paths' in config file")

        try:
            return PathsConfig.model_validate(self._data["paths"])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"Paths configuration validation failed: {error_messages}") from e

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

    def getEncodingName(self) -> str:
        """Get default tiktoken encoding for token counting."""
        return cast(str, self._data["defaults"]["encoding_name"])

    def getRepetitionMinK(self) -> int:
        """Get default minimum phrase length for repetition detection."""
        return cast(int, self._data["defaults"]["repetition_min_k"])

    def getRepetitionMinRepetitions(self) -> int:
        """Get default minimum consecutive repetitions for detection."""
        return cast(int, self._data["defaults"]["repetition_min_repetitions"])

    def getDetectMinK(self) -> int:
        """Get default min_k for detect() method."""
        return cast(int, self._data["defaults"]["detect_min_k"])

    def getHallucinationClassifierModel(self) -> tuple[float, float, float]:
        """Get SVM model coefficients for hallucination classification.

        Returns:
            Tuple of (coef_repetitions, coef_sequence_length, intercept)
        """
        hc = self._data["hallucination_detection"]
        return (
            hc["coef_repetitions"],
            hc["coef_sequence_length"],
            hc["intercept"],
        )

    def getConfigPath(self) -> Path:
        """Get the path to config.yaml."""
        return self.config_path

    def getDataDir(self) -> Path:
        """Get the data directory path.

        Returns:
            Path object pointing to the data directory (relative or absolute).
        """
        return Path(self._paths.data_dir)

    def getDataDownloadsDir(self) -> Path:
        """Get the data downloads directory path.

        Returns:
            Path object pointing to the data/downloads directory.
        """
        return Path(self._paths.data_downloads_dir)

    def getDataDownloadsVideosDir(self) -> Path:
        """Get the data downloads videos directory path.

        Returns:
            Path object pointing to the data/downloads/videos directory.
        """
        return Path(self._paths.data_downloads_videos_dir)

    def getDataDownloadsTranscriptsDir(self) -> Path:
        """Get the data downloads transcripts directory path.

        Returns:
            Path object pointing to the data/downloads/transcripts directory.
        """
        return Path(self._paths.data_downloads_transcripts_dir)

    def getDataDownloadsTranscriptsHallucinationsDir(self) -> Path:
        """Get the data downloads transcripts hallucinations directory path.

        Returns:
            Path object pointing to the data/downloads/transcripts-hallucinations directory.
        """
        return Path(self._paths.data_downloads_transcripts_hallucinations_dir)

    def getDataDownloadsTranscriptsCleanedDir(self) -> Path:
        """Get the data downloads cleaned transcripts directory path.

        Returns:
            Path object pointing to the data/downloads/transcripts_cleaned directory.
        """
        return Path(self._paths.data_downloads_transcripts_cleaned_dir)

    def getDataTranscriptsTopicsDir(self) -> Path:
        """Get the transcripts split by topics directory path.

        Returns:
            Path object pointing to the data/downloads/transcripts-topics directory.
        """
        return Path(self._paths.data_transcripts_topics_dir)

    def getDataDownloadsAudioDir(self) -> Path:
        """Get the data downloads audio directory path.

        Returns:
            Path object pointing to the data/downloads/audio directory.
        """
        return Path(self._paths.data_downloads_audio_dir)

    def getDataDownloadsMetadataDir(self) -> Path:
        """Get the data downloads metadata directory path.

        Returns:
            Path object pointing to the data/downloads/metadata directory.
        """
        return Path(self._paths.data_downloads_metadata_dir)

    def getDataOutputDir(self) -> Path:
        """Get the data output directory path.

        Returns:
            Path object pointing to the data/output directory.
        """
        return Path(self._paths.data_output_dir)

    def getDataInputDir(self) -> Path:
        """Get the data input directory path.

        Returns:
            Path object pointing to the data/input directory.
        """
        return Path(self._paths.data_input_dir)

    def getDataTempDir(self) -> Path:
        """Get the data temp directory path.

        Returns:
            Path object pointing to the data/temp directory.
        """
        return Path(self._paths.data_temp_dir)

    def getDataArchiveDir(self) -> Path:
        """Get the data archive directory path.

        Returns:
            Path object pointing to the data/archive directory.
        """
        return Path(self._paths.data_archive_dir)

    def getDataArchiveVideosDir(self) -> Path:
        """Get the data archive videos directory path.

        Returns:
            Path object pointing to the data/archive/videos directory.
        """
        return Path(self._paths.data_archive_videos_dir)

    def getDataLogsDir(self) -> Path:
        """Get the logs directory path.

        Returns:
            Path object pointing to the logs directory.
        """
        return Path(self._paths.data_logs_dir)

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
