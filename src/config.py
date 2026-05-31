"""Configuration loader for the Agentic News Generator."""

from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


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
    transcription_limiter: int = Field(
        ...,
        alias="transcription-limiter",
        ge=0,
        description="Maximum number of new transcriptions to perform per run for this channel",
    )
    language: str = Field(
        ...,
        description="ISO language code for channel content (e.g., 'en', 'de', 'ja').",
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, language: str) -> str:
        """Validate that the language code is supported by Whisper.

        Args:
            language: Language code to validate.

        Returns:
            The validated language code.

        Raises:
            ValueError: If the language code is not supported by Whisper.
        """
        from src.util.whisper_languages import WhisperLanguages

        supported = WhisperLanguages.get_supported_languages()

        if language not in supported:
            raise ValueError(
                f"Invalid language code: '{language}'. "
                f"Must be one of Whisper's supported languages. "
                f"Supported codes: {', '.join(sorted(supported.keys()))}"
            )

        return language

    model_config = ConfigDict(frozen=True, extra="forbid")


class LLMConfig(BaseModel):
    """Configuration for an individual LLM connection via litellm."""

    model: str = Field(
        ...,
        description="litellm model string (e.g., 'openai/gpt-4', 'anthropic/claude-3-5-sonnet', 'openai/local-model' for LM Studio)",
    )
    api_base: str | None = Field(..., description="API base URL (for LM Studio or custom endpoints, None for standard providers)")
    api_key: str = Field(..., description="API key for the LLM service")
    context_window: int = Field(..., description="Model context window size in tokens")
    max_tokens: int = Field(..., description="Maximum tokens for response/completion")
    temperature: float = Field(..., description="Sampling temperature (0.0-1.0)")
    context_window_threshold: int = Field(
        ...,
        description="Percentage threshold (0-100) for context window usage before raising an error",
        ge=0,
        le=100,
    )
    max_retries: int = Field(..., gt=0, description="Number of retries on LLM parsing errors")
    retry_delay: float = Field(..., gt=0, description="Delay in seconds between retries")

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicSegmentationConfig(BaseModel):
    """Configuration for topic segmentation agent system."""

    agent_llm: LLMConfig = Field(..., description="LLM config for segmentation agent")
    critic_llm: LLMConfig = Field(..., description="LLM config for critic agent")
    retry_limit: int = Field(..., description="Maximum retry attempts")

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicsExperimentConfig(BaseModel):
    """Configuration for the experimental LLM topic segmentation script."""

    api_base: str = Field(..., description="LM Studio OpenAI-compatible base URL")
    models_endpoint: str = Field(..., description="LM Studio native /api/v0/models endpoint URL")
    api_key: str = Field(..., description="API key for LM Studio (any non-empty string)")
    models_http_timeout: float = Field(..., gt=0, description="HTTP timeout in seconds for the models endpoint call")
    output_subdir: str = Field(..., min_length=1, description="Sub-directory under data_output_dir for experiment JSON files")
    temperature: float = Field(..., ge=0.0, description="Sampling temperature for segmentation calls")
    max_output_tokens: int = Field(..., gt=0, description="Maximum tokens reserved for LLM response")
    token_safety_factor: float = Field(..., ge=1.0, description="Multiplier applied to measured input tokens")
    prompt_overhead_tokens: int = Field(..., ge=0, description="Extra tokens reserved for system prompt and wrapping")
    prefer_loaded: bool = Field(..., description="Prefer models already loaded in LM Studio when selecting")
    max_retries: int = Field(..., gt=0, description="Number of LLM retries on parse/validation errors")
    retry_delay: float = Field(..., gt=0, description="Delay in seconds between retries")

    model_config = ConfigDict(frozen=True, extra="forbid")


class SummarizeTranscriptsConfig(BaseModel):
    """Configuration for transcript summarization via LLM."""

    llm: LLMConfig = Field(..., description="LLM config for transcript summarization")
    skip_transcripts_above_context_window_pct: int = Field(
        ...,
        ge=0,
        le=100,
        description="Skip summarization when transcript token count exceeds this percentage of the model context window",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


class AgenticReviewLLMConfig(BaseModel):
    """LLM connection settings for agentic review tools."""

    model: str = Field(..., min_length=1, description="Model name in LM Studio")
    base_url: str = Field(..., min_length=1, description="LM Studio OpenAI-compatible base URL")
    api_key: str = Field(..., min_length=1, description="API key (any non-empty string for local LLMs)")

    model_config = ConfigDict(frozen=True, extra="forbid")


class AgenticReviewConfig(BaseModel):
    """Configuration for agentic code review tools using a local LLM via Autogen."""

    llm: AgenticReviewLLMConfig = Field(..., description="LLM connection settings")

    model_config = ConfigDict(frozen=True, extra="forbid")


class UrlCleanContentConfig(BaseModel):
    """Configuration for URL raw-content Markdown formatting via LLM."""

    llm: LLMConfig = Field(..., description="LLM config for URL content formatting")
    prompt_template: str = Field(..., min_length=1, description="Prompt template path relative to project root")
    skip_documents_above_context_window_pct: int = Field(
        ...,
        ge=0,
        le=100,
        description="Skip URL documents when prompt token count exceeds this percentage of the model context window",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicsConfig(BaseModel):
    """Configuration for topic extraction from hallucination-freed SRT transcripts."""

    input_dir: str = Field(..., min_length=1, description="Input directory of hallucination-freed SRT files (relative to data_dir)")
    output_dir: str = Field(..., min_length=1, description="Output directory for topic extraction results (relative to data_dir)")
    codex_model: str = Field(..., min_length=1, description="Codex model identifier passed to `codex exec -m`")
    prompt_template: str = Field(
        ...,
        min_length=1,
        description="Prompt template path (relative to project root) with {{INPUT_SRT_FILE}} and {{OUTPUT_JSON_FILE}} placeholders",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


class UrlProcessingConfig(BaseModel):
    """Configuration for URL ingestion pipeline folders relative to data_dir."""

    base_dir: str = Field(..., min_length=1, description="Base URL ingestion directory relative to data_dir")
    inbox_dir: str = Field(..., min_length=1, description="URL inbox directory relative to base_dir")
    raw_dir: str = Field(..., min_length=1, description="Raw downloaded URL data directory relative to base_dir")
    cleaned_dir: str = Field(..., min_length=1, description="Cleaned URL content directory relative to base_dir")
    test_base_dir: str = Field(..., min_length=1, description="Test URL ingestion directory relative to data_dir")

    @field_validator("base_dir", "inbox_dir", "raw_dir", "cleaned_dir", "test_base_dir")
    @classmethod
    def validate_relative_path(cls, path_value: str) -> str:
        """Validate URL processing paths are relative folder fragments."""
        if Path(path_value).is_absolute():
            raise ValueError("URL processing paths must be relative to data_dir")
        return path_value

    model_config = ConfigDict(frozen=True, extra="forbid")


class TranscriptionConfig(BaseModel):
    """Configuration for MLX Whisper transcription."""

    model_en_name: str = Field(..., description="English-only Whisper model name")
    model_en_repo: str = Field(..., description="English-only Whisper model repository")
    model_multi_name: str = Field(..., description="Multilingual Whisper model name")
    model_multi_repo: str = Field(..., description="Multilingual Whisper model repository")
    hallucination_silence_threshold: float = Field(
        ...,
        description="Silence threshold in seconds for hallucination detection",
    )
    compression_ratio_threshold: float = Field(
        ...,
        description="Compression ratio threshold for hallucination detection",
    )
    use_youtube_metadata: bool = Field(
        ...,
        description="Whether to use YouTube video metadata in transcription prompts",
    )
    description_max_length: int = Field(
        ...,
        ge=0,
        description="Maximum characters to include from video description",
    )
    sleep_between_files: int = Field(
        ...,
        ge=0,
        description="Seconds to pause between transcribing files",
    )
    metadata_video_subdir: str = Field(
        ...,
        description="Subdirectory name for video metadata (.info.json files)",
    )
    verbose: bool = Field(
        ...,
        description="Enable verbose output during transcription",
    )
    min_duration: int = Field(
        ...,
        ge=0,
        description="Minimum video duration in seconds to transcribe (shorter videos are skipped)",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


class RaindropIoConfig(BaseModel):
    """Configuration for the Raindrop.io integration."""

    token: str = Field(..., min_length=1, description="Raindrop.io API token")

    model_config = ConfigDict(frozen=True, extra="forbid")


class IntegrationsConfig(BaseModel):
    """Configuration for external service integrations."""

    raindrop_io: RaindropIoConfig = Field(..., description="Raindrop.io integration configuration")

    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsConfig(BaseModel):
    """Configuration for all project directory paths."""

    data_dir: str = Field(..., description="Root data directory path", min_length=1)
    data_models_dir: str = Field(..., description="ML models directory path", min_length=1)
    data_downloads_dir: str = Field(..., description="Downloads directory path", min_length=1)
    data_downloads_videos_dir: str = Field(..., description="Downloaded videos directory path", min_length=1)
    data_downloads_transcripts_dir: str = Field(..., description="Transcripts directory path", min_length=1)
    data_downloads_transcripts_hallucinations_dir: str = Field(
        ..., description="Transcript hallucinations analysis directory path", min_length=1
    )
    data_downloads_transcripts_cleaned_dir: str = Field(..., description="Cleaned transcripts directory path", min_length=1)
    data_downloads_transcripts_summaries_dir: str = Field(..., description="Transcript summaries directory path", min_length=1)
    data_downloads_audio_dir: str = Field(..., description="Audio files directory path", min_length=1)
    data_downloads_metadata_dir: str = Field(..., description="Metadata files directory path", min_length=1)
    data_output_dir: str = Field(..., description="Output directory path", min_length=1)
    data_input_dir: str = Field(..., description="Input directory path", min_length=1)
    data_temp_dir: str = Field(..., description="Temporary files directory path", min_length=1)
    data_archive_dir: str = Field(..., description="Archive directory path", min_length=1)
    data_archive_videos_dir: str = Field(..., description="Archived videos directory path", min_length=1)
    data_logs_dir: str = Field(..., description="Log files directory path", min_length=1)
    reports_dir: str = Field(..., description="Model benchmark reports directory path", min_length=1)

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
        self._paths = self._validate_paths()
        self._validate_optional_sections()

    def _validate_optional_sections(self) -> None:
        """Validate all optional config sections that are present."""
        self._validate_pipeline_sections()
        self._validate_tool_sections()

    def _validate_pipeline_sections(self) -> None:
        """Validate optional pipeline-related config sections."""
        if "topic_segmentation" in self._data:
            self._topic_segmentation = self._validate_topic_segmentation()
        if "transcription" in self._data:
            self._transcription = self._validate_transcription()
        if "topics_experiment" in self._data:
            self._topics_experiment = self._validate_topics_experiment()
        if "summarize_transcripts" in self._data:
            self._summarize_transcripts = self._validate_summarize_transcripts()
        if "url_clean_content" in self._data:
            self._url_clean_content = self._validate_url_clean_content()
        if "topics" in self._data:
            self._topics = self._validate_topics()
        if "url_processing" in self._data:
            self._url_processing = self._validate_url_processing()

    def _validate_tool_sections(self) -> None:
        """Validate optional tool-related config sections."""
        if "agentic_unit_test_reviews" in self._data:
            self._agentic_unit_test_reviews = self._validate_agentic_review("agentic_unit_test_reviews")
        if "agentic_shell_script_reviews" in self._data:
            self._agentic_shell_script_reviews = self._validate_agentic_review("agentic_shell_script_reviews")
        if "integrations" in self._data:
            self._integrations = self._validate_integrations()

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

    def _validate_transcription(self) -> TranscriptionConfig:
        """Validate transcription configuration.

        Returns:
            Validated TranscriptionConfig instance.

        Raises:
            ValueError: If transcription configuration is invalid.
        """
        try:
            return TranscriptionConfig.model_validate(self._data["transcription"])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"Transcription configuration validation failed: {error_messages}") from e

    def _validate_topics_experiment(self) -> TopicsExperimentConfig:
        """Validate topics_experiment configuration.

        Raises:
            ValueError: If topics_experiment configuration is invalid.
        """
        try:
            return TopicsExperimentConfig.model_validate(self._data["topics_experiment"])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"Topics experiment configuration validation failed: {error_messages}") from e

    def _validate_summarize_transcripts(self) -> SummarizeTranscriptsConfig:
        """Validate summarize_transcripts configuration."""
        try:
            return SummarizeTranscriptsConfig.model_validate(self._data["summarize_transcripts"])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"Summarize transcripts configuration validation failed: {error_messages}") from e

    def _validate_url_clean_content(self) -> UrlCleanContentConfig:
        """Validate url_clean_content configuration."""
        try:
            return UrlCleanContentConfig.model_validate(self._data["url_clean_content"])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"URL clean content configuration validation failed: {error_messages}") from e

    def _validate_topics(self) -> TopicsConfig:
        """Validate topics configuration."""
        try:
            return TopicsConfig.model_validate(self._data["topics"])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"Topics configuration validation failed: {error_messages}") from e

    def _validate_url_processing(self) -> UrlProcessingConfig:
        """Validate URL processing configuration."""
        try:
            return UrlProcessingConfig.model_validate(self._data["url_processing"])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"URL processing configuration validation failed: {error_messages}") from e

    def _validate_agentic_review(self, key: str) -> AgenticReviewConfig:
        """Validate an agentic review configuration section."""
        try:
            return AgenticReviewConfig.model_validate(self._data[key])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"{key} configuration validation failed: {error_messages}") from e

    def _validate_integrations(self) -> IntegrationsConfig:
        """Validate integrations configuration."""
        try:
            return IntegrationsConfig.model_validate(self._data["integrations"])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"Integrations configuration validation failed: {error_messages}") from e

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

    def get_paths_config(self) -> PathsConfig:
        """Get the validated paths configuration model."""
        return self._paths

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

    def get_topics_experiment_config(self) -> TopicsExperimentConfig:
        """Get topics_experiment configuration.

        Raises:
            KeyError: If topics_experiment section is not configured.
        """
        if not hasattr(self, "_topics_experiment"):
            raise KeyError("Missing required key 'topics_experiment' in config file")
        return self._topics_experiment

    def get_summarize_transcripts_config(self) -> SummarizeTranscriptsConfig:
        """Get summarize_transcripts configuration."""
        if not hasattr(self, "_summarize_transcripts"):
            raise KeyError("Missing required key 'summarize_transcripts' in config file")
        return self._summarize_transcripts

    def get_url_clean_content_config(self) -> UrlCleanContentConfig:
        """Get URL clean-content configuration."""
        if not hasattr(self, "_url_clean_content"):
            raise KeyError("Missing required key 'url_clean_content' in config file")
        return self._url_clean_content

    def get_topics_config(self) -> TopicsConfig:
        """Get topics configuration."""
        if not hasattr(self, "_topics"):
            raise KeyError("Missing required key 'topics' in config file")
        return self._topics

    def get_url_processing_config(self) -> UrlProcessingConfig:
        """Get URL processing configuration."""
        if not hasattr(self, "_url_processing"):
            raise KeyError("Missing required key 'url_processing' in config file")
        return self._url_processing

    def get_agentic_unit_test_reviews_config(self) -> AgenticReviewConfig:
        """Get agentic unit test reviews configuration."""
        if not hasattr(self, "_agentic_unit_test_reviews"):
            raise KeyError("Missing required key 'agentic_unit_test_reviews' in config file")
        return self._agentic_unit_test_reviews

    def get_agentic_shell_script_reviews_config(self) -> AgenticReviewConfig:
        """Get agentic shell script reviews configuration."""
        if not hasattr(self, "_agentic_shell_script_reviews"):
            raise KeyError("Missing required key 'agentic_shell_script_reviews' in config file")
        return self._agentic_shell_script_reviews

    def get_integrations_config(self) -> IntegrationsConfig:
        """Get integrations configuration."""
        if not hasattr(self, "_integrations"):
            raise KeyError("Missing required key 'integrations' in config file")
        return self._integrations

    def get_raindrop_token(self) -> str:
        """Get the Raindrop.io API token."""
        return self.get_integrations_config().raindrop_io.token

    def get_encoding_name(self) -> str:
        """Get default tiktoken encoding for token counting."""
        return cast(str, self._data["defaults"]["encoding_name"])

    def get_repetition_min_k(self) -> int:
        """Get default minimum phrase length for repetition detection."""
        return cast(int, self._data["defaults"]["repetition_min_k"])

    def get_repetition_min_repetitions(self) -> int:
        """Get default minimum consecutive repetitions for detection."""
        return cast(int, self._data["defaults"]["repetition_min_repetitions"])

    def get_detect_min_k(self) -> int:
        """Get default min_k for detect() method."""
        return cast(int, self._data["defaults"]["detect_min_k"])

    def get_hallucination_classifier_model(self) -> tuple[float, float, float]:
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

    def get_language_analysis_min_confidence(self) -> float:
        """Get minimum FastText confidence for a non-English classification to be trusted."""
        return cast(float, self._data["language_analysis"]["min_detection_confidence"])

    def get_config_path(self) -> Path:
        """Get the path to config.yaml."""
        return self.config_path

    def get_data_dir(self) -> Path:
        """Get the data directory path.

        Returns:
            Path object pointing to the data directory (relative or absolute).
        """
        return Path(self._paths.data_dir)

    def get_data_models_dir(self) -> Path:
        """Get the data models directory path.

        Returns:
            Path object for the models directory (e.g., ./data/models/).
        """
        return Path(self._paths.data_models_dir)

    def get_data_downloads_dir(self) -> Path:
        """Get the data downloads directory path.

        Returns:
            Path object pointing to the data/downloads directory.
        """
        return Path(self._paths.data_downloads_dir)

    def get_data_downloads_videos_dir(self) -> Path:
        """Get the data downloads videos directory path.

        Returns:
            Path object pointing to the data/downloads/videos directory.
        """
        return Path(self._paths.data_downloads_videos_dir)

    def get_data_downloads_transcripts_dir(self) -> Path:
        """Get the data downloads transcripts directory path.

        Returns:
            Path object pointing to the data/downloads/transcripts directory.
        """
        return Path(self._paths.data_downloads_transcripts_dir)

    def get_data_downloads_transcripts_hallucinations_dir(self) -> Path:
        """Get the data downloads transcripts hallucinations directory path.

        Returns:
            Path object pointing to the data/downloads/transcripts-hallucinations directory.
        """
        return Path(self._paths.data_downloads_transcripts_hallucinations_dir)

    def get_data_downloads_transcripts_cleaned_dir(self) -> Path:
        """Get the data downloads cleaned transcripts directory path.

        Returns:
            Path object pointing to the data/downloads/transcripts_cleaned directory.
        """
        return Path(self._paths.data_downloads_transcripts_cleaned_dir)

    def get_data_downloads_transcripts_summaries_dir(self) -> Path:
        """Get the transcript summaries directory path."""
        return Path(self._paths.data_downloads_transcripts_summaries_dir)

    def get_data_downloads_audio_dir(self) -> Path:
        """Get the data downloads audio directory path.

        Returns:
            Path object pointing to the data/downloads/audio directory.
        """
        return Path(self._paths.data_downloads_audio_dir)

    def get_data_downloads_metadata_dir(self) -> Path:
        """Get the data downloads metadata directory path.

        Returns:
            Path object pointing to the data/downloads/metadata directory.
        """
        return Path(self._paths.data_downloads_metadata_dir)

    def get_data_output_dir(self) -> Path:
        """Get the data output directory path.

        Returns:
            Path object pointing to the data/output directory.
        """
        return Path(self._paths.data_output_dir)

    def get_data_input_dir(self) -> Path:
        """Get the data input directory path.

        Returns:
            Path object pointing to the data/input directory.
        """
        return Path(self._paths.data_input_dir)

    def get_data_temp_dir(self) -> Path:
        """Get the data temp directory path.

        Returns:
            Path object pointing to the data/temp directory.
        """
        return Path(self._paths.data_temp_dir)

    def get_data_archive_dir(self) -> Path:
        """Get the data archive directory path.

        Returns:
            Path object pointing to the data/archive directory.
        """
        return Path(self._paths.data_archive_dir)

    def get_data_archive_videos_dir(self) -> Path:
        """Get the data archive videos directory path.

        Returns:
            Path object pointing to the data/archive/videos directory.
        """
        return Path(self._paths.data_archive_videos_dir)

    def get_data_logs_dir(self) -> Path:
        """Get the data logs directory path.

        Returns:
            Path object pointing to the data/logs directory.
        """
        return Path(self._paths.data_logs_dir)

    def get_reports_dir(self) -> Path:
        """Get the reports directory path.

        Returns:
            Path object pointing to the reports directory.
        """
        return Path(self._paths.reports_dir)

    def get_url_base_dir(self) -> Path:
        """Get the URL ingestion base directory under data_dir."""
        return self.get_data_dir() / self.get_url_processing_config().base_dir

    def get_url_inbox_dir(self) -> Path:
        """Get the URL inbox directory under the URL ingestion base directory."""
        return self.get_url_base_dir() / self.get_url_processing_config().inbox_dir

    def get_url_raw_dir(self) -> Path:
        """Get the URL raw data directory under the URL ingestion base directory."""
        return self.get_url_base_dir() / self.get_url_processing_config().raw_dir

    def get_url_cleaned_dir(self) -> Path:
        """Get the URL cleaned content directory under the URL ingestion base directory."""
        return self.get_url_base_dir() / self.get_url_processing_config().cleaned_dir

    def get_url_test_base_dir(self) -> Path:
        """Get the URL ingestion test base directory under data_dir."""
        return self.get_data_dir() / self.get_url_processing_config().test_base_dir

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

    # Transcription Configuration Getters

    def get_transcription_model_en_name(self) -> str:
        """Get English Whisper model name.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.model_en_name

    def get_transcription_model_en_repo(self) -> str:
        """Get English Whisper model repository.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.model_en_repo

    def get_transcription_model_multi_name(self) -> str:
        """Get multilingual Whisper model name.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.model_multi_name

    def get_transcription_model_multi_repo(self) -> str:
        """Get multilingual Whisper model repository.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.model_multi_repo

    def get_transcription_hallucination_silence_threshold(self) -> float:
        """Get hallucination silence threshold in seconds.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.hallucination_silence_threshold

    def get_transcription_compression_ratio_threshold(self) -> float:
        """Get compression ratio threshold.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.compression_ratio_threshold

    def get_transcription_use_youtube_metadata(self) -> bool:
        """Get whether to use YouTube metadata in prompts.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.use_youtube_metadata

    def get_transcription_metadata_video_subdir(self) -> str:
        """Get metadata video subdirectory name.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.metadata_video_subdir

    def get_transcription_verbose(self) -> bool:
        """Get verbose output flag.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.verbose

    def get_transcription_description_max_length(self) -> int:
        """Get maximum description length in characters.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.description_max_length

    def get_transcription_sleep_between_files(self) -> int:
        """Get sleep duration between files in seconds.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.sleep_between_files

    def get_transcription_min_duration(self) -> int:
        """Get minimum video duration in seconds for transcription.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.min_duration
