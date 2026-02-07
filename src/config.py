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
    max_retries: int = Field(..., ge=0, description="Number of retries after an LLM request failure")
    retry_delay: float = Field(..., gt=0, description="Delay in seconds between retries")
    timeout_seconds: int = Field(..., gt=0, description="Request timeout in seconds")

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicSegmentationConfig(BaseModel):
    """Configuration for topic segmentation agent system."""

    agent_llm: LLMConfig = Field(..., description="LLM config for segmentation agent")
    critic_llm: LLMConfig = Field(..., description="LLM config for critic agent")
    retry_limit: int = Field(..., description="Maximum retry attempts")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleGenerationOutputConfig(BaseModel):
    """Output paths for canonical and run artifacts."""

    final_articles_dir: str = Field(..., min_length=1, description="Canonical output root directory")
    run_artifacts_dir: str = Field(..., min_length=1, description="Run artifacts root directory")
    save_intermediate_results: bool = Field(..., description="Whether to persist iteration artifacts")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleGenerationPromptConfig(BaseModel):
    """Prompt file configuration for editor system."""

    root_dir: str = Field(..., min_length=1, description="Prompt templates root directory")
    writer_prompt_file: str = Field(..., min_length=1, description="Writer prompt filename")
    revision_prompt_file: str = Field(..., min_length=1, description="Revision prompt filename")
    article_review_prompt_file: str = Field(..., min_length=1, description="Article-review prompt filename")
    concern_mapping_prompt_file: str = Field(..., min_length=1, description="Concern-mapping prompt filename")
    specialists_dir: str = Field(..., min_length=1, description="Specialists prompt subdirectory")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleGenerationEditorConfig(BaseModel):
    """Chief editor settings."""

    editor_max_rounds: int = Field(..., gt=0, description="Maximum editor feedback rounds")
    output: ArticleGenerationOutputConfig = Field(..., description="Output handling configuration")
    prompts: ArticleGenerationPromptConfig = Field(..., description="Prompt file configuration")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleGenerationSpecialistLLMConfig(BaseModel):
    """Specialist LLM configurations."""

    fact_check_llm: LLMConfig = Field(..., description="Fact-check specialist LLM")
    evidence_finding_llm: LLMConfig = Field(..., description="Evidence-finding specialist LLM")
    opinion_llm: LLMConfig = Field(..., description="Opinion specialist LLM")
    attribution_llm: LLMConfig = Field(..., description="Attribution specialist LLM")
    style_review_llm: LLMConfig = Field(..., description="Style-review specialist LLM")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleGenerationAgentsConfig(BaseModel):
    """LLM configuration for all article-generation agents."""

    writer_llm: LLMConfig = Field(..., description="Writer agent LLM")
    article_review_llm: LLMConfig = Field(..., description="Article-review agent LLM")
    concern_mapping_llm: LLMConfig = Field(..., description="Concern-mapping agent LLM")
    specialists: ArticleGenerationSpecialistLLMConfig = Field(..., description="Specialist agent LLM settings")

    model_config = ConfigDict(frozen=True, extra="forbid")


class KnowledgeBaseEmbeddingConfig(BaseModel):
    """Embedding config for knowledge base indexing."""

    provider: str = Field(..., min_length=1, description="Embedding provider")
    model_name: str = Field(..., min_length=1, description="Embedding model name")
    api_base: str | None = Field(..., description="Embedding API base URL")
    api_key: str = Field(..., min_length=1, description="Embedding API key")
    timeout_seconds: int = Field(..., gt=0, description="Embedding timeout in seconds")

    model_config = ConfigDict(frozen=True, extra="forbid")


class KnowledgeBaseConfig(BaseModel):
    """Knowledge base indexing and retrieval settings."""

    data_dir: str = Field(..., min_length=1, description="Knowledge base source docs directory")
    index_dir: str = Field(..., min_length=1, description="Persisted index directory")
    chunk_size_tokens: int = Field(..., gt=0, description="Chunk size in tokens")
    chunk_overlap_tokens: int = Field(..., ge=0, description="Chunk overlap in tokens")
    timeout_seconds: int = Field(..., gt=0, description="Knowledge base search timeout in seconds")
    embedding: KnowledgeBaseEmbeddingConfig = Field(..., description="Embedding settings")

    model_config = ConfigDict(frozen=True, extra="forbid")


class PerplexityConfig(BaseModel):
    """Perplexity API settings."""

    api_base: str = Field(..., min_length=1, description="Perplexity API base URL")
    api_key: str = Field(..., min_length=1, description="Perplexity API key")
    model: str = Field(..., min_length=1, description="Perplexity model name")
    timeout_seconds: int = Field(..., gt=0, description="Perplexity timeout in seconds")

    model_config = ConfigDict(frozen=True, extra="forbid")


class InstitutionalMemoryConfig(BaseModel):
    """Institutional memory storage settings."""

    data_dir: str = Field(..., min_length=1, description="Institutional memory root directory")
    fact_checking_subdir: str = Field(..., min_length=1, description="Fact-check cache subdirectory")
    evidence_finding_subdir: str = Field(..., min_length=1, description="Evidence-finding cache subdirectory")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleGenerationConfig(BaseModel):
    """Configuration for the multi-agent article-generation system."""

    editor: ArticleGenerationEditorConfig = Field(..., description="Chief editor settings")
    agents: ArticleGenerationAgentsConfig = Field(..., description="Agent LLM settings")
    knowledge_base: KnowledgeBaseConfig = Field(..., description="Knowledge base settings")
    perplexity: PerplexityConfig = Field(..., description="Perplexity API settings")
    institutional_memory: InstitutionalMemoryConfig = Field(..., description="Institutional memory settings")
    allowed_styles: list[str] = Field(..., min_length=1, description="Allowed article style values")
    default_style_mode: str = Field(..., min_length=1, description="Default style mode")

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicDetectionEmbeddingConfig(BaseModel):
    """Configuration for topic detection embedding generation."""

    provider: str = Field(..., description="Embedding provider type (e.g., 'lmstudio')")
    model_name: str = Field(..., description="Model name for embedding generation")
    api_base: str | None = Field(..., description="API base URL for the embedding service")
    api_key: str | None = Field(..., description="API key for the embedding service (use 'lm-studio' for local LM Studio)")

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicDetectionSlidingWindowConfig(BaseModel):
    """Configuration for topic detection sliding window segmenter."""

    window_size: int = Field(..., gt=0, description="Number of words per chunk for embedding")
    stride: int = Field(..., gt=0, description="Number of words to advance between chunks")
    threshold_method: str = Field(..., description="Method for determining boundaries ('relative', 'absolute', 'percentile')")
    threshold_value: float = Field(..., description="Threshold value for boundary detection")
    smoothing_passes: int = Field(..., ge=0, description="Number of smoothing passes on similarity curve")

    model_config = ConfigDict(frozen=True, extra="forbid")


class TopicDetectionConfig(BaseModel):
    """Complete topic detection configuration."""

    embedding: TopicDetectionEmbeddingConfig = Field(..., description="Embedding generator configuration")
    sliding_window: TopicDetectionSlidingWindowConfig = Field(..., description="Sliding window segmenter configuration")
    topic_detection_llm: LLMConfig = Field(..., description="LLM config for topic extraction agent")
    output_dir: str = Field(..., description="Output directory for topic detection results")

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
    data_transcripts_topics_dir: str = Field(..., description="Transcripts split by topics directory path", min_length=1)
    data_downloads_audio_dir: str = Field(..., description="Audio files directory path", min_length=1)
    data_downloads_metadata_dir: str = Field(..., description="Metadata files directory path", min_length=1)
    data_output_dir: str = Field(..., description="Output directory path", min_length=1)
    data_input_dir: str = Field(..., description="Input directory path", min_length=1)
    data_temp_dir: str = Field(..., description="Temporary files directory path", min_length=1)
    data_archive_dir: str = Field(..., description="Archive directory path", min_length=1)
    data_archive_videos_dir: str = Field(..., description="Archived videos directory path", min_length=1)
    data_logs_dir: str = Field(..., description="Logs directory path", min_length=1)
    data_output_articles_dir: str = Field(..., description="Generated articles directory path", min_length=1)
    data_articles_input_dir: str = Field(..., description="Article generation input bundles directory path", min_length=1)
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

        # Validate paths section (required)
        self._paths = self._validate_paths()

        # Validate topic_segmentation section if present
        if "topic_segmentation" in self._data:
            self._topic_segmentation = self._validate_topic_segmentation()

        # Validate article_generation section if present
        if "article_generation" in self._data:
            self._article_generation = self._validate_article_generation()

        # Validate transcription section if present
        if "transcription" in self._data:
            self._transcription = self._validate_transcription()

        # Validate topic_detection section if present
        if "topic_detection" in self._data:
            self._topic_detection = self._validate_topic_detection()

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

    def _validate_article_generation(self) -> ArticleGenerationConfig:
        """Validate article generation configuration.

        Returns:
            Validated ArticleGenerationConfig instance.

        Raises:
            ValueError: If article generation configuration is invalid.
        """
        try:
            return ArticleGenerationConfig.model_validate(self._data["article_generation"])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"Article generation configuration validation failed: {error_messages}") from e

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

    def _validate_topic_detection(self) -> TopicDetectionConfig:
        """Validate topic detection configuration.

        Returns:
            Validated TopicDetectionConfig instance.

        Raises:
            ValueError: If topic detection configuration is invalid.
        """
        try:
            return TopicDetectionConfig.model_validate(self._data["topic_detection"])
        except ValidationError as e:
            error_messages = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"Topic detection configuration validation failed: {error_messages}") from e

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

    def get_article_generation_config(self) -> ArticleGenerationConfig:
        """Get article generation configuration.

        Returns:
            ArticleGenerationConfig instance.

        Raises:
            KeyError: If article_generation section is not configured.
        """
        if not hasattr(self, "_article_generation"):
            raise KeyError("Missing required key 'article_generation' in config file")
        return self._article_generation

    def get_allowed_article_styles(self) -> list[str]:
        """Get list of allowed article writing styles.

        Returns:
            List of allowed style strings (e.g., ["NATURE_NEWS", "SCIAM_MAGAZINE"]).

        Raises:
            KeyError: If article_generation section is not configured.
        """
        if not hasattr(self, "_article_generation"):
            raise KeyError("Missing required key 'article_generation' in config file")
        return self._article_generation.allowed_styles

    def get_article_timeout_seconds(self) -> int:
        """Get writer-agent timeout for article generation in seconds.

        Returns:
            Timeout in seconds.

        Raises:
            KeyError: If article_generation section is not configured.
        """
        if not hasattr(self, "_article_generation"):
            raise KeyError("Missing required key 'article_generation' in config file")
        return self._article_generation.agents.writer_llm.timeout_seconds

    def get_article_editor_max_rounds(self) -> int:
        """Get max editor review rounds for article generation."""
        if not hasattr(self, "_article_generation"):
            raise KeyError("Missing required key 'article_generation' in config file")
        return self._article_generation.editor.editor_max_rounds

    def get_topic_detection_config(self) -> TopicDetectionConfig:
        """Get topic detection configuration.

        Returns:
            TopicDetectionConfig instance.

        Raises:
            KeyError: If topic_detection section is not configured.
        """
        if not hasattr(self, "_topic_detection"):
            raise KeyError("Missing required key 'topic_detection' in config file")
        return self._topic_detection

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

    def getDataModelsDir(self) -> Path:
        """Get the data models directory path.

        Returns:
            Path object for the models directory (e.g., ./data/models/).
        """
        return Path(self._paths.data_models_dir)

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

    def getDataOutputArticlesDir(self) -> Path:
        """Get the generated articles directory path.

        Returns:
            Path object pointing to the data/output/articles directory.
        """
        return Path(self._paths.data_output_articles_dir)

    def getDataArticlesInputDir(self) -> Path:
        """Get the article generation input bundles directory path."""
        return Path(self._paths.data_articles_input_dir)

    def getReportsDir(self) -> Path:
        """Get the reports directory path.

        Returns:
            Path object pointing to the reports directory.
        """
        return Path(self._paths.reports_dir)

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

    def getTranscriptionModelEnName(self) -> str:
        """Get English Whisper model name.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.model_en_name

    def getTranscriptionModelEnRepo(self) -> str:
        """Get English Whisper model repository.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.model_en_repo

    def getTranscriptionModelMultiName(self) -> str:
        """Get multilingual Whisper model name.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.model_multi_name

    def getTranscriptionModelMultiRepo(self) -> str:
        """Get multilingual Whisper model repository.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.model_multi_repo

    def getTranscriptionHallucinationSilenceThreshold(self) -> float:
        """Get hallucination silence threshold in seconds.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.hallucination_silence_threshold

    def getTranscriptionCompressionRatioThreshold(self) -> float:
        """Get compression ratio threshold.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.compression_ratio_threshold

    def getTranscriptionUseYoutubeMetadata(self) -> bool:
        """Get whether to use YouTube metadata in prompts.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.use_youtube_metadata

    def getTranscriptionMetadataVideoSubdir(self) -> str:
        """Get metadata video subdirectory name.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.metadata_video_subdir

    def getTranscriptionVerbose(self) -> bool:
        """Get verbose output flag.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.verbose

    def getTranscriptionDescriptionMaxLength(self) -> int:
        """Get maximum description length in characters.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.description_max_length

    def getTranscriptionSleepBetweenFiles(self) -> int:
        """Get sleep duration between files in seconds.

        Raises:
            KeyError: If transcription section is not configured.
        """
        if not hasattr(self, "_transcription"):
            raise KeyError("Missing required key 'transcription' in config file")
        return self._transcription.sleep_between_files
