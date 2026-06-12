"""Pydantic data contracts for the analytics package.

Summaries are treated as raw, unstructured text: the analytics never parses a
summary's internal structure or assumes which fields/labels it contains. Term
statistics come from TF-IDF over the raw summary text; nothing is extracted by
label or section.
"""

from pydantic import BaseModel, ConfigDict, Field


class VideoMetadata(BaseModel):
    """Video metadata extracted from a yt-dlp ``.info.json`` file."""

    title: str = Field(..., min_length=1, description="Video title")
    upload_date: str | None = Field(..., description="ISO upload date (YYYY-MM-DD); None when absent in metadata")
    description: str | None = Field(..., description="Video description; None when absent in metadata")
    video_id: str | None = Field(..., description="YouTube id from the metadata 'id' field; None when absent")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ChannelMeta(BaseModel):
    """Channel-level metadata copied from the channel configuration."""

    language: str = Field(..., description="ISO language code of the channel")
    category: str = Field(..., description="Channel category from config")
    description: str = Field(..., description="Channel description from config")

    model_config = ConfigDict(frozen=True, extra="forbid")


class RecordPaths(BaseModel):
    """Source file paths joined into one corpus record."""

    cleaned_txt: str = Field(..., min_length=1, description="Cleaned plain-text transcript path")
    cleaned_srt: str | None = Field(..., description="Cleaned SRT path; None when absent")
    summary_md: str | None = Field(..., description="Summary markdown path; None when absent")
    metadata_json: str = Field(..., min_length=1, description="yt-dlp info.json path")

    model_config = ConfigDict(frozen=True, extra="forbid")


class SummaryStats(BaseModel):
    """Size statistics of one summary's normalized text (no parsed content)."""

    word_count: int = Field(..., ge=0, description="Whitespace word count of the normalized summary text")
    char_count: int = Field(..., ge=0, description="Character count of the normalized summary text")

    model_config = ConfigDict(frozen=True, extra="forbid")


class CorpusRecord(BaseModel):
    """One cleaned transcript joined to its metadata and channel config.

    The summary, when present, is referenced by path only (``paths.summary_md``)
    plus size statistics — never parsed into fields. Ranking stages re-read and
    normalize the raw text from disk.
    """

    video_id: str = Field(..., min_length=1, description="YouTube video ID (not unique: re-downloads share it)")
    channel: str = Field(..., min_length=1, description="Sanitized channel directory name")
    title: str = Field(..., min_length=1, description="Video title from metadata")
    upload_date: str | None = Field(..., description="ISO upload date; None when absent in metadata")
    duration_seconds: int | None = Field(..., description="Duration from the last SRT entry; None without SRT")
    word_count: int = Field(..., ge=0, description="Whitespace word count of the cleaned txt")
    has_summary: bool = Field(..., description="Whether a summary file exists for this transcript")
    paths: RecordPaths = Field(..., description="Joined source file paths")
    channel_meta: ChannelMeta = Field(..., description="Channel metadata from config")
    summary_stats: SummaryStats | None = Field(..., description="Normalized-summary size stats; None without a summary")

    model_config = ConfigDict(frozen=True, extra="forbid")


class CorpusIndex(BaseModel):
    """The full joined corpus, sorted by (channel, upload_date, video_id)."""

    records: list[CorpusRecord] = Field(..., description="All corpus records in stable order")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ThemeVideo(BaseModel):
    """One contributing video reference attached to a theme entry."""

    video_id: str = Field(..., min_length=1, description="YouTube video ID")
    channel: str = Field(..., min_length=1, description="Sanitized channel directory name")
    title: str = Field(..., min_length=1, description="Video title from metadata")
    upload_date: str | None = Field(..., description="ISO upload date; None when absent in metadata")
    summary_md: str | None = Field(..., description="Summary markdown path for drill-down reading")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ThemeEntry(BaseModel):
    """One ranked TF-IDF feature (unigram term or multi-word phrase)."""

    term: str = Field(..., min_length=1, description="Vocabulary term or phrase")
    score: float = Field(..., ge=0.0, description="Mean TF-IDF score across the filtered corpus")
    document_frequency: int = Field(..., ge=1, description="Number of videos whose summary contains the feature")
    channels: list[str] = Field(..., description="Sorted unique channels of all contributing videos")
    videos: list[ThemeVideo] = Field(..., description="Top contributing videos by TF-IDF weight, capped")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ThemeReport(BaseModel):
    """TF-IDF term and phrase ranking for one filtered corpus window."""

    lookback_days: int = Field(..., gt=0, description="Lookback window applied")
    channel_filter: str | None = Field(..., description="Channel filter applied; None means all channels")
    video_count: int = Field(..., ge=0, description="Summarized videos considered after filtering")
    term_themes: list[ThemeEntry] = Field(..., description="Ranked single-word TF-IDF terms over normalized summary text")
    phrase_themes: list[ThemeEntry] = Field(..., description="Ranked multi-word phrases meeting the document-frequency threshold")

    model_config = ConfigDict(frozen=True, extra="forbid")


class TimelineBucket(BaseModel):
    """Aggregated statistics for one time bucket (ISO week or month)."""

    bucket: str = Field(..., min_length=1, description="Bucket key (e.g. 2026-W23 or 2026-06)")
    video_count: int = Field(..., gt=0, description="Summarized videos uploaded in this bucket")
    channels: dict[str, int] = Field(..., description="Videos per channel, keys sorted")
    top_terms: list[str] = Field(..., description="Top single-word TF-IDF terms local to this bucket")
    top_phrases: list[str] = Field(..., description="Top multi-word TF-IDF phrases local to this bucket")

    model_config = ConfigDict(frozen=True, extra="forbid")


class TimelineReport(BaseModel):
    """Chronological corpus activity for one filtered window."""

    lookback_days: int = Field(..., gt=0, description="Lookback window applied")
    channel_filter: str | None = Field(..., description="Channel filter applied; None means all channels")
    bucket_type: str = Field(..., min_length=1, description="Bucket granularity: week or month")
    video_count: int = Field(..., ge=0, description="Summarized videos considered after filtering")
    buckets: list[TimelineBucket] = Field(..., description="Buckets in chronological order")

    model_config = ConfigDict(frozen=True, extra="forbid")


class AnalyticsSnapshot(BaseModel):
    """Persisted top-results snapshot used for run-over-run diffing."""

    generated_on: str = Field(..., min_length=1, description="ISO date of the run that wrote the snapshot")
    top_terms: list[str] = Field(..., description="Top TF-IDF terms of that run")
    top_phrases: list[str] = Field(..., description="Top TF-IDF phrases of that run")

    model_config = ConfigDict(frozen=True, extra="forbid")


class EmergingDiff(BaseModel):
    """What is new in this run compared to the previous snapshot."""

    new_terms: list[str] = Field(..., description="Terms absent from the previous snapshot, sorted")
    new_phrases: list[str] = Field(..., description="Phrases absent from the previous snapshot, sorted")
    previous_generated_on: str | None = Field(..., description="Date of the previous snapshot; None on first run")

    model_config = ConfigDict(frozen=True, extra="forbid")
