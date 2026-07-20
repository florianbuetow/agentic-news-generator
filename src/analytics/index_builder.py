"""Builder for the analytics corpus index.

Walks the cleaned transcripts tree, joins each transcript to its summary,
yt-dlp metadata, and channel configuration, and serializes the result as
``corpus_index.json``. Fail-fast: any join or parse failure aborts the run.
Three conditions are tolerated rather than fatal: a cleaned transcript without
a summary file is indexed with ``has_summary: false``; a summary that is empty
after normalization counts as missing too (Amendment 9 — the upstream
summarizer sometimes writes 0-byte files; a warning is printed); and the same
video downloaded more than once (multiple files sharing one video_id) is
indexed once per cleaned transcript, each paired to the sibling files whose
stem matches it.

The metadata video subdirectory is fixed to ``video`` per the documented
on-disk layout produced by the download pipeline.
"""

from pathlib import Path

from src.analytics.errors import AnalyticsError, EmptySummaryError, JoinError, MetadataError
from src.analytics.ids import extract_youtube_id
from src.analytics.metadata_reader import load_video_metadata
from src.analytics.models import ChannelMeta, CorpusIndex, CorpusRecord, RecordPaths, SummaryStats
from src.analytics.text_normalize import load_normalized_summary, word_count
from src.config import ChannelConfig, Config
from src.util.channel_name import sanitize_channel_name
from src.util.srt_util import SRTUtil

CORPUS_INDEX_FILENAME: str = "corpus_index.json"
METADATA_VIDEO_SUBDIR: str = "video"


def build_index(config: Config) -> CorpusIndex:
    """Build the joined corpus index from the cleaned-and-summarized corpus.

    Args:
        config: Loaded project configuration.

    Returns:
        CorpusIndex with records sorted by (channel, upload_date, video_id).

    Raises:
        AnalyticsError: If the cleaned corpus is empty or a file is unreadable.
        JoinError: If files cannot be joined to IDs or channel config.
        MetadataError: If video metadata is missing or invalid.
    """
    cleaned_dir = config.get_data_downloads_transcripts_cleaned_dir()
    summaries_dir = config.get_data_downloads_transcripts_summaries_dir()
    metadata_dir = config.get_data_downloads_metadata_dir()
    channel_by_dir = _channel_map(config)

    records: list[CorpusRecord] = []
    for channel_dir in _iter_channel_dirs(cleaned_dir):
        if channel_dir.name not in channel_by_dir:
            raise JoinError(f"Cleaned channel directory '{channel_dir.name}' matches no sanitized channel name from config.yaml")
        channel_config = channel_by_dir[channel_dir.name]
        txt_files = _iter_cleaned_txt_files(channel_dir)
        print(f"Processing channel: {channel_dir.name} ({len(txt_files)} cleaned transcripts)", flush=True)
        records.extend(
            _build_record(
                txt_file=txt_file,
                channel_name=channel_dir.name,
                channel_config=channel_config,
                summaries_channel_dir=summaries_dir / channel_dir.name,
                metadata_video_dir=metadata_dir / channel_dir.name / METADATA_VIDEO_SUBDIR,
            )
            for txt_file in txt_files
        )

    if not records:
        raise AnalyticsError(f"No cleaned transcripts found under: {cleaned_dir}")

    records.sort(key=_record_sort_key)
    with_summary = sum(1 for record in records if record.has_summary)
    print(f"Indexed {len(records)} videos ({with_summary} with summaries)", flush=True)
    return CorpusIndex(records=records)


def write_corpus_index(index: CorpusIndex, output_dir: Path) -> Path:
    """Serialize the corpus index as JSON into the analytics output directory.

    Args:
        index: The corpus index to write.
        output_dir: Analytics output directory (created when missing).

    Returns:
        Path of the written corpus_index.json.

    Raises:
        AnalyticsError: If the output directory cannot be created.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise AnalyticsError(f"Cannot create analytics output directory: {output_dir}: {e}") from e
    output_path = output_dir / CORPUS_INDEX_FILENAME
    output_path.write_text(index.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} ({len(index.records)} records)", flush=True)
    return output_path


def _channel_map(config: Config) -> dict[str, ChannelConfig]:
    """Map sanitized channel directory names to their configurations.

    Raises:
        JoinError: If two configured channels sanitize to the same directory
            name — records could silently receive the wrong channel_meta.
    """
    channel_by_dir: dict[str, ChannelConfig] = {}
    for channel in config.get_channels():
        sanitized = sanitize_channel_name(channel.name)
        if sanitized in channel_by_dir:
            raise JoinError(f"Channels '{channel_by_dir[sanitized].name}' and '{channel.name}' both sanitize to directory '{sanitized}'")
        channel_by_dir[sanitized] = channel
    return channel_by_dir


def _iter_channel_dirs(cleaned_dir: Path) -> list[Path]:
    """List channel directories in the cleaned corpus in stable order.

    Hidden dot-directories (a stray ``.claude`` project dir, ``.DS_Store``,
    macOS ``._`` AppleDouble entries) are never channels: sanitize_channel_name
    strips ``.``, so no real channel directory can start with one. They are
    skipped rather than treated as an unconfigured channel.
    """
    if not cleaned_dir.is_dir():
        return []
    return sorted(
        (entry for entry in cleaned_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")),
        key=lambda entry: entry.name,
    )


def _iter_cleaned_txt_files(channel_dir: Path) -> list[Path]:
    """List cleaned .txt transcripts of one channel in stable order.

    Channel directories are flat per the documented layout; nested files are
    not part of the corpus.
    """
    return sorted(
        (entry for entry in channel_dir.glob("*.txt") if entry.is_file() and not entry.name.startswith("._")),
        key=str,
    )


def _build_record(
    txt_file: Path,
    channel_name: str,
    channel_config: ChannelConfig,
    summaries_channel_dir: Path,
    metadata_video_dir: Path,
) -> CorpusRecord:
    """Join one cleaned transcript with its metadata and optional summary.

    The corpus contains the same video downloaded more than once (e.g. re-fetched
    after a channel rename), so a video_id is not unique: every cleaned transcript
    is indexed as its own record. When several sibling files share a video_id,
    each record is paired with the file whose stem matches this transcript.
    """
    srt_file = txt_file.with_suffix(".srt")
    has_srt = srt_file.is_file()
    duration_seconds = _srt_duration_seconds(srt_file) if has_srt else None

    video_id = extract_youtube_id(txt_file.name)
    if video_id is None:
        # Amendment 10: early downloads carry no bracketed ID; join by exact
        # stem and take the id from metadata. Token search is off-limits here —
        # it would steal the re-download's files.
        metadata_path = metadata_video_dir / f"{txt_file.stem}.info.json"
        if not metadata_path.is_file():
            raise JoinError(f"Cleaned transcript has no [video_id] token and no stem-matched info.json: {txt_file}")
        metadata = load_video_metadata(metadata_path)
        if metadata.video_id is None:
            raise MetadataError(f"Metadata lacks the 'id' required to identify an ID-less transcript: {metadata_path}")
        video_id = metadata.video_id
        summary_path = _stem_summary_file(summaries_channel_dir, txt_file.stem)
    else:
        metadata_path = _find_metadata_file(metadata_video_dir, channel_name, video_id, txt_file.stem)
        metadata = load_video_metadata(metadata_path)
        summary_path = _find_summary_file(summaries_channel_dir, video_id, txt_file.stem)
    summary_stats = _summary_stats(summary_path) if summary_path is not None else None
    if summary_stats is None:
        summary_path = None

    return CorpusRecord(
        video_id=video_id,
        channel=channel_name,
        title=metadata.title,
        upload_date=metadata.upload_date,
        duration_seconds=duration_seconds,
        word_count=_cleaned_word_count(txt_file),
        has_summary=summary_path is not None,
        paths=RecordPaths(
            cleaned_txt=str(txt_file),
            cleaned_srt=str(srt_file) if has_srt else None,
            summary_md=str(summary_path) if summary_path is not None else None,
            metadata_json=str(metadata_path),
        ),
        channel_meta=ChannelMeta(
            language=channel_config.language,
            category=channel_config.category,
            description=channel_config.description,
        ),
        summary_stats=summary_stats,
    )


def _srt_duration_seconds(srt_file: Path) -> int | None:
    """Duration in whole seconds from the last SRT entry end timestamp.

    A malformed SRT aborts the run with the file path attached (the srt
    library's own error lacks it); an entry-less SRT yields an unknown
    duration.
    """
    try:
        entries = SRTUtil.parse_srt_file(srt_file)
    except Exception as e:
        raise AnalyticsError(f"Cleaned SRT file cannot be parsed: {srt_file}: {e}") from e
    if not entries:
        return None
    return int(entries[-1].end.total_seconds())


def _find_metadata_file(metadata_video_dir: Path, channel_name: str, video_id: str, txt_stem: str) -> Path:
    """Locate one transcript's info.json by bracketed video ID, then by stem.

    A unique ID match is used directly (robust to a title edited between the
    metadata and transcript downloads). When several info.json share the ID —
    the same video downloaded more than once — the one whose stem matches this
    transcript is chosen so each duplicate keeps its own metadata.
    """
    candidates = _files_containing_id(metadata_video_dir, "*.info.json", video_id)
    if not candidates:
        raise MetadataError(f"No info.json found for video_id '{video_id}' in channel '{channel_name}' under: {metadata_video_dir}")
    if len(candidates) == 1:
        return candidates[0]
    stem_matches = [candidate for candidate in candidates if candidate.name.removesuffix(".info.json") == txt_stem]
    if len(stem_matches) == 1:
        return stem_matches[0]
    raise JoinError(f"Transcript '{txt_stem}' cannot be paired to one info.json for video_id '{video_id}': {candidates}")


def _stem_summary_file(summaries_channel_dir: Path, txt_stem: str) -> Path | None:
    """Summary for an ID-less transcript: exact stem match only (Amendment 10)."""
    candidate = summaries_channel_dir / f"{txt_stem}.md"
    if candidate.is_file():
        return candidate
    return None


def _find_summary_file(summaries_channel_dir: Path, video_id: str, txt_stem: str) -> Path | None:
    """Locate this transcript's summary markdown; absence is a coverage gap.

    A unique ID match is used directly. When several summaries share the ID
    (the video was downloaded more than once), the one whose stem matches this
    transcript is used; if none matches, this transcript simply has no summary.
    """
    candidates = _files_containing_id(summaries_channel_dir, "*.md", video_id)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    stem_matches = [candidate for candidate in candidates if candidate.stem == txt_stem]
    if len(stem_matches) == 1:
        return stem_matches[0]
    return None


def _files_containing_id(directory: Path, pattern: str, video_id: str) -> list[Path]:
    """List files in a directory whose name contains the bracketed ID."""
    if not directory.is_dir():
        return []
    token = f"[{video_id}]"
    return sorted(
        (entry for entry in directory.glob(pattern) if entry.is_file() and not entry.name.startswith("._") and token in entry.name),
        key=str,
    )


def _cleaned_word_count(txt_file: Path) -> int:
    """Whitespace word count of the cleaned transcript, failing fast with path.

    Raises:
        AnalyticsError: If the cleaned transcript cannot be read or decoded.
    """
    try:
        return word_count(txt_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        raise AnalyticsError(f"Cleaned transcript cannot be read: {txt_file}: {e}") from e


def _summary_stats(summary_path: Path) -> SummaryStats | None:
    """Size statistics of one summary's normalized text (never parsed).

    A summary that is empty after normalization is a coverage gap, not a fatal
    error (Amendment 9): the upstream summarizer sometimes writes 0-byte files.

    Returns:
        Stats for a non-empty summary; None (with a printed warning) when the
        summary is empty after normalization.

    Raises:
        AnalyticsError: If the summary file cannot be read.
    """
    try:
        normalized = load_normalized_summary(summary_path)
    except EmptySummaryError:
        print(f"Warning: summary is empty after normalization; treated as missing: {summary_path}", flush=True)
        return None
    return SummaryStats(word_count=word_count(normalized), char_count=len(normalized))


def _record_sort_key(record: CorpusRecord) -> tuple[str, str, str]:
    """Stable sort key: channel, upload_date (absent first), video_id."""
    upload_date = record.upload_date if record.upload_date is not None else ""
    return (record.channel, upload_date, record.video_id)
