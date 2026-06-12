"""Timeline builder: corpus activity bucketed by ISO week or calendar month.

Uses the same candidate filtering as the theme ranker (summarized records in
the lookback window, optional channel filter). Per bucket it reports the
video count, a per-channel breakdown, and the top TF-IDF terms and phrases
computed locally over that bucket's normalized summaries (the phrase
document-frequency threshold from config applies within each bucket).
"""

from collections import Counter
from datetime import date

from src.analytics.errors import MetadataError
from src.analytics.models import CorpusIndex, CorpusRecord, TimelineBucket, TimelineReport
from src.analytics.theme_ranker import filter_candidate_records, rank_features_for_records
from src.config import AnalyticsConfig

BUCKET_TOP_TERMS: int = 5
BUCKET_TOP_PHRASES: int = 5


def build_timeline(index: CorpusIndex, analytics_config: AnalyticsConfig, reference_date: date) -> TimelineReport:
    """Bucket the filtered corpus chronologically.

    Args:
        index: The joined corpus index.
        analytics_config: Analytics knobs (lookback, filters, bucket type).
        reference_date: Date the lookback window is anchored to.

    Returns:
        TimelineReport with buckets in chronological order.
    """
    candidates = filter_candidate_records(index, analytics_config, reference_date)

    grouped: dict[str, list[CorpusRecord]] = {}
    for record in candidates:
        grouped.setdefault(_bucket_key(record, analytics_config.timeline_bucket), []).append(record)

    buckets = [_build_bucket(key, grouped[key], analytics_config) for key in sorted(grouped)]
    return TimelineReport(
        lookback_days=analytics_config.lookback_days,
        channel_filter=analytics_config.channel_filter,
        bucket_type=analytics_config.timeline_bucket,
        video_count=len(candidates),
        buckets=buckets,
    )


def _bucket_key(record: CorpusRecord, bucket_type: str) -> str:
    """Bucket key for a record's upload date."""
    if record.upload_date is None:
        raise MetadataError(f"Record '{record.video_id}' lacks upload_date required for timeline bucketing")
    upload_date = date.fromisoformat(record.upload_date)
    if bucket_type == "week":
        iso = upload_date.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return f"{upload_date.year}-{upload_date.month:02d}"


def _build_bucket(key: str, records: list[CorpusRecord], analytics_config: AnalyticsConfig) -> TimelineBucket:
    """Aggregate one bucket's statistics."""
    channel_counts = Counter(record.channel for record in records)
    term_themes, phrase_themes = rank_features_for_records(records, analytics_config)
    return TimelineBucket(
        bucket=key,
        video_count=len(records),
        channels={channel: channel_counts[channel] for channel in sorted(channel_counts)},
        top_terms=[entry.term for entry in term_themes[:BUCKET_TOP_TERMS]],
        top_phrases=[entry.term for entry in phrase_themes[:BUCKET_TOP_PHRASES]],
    )
