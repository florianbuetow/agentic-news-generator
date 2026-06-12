"""TF-IDF theme ranking over normalized, opaque summary text.

Summaries are never parsed (plan Amendment 7): each candidate record
contributes one TF-IDF document — its normalized summary text re-read from
``paths.summary_md``, optionally extended with the normalized cleaned
transcript. Single-word features rank as terms; multi-word features meeting
the document-frequency threshold rank as phrase themes. All knobs come from
AnalyticsConfig; the reference date is injected so runs and tests are
deterministic.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analytics.errors import AnalyticsError, MetadataError
from src.analytics.models import CorpusIndex, CorpusRecord, ThemeEntry, ThemeReport, ThemeVideo
from src.analytics.text_normalize import load_normalized_summary, normalize_markdown
from src.config import AnalyticsConfig
from src.util.channel_name import sanitize_channel_name

TFIDF_MAX_FEATURES: int = 10000
TFIDF_TOKEN_PATTERN: str = r"(?u)\b\w\w+\b"
SCORE_DECIMALS: int = 6


def rank_themes(index: CorpusIndex, analytics_config: AnalyticsConfig, reference_date: date) -> ThemeReport:
    """Rank TF-IDF terms and phrases for the configured window.

    Args:
        index: The joined corpus index.
        analytics_config: Analytics knobs (lookback, filters, caps, n-grams).
        reference_date: Date the lookback window is anchored to.

    Returns:
        ThemeReport with ranked term and phrase themes.
    """
    candidates = filter_candidate_records(index, analytics_config, reference_date)
    term_themes, phrase_themes = rank_features_for_records(candidates, analytics_config)
    return ThemeReport(
        lookback_days=analytics_config.lookback_days,
        channel_filter=analytics_config.channel_filter,
        video_count=len(candidates),
        term_themes=term_themes,
        phrase_themes=phrase_themes,
    )


def filter_candidate_records(index: CorpusIndex, analytics_config: AnalyticsConfig, reference_date: date) -> list[CorpusRecord]:
    """Select summarized records inside the lookback window and channel filter.

    Args:
        index: The joined corpus index.
        analytics_config: Analytics knobs (lookback, channel filter).
        reference_date: Date the lookback window is anchored to.

    Returns:
        Candidate records in index order.

    Raises:
        AnalyticsError: If channel_filter matches no channel in the corpus.
        MetadataError: If a candidate record lacks an upload_date.
    """
    channel_filter = _sanitized_channel_filter(index, analytics_config)
    cutoff = reference_date - timedelta(days=analytics_config.lookback_days)

    candidates: list[CorpusRecord] = []
    for record in index.records:
        if not record.has_summary:
            continue
        if channel_filter is not None and record.channel != channel_filter:
            continue
        if record.upload_date is None:
            raise MetadataError(f"Record '{record.video_id}' lacks upload_date required for lookback filtering")
        if date.fromisoformat(record.upload_date) >= cutoff:
            candidates.append(record)
    return candidates


def rank_features_for_records(records: list[CorpusRecord], analytics_config: AnalyticsConfig) -> tuple[list[ThemeEntry], list[ThemeEntry]]:
    """Rank TF-IDF features over one document per summarized record.

    Args:
        records: Summarized corpus records (one TF-IDF document each).
        analytics_config: Caps, n-gram range, document-frequency threshold.

    Returns:
        (term_themes, phrase_themes): single-word terms sorted by descending
        mean TF-IDF then alphabetically, capped at top_n_terms; multi-word
        phrases meeting min_theme_document_frequency sorted by descending
        document frequency, then score, then alphabetically, capped at
        top_n_themes.
    """
    documents = [_record_document(record, analytics_config) for record in records]
    if not documents:
        return ([], [])

    vectorizer = TfidfVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words="english",
        token_pattern=TFIDF_TOKEN_PATTERN,
        ngram_range=(analytics_config.tfidf_ngram_range_min, analytics_config.tfidf_ngram_range_max),
        max_df=1.0,
        min_df=1,
        max_features=TFIDF_MAX_FEATURES,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    try:
        vectorizer.fit(documents)
    except ValueError as e:
        if "empty vocabulary" not in str(e):
            raise
        print("No TF-IDF vocabulary survives stop-word removal; theme lists are empty", flush=True)
        return ([], [])

    matrix = vectorizer.transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    score_sums = [0.0] * len(feature_names)
    contributors: list[list[tuple[float, CorpusRecord]]] = [[] for _ in feature_names]
    for document_index, record in enumerate(records):
        row = matrix[document_index]
        for feature_index, value in zip(row.indices, row.data, strict=True):
            score_sums[feature_index] += value
            contributors[feature_index].append((value, record))

    entries: list[ThemeEntry] = []
    for feature_index, term in enumerate(feature_names):
        contributing = contributors[feature_index]
        if not contributing:
            continue
        entries.append(
            ThemeEntry(
                term=term,
                score=round(score_sums[feature_index] / len(documents), SCORE_DECIMALS),
                document_frequency=len(contributing),
                channels=sorted({record.channel for _, record in contributing}),
                videos=_theme_videos(contributing, analytics_config.top_n_videos_per_theme),
            )
        )

    term_themes = [entry for entry in entries if " " not in entry.term]
    term_themes.sort(key=lambda entry: (-entry.score, entry.term))
    phrase_themes = [
        entry for entry in entries if " " in entry.term and entry.document_frequency >= analytics_config.min_theme_document_frequency
    ]
    phrase_themes.sort(key=lambda entry: (-entry.document_frequency, -entry.score, entry.term))
    return (term_themes[: analytics_config.top_n_terms], phrase_themes[: analytics_config.top_n_themes])


def _sanitized_channel_filter(index: CorpusIndex, analytics_config: AnalyticsConfig) -> str | None:
    """Sanitize the configured channel filter and verify it exists in the corpus."""
    if analytics_config.channel_filter is None:
        return None
    sanitized = sanitize_channel_name(analytics_config.channel_filter)
    known_channels = {record.channel for record in index.records}
    if sanitized not in known_channels:
        raise AnalyticsError(
            f"channel_filter '{analytics_config.channel_filter}' (sanitized '{sanitized}') matches no channel in the corpus index"
        )
    return sanitized


def _record_document(record: CorpusRecord, analytics_config: AnalyticsConfig) -> str:
    """One TF-IDF document: normalized summary text, optionally + cleaned txt.

    Raises:
        AnalyticsError: If a referenced source file cannot be read, or a
            summarized record carries no summary path.
        EmptySummaryError: If the summary is empty after normalization.
    """
    if record.paths.summary_md is None:
        raise AnalyticsError(f"Record '{record.video_id}' has has_summary=True but no summary path")
    document = load_normalized_summary(Path(record.paths.summary_md))
    if not analytics_config.include_cleaned_txt_in_tfidf:
        return document
    txt_path = Path(record.paths.cleaned_txt)
    try:
        txt_text = txt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise AnalyticsError(f"Cleaned transcript cannot be read: {txt_path}: {e}") from e
    return f"{document} {normalize_markdown(txt_text)}".strip()


def _theme_videos(contributing: list[tuple[float, CorpusRecord]], cap: int) -> list[ThemeVideo]:
    """Reference the highest-weight contributing videos, capped for report size."""
    ordered = sorted(contributing, key=lambda pair: (-pair[0], pair[1].channel, pair[1].video_id, pair[1].title))
    return [
        ThemeVideo(
            video_id=record.video_id,
            channel=record.channel,
            title=record.title,
            upload_date=record.upload_date,
            summary_md=record.paths.summary_md,
        )
        for _, record in ordered[:cap]
    ]
