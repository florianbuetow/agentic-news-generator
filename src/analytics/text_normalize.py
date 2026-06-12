"""Format-agnostic markdown normalization for opaque summary documents.

Per plan Amendment 7, summaries are never parsed against the summarize
template: this module strips generic markdown syntax (heading markers,
emphasis characters, list prefixes, blockquote markers), preserves every
word, and collapses whitespace. No label, heading, or section is interpreted
semantically.
"""

import re
from pathlib import Path

from src.analytics.errors import AnalyticsError, EmptySummaryError

HEADING_MARKER_RE: re.Pattern[str] = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
LIST_PREFIX_RE: re.Pattern[str] = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
BLOCKQUOTE_MARKER_RE: re.Pattern[str] = re.compile(r"^\s*>\s?", re.MULTILINE)
EMPHASIS_CHARS_RE: re.Pattern[str] = re.compile(r"[*_`]")


def normalize_markdown(text: str) -> str:
    """Strip generic markdown syntax, keep every word, collapse whitespace.

    Args:
        text: Arbitrary markdown (or plain) text.

    Returns:
        Plain text suitable as a TF-IDF document; "" for word-less input.
    """
    without_quotes = BLOCKQUOTE_MARKER_RE.sub("", text)
    without_headings = HEADING_MARKER_RE.sub("", without_quotes)
    without_lists = LIST_PREFIX_RE.sub("", without_headings)
    without_emphasis = EMPHASIS_CHARS_RE.sub("", without_lists)
    return " ".join(without_emphasis.split())


def word_count(text: str) -> int:
    """Count whitespace-separated words.

    Args:
        text: Any text.

    Returns:
        Number of whitespace-separated tokens.
    """
    return len(text.split())


def load_normalized_summary(path: Path) -> str:
    """Read one summary file and return its normalized text, failing fast.

    Args:
        path: Summary markdown file.

    Returns:
        Normalized non-empty plain text.

    Raises:
        AnalyticsError: If the file cannot be read or decoded.
        EmptySummaryError: If the file is empty after normalization.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise AnalyticsError(f"Summary file cannot be read: {path}: {e}") from e
    normalized = normalize_markdown(text)
    if not normalized:
        raise EmptySummaryError(f"Summary file is empty after normalization: {path}")
    return normalized
