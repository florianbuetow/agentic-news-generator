"""Error hierarchy for the analytics package.

Analytics fails fast: any of these errors aborts the whole run with full
context, so partial indexes never masquerade as complete research reports.
"""


class AnalyticsError(Exception):
    """Base error for all analytics failures."""


class EmptySummaryError(AnalyticsError):
    """A present summary file is empty after format-agnostic normalization."""


class MetadataError(AnalyticsError):
    """Video metadata is missing, unreadable, or lacks required fields."""


class JoinError(AnalyticsError):
    """Corpus files cannot be joined to metadata or channel configuration."""
