"""Unit tests for format-agnostic summary text normalization.

Summaries are opaque markdown documents (plan Amendment 7): normalization
strips generic markup and preserves words; nothing is interpreted as a
template section or field label. Heterogeneous fixtures (template-shaped and
freeform) prove independence from the summarize prompt layout.
"""

from pathlib import Path

import pytest

from src.analytics.errors import AnalyticsError, EmptySummaryError
from src.analytics.text_normalize import load_normalized_summary, normalize_markdown, word_count

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "analytics"


class TestNormalizeMarkdown:
    """Generic markup stripping without section interpretation."""

    def test_heading_markers_stripped_words_kept(self) -> None:
        """ATX markers disappear; heading words stay as plain text."""
        assert normalize_markdown("# Overview\nSome text") == "Overview Some text"

    def test_emphasis_markers_stripped(self) -> None:
        """Bold/italic/code markers disappear; the words stay."""
        assert normalize_markdown("**Key points:** and *emphasis* and `code`") == "Key points: and emphasis and code"

    def test_bullet_prefixes_stripped(self) -> None:
        """Dash, star, and numbered list prefixes disappear; items stay."""
        assert normalize_markdown("- first item\n* second item\n3. third item") == "first item second item third item"

    def test_whitespace_collapsed(self) -> None:
        """Newlines, tabs, and runs of spaces collapse to single spaces."""
        assert normalize_markdown("words\n\n\nwith   gaps\t\there") == "words with gaps here"

    def test_empty_string_normalizes_to_empty(self) -> None:
        """An empty document stays empty."""
        assert normalize_markdown("") == ""

    def test_markup_only_normalizes_to_empty(self) -> None:
        """A document of pure markup has no words left."""
        assert normalize_markdown("##\n\n- \n**\n") == ""

    def test_template_shaped_fixture_keeps_words_drops_markup(self) -> None:
        """The template-shaped fixture becomes plain words, not parsed fields."""
        text = (FIXTURES_DIR / "sample_summary_template_shaped.md").read_text(encoding="utf-8")
        normalized = normalize_markdown(text)
        assert "Overview" in normalized
        assert "Key points:" in normalized
        assert "#" not in normalized
        assert "*" not in normalized

    def test_freeform_fixture_normalizes_nonempty(self) -> None:
        """A freeform prose summary is equally valid input."""
        text = (FIXTURES_DIR / "sample_summary_freeform.md").read_text(encoding="utf-8")
        normalized = normalize_markdown(text)
        assert "maintainership" in normalized
        assert normalized == " ".join(normalized.split())


class TestWordCount:
    """Whitespace word counting."""

    def test_counts_whitespace_separated_words(self) -> None:
        """Words across newlines and spaces are counted."""
        assert word_count("a b\nc") == 3

    def test_empty_text_counts_zero(self) -> None:
        """Empty text has zero words."""
        assert word_count("") == 0


class TestLoadNormalizedSummary:
    """File loading with fail-fast error semantics."""

    def test_returns_normalized_text(self, tmp_path: Path) -> None:
        """A readable summary loads as normalized plain text."""
        path = tmp_path / "summary.md"
        path.write_text("# A\n\nSome **bold** words\n", encoding="utf-8")
        assert load_normalized_summary(path) == "A Some bold words"

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """An empty summary file fails fast with its path."""
        path = tmp_path / "summary.md"
        path.write_text("", encoding="utf-8")
        with pytest.raises(EmptySummaryError, match="summary.md"):
            load_normalized_summary(path)

    def test_markup_only_file_raises(self, tmp_path: Path) -> None:
        """A summary that is pure markup is empty after normalization."""
        path = tmp_path / "summary.md"
        path.write_text("## \n\n- \n", encoding="utf-8")
        with pytest.raises(EmptySummaryError, match="summary.md"):
            load_normalized_summary(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """An unreadable summary aborts with an I/O error, not silence."""
        with pytest.raises(AnalyticsError, match="cannot be read"):
            load_normalized_summary(tmp_path / "missing.md")

    def test_undecodable_file_raises(self, tmp_path: Path) -> None:
        """Bytes that are not UTF-8 abort with a decode error."""
        path = tmp_path / "summary.md"
        path.write_bytes(b"\xff\xfe\xfa")
        with pytest.raises(AnalyticsError, match="cannot be read"):
            load_normalized_summary(path)
