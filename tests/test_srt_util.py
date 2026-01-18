"""Tests for SRT utility module."""

from datetime import timedelta
from pathlib import Path

import pytest

from src.util.srt_util import SRTEntry, SRTUtil


class TestSRTEntry:
    """Tests for SRTEntry dataclass."""

    def test_srt_entry_creation(self) -> None:
        """Test creating an SRTEntry."""
        entry = SRTEntry(
            index=1,
            start=timedelta(seconds=0),
            end=timedelta(seconds=5),
            content="Hello world",
            word_start=0,
            word_end=2,
        )
        assert entry.index == 1
        assert entry.start == timedelta(seconds=0)
        assert entry.end == timedelta(seconds=5)
        assert entry.content == "Hello world"
        assert entry.word_start == 0
        assert entry.word_end == 2

    def test_srt_entry_is_frozen(self) -> None:
        """Test that SRTEntry is immutable."""
        entry = SRTEntry(
            index=1,
            start=timedelta(seconds=0),
            end=timedelta(seconds=5),
            content="Hello",
            word_start=0,
            word_end=1,
        )
        with pytest.raises(AttributeError):
            entry.content = "Changed"  # type: ignore[misc]


class TestSRTUtilParsing:
    """Tests for SRTUtil parsing functionality."""

    def test_parse_simple_srt_file(self, tmp_path: Path) -> None:
        """Test parsing a simple SRT file."""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello world

2
00:00:05,000 --> 00:00:10,000
This is a test
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        entries = SRTUtil.parse_srt_file(srt_file)

        assert len(entries) == 2
        assert entries[0].index == 1
        assert entries[0].content == "Hello world"
        assert entries[0].word_start == 0
        assert entries[0].word_end == 2
        assert entries[1].index == 2
        assert entries[1].content == "This is a test"
        assert entries[1].word_start == 2
        assert entries[1].word_end == 6

    def test_parse_srt_tracks_word_positions(self, tmp_path: Path) -> None:
        """Test that word positions are tracked correctly across entries."""
        srt_content = """1
00:00:00,000 --> 00:00:02,000
One two three

2
00:00:02,000 --> 00:00:04,000
Four five

3
00:00:04,000 --> 00:00:06,000
Six seven eight nine
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        entries = SRTUtil.parse_srt_file(srt_file)

        # Entry 1: words 0-2 (one, two, three)
        assert entries[0].word_start == 0
        assert entries[0].word_end == 3
        # Entry 2: words 3-4 (four, five)
        assert entries[1].word_start == 3
        assert entries[1].word_end == 5
        # Entry 3: words 5-8 (six, seven, eight, nine)
        assert entries[2].word_start == 5
        assert entries[2].word_end == 9

    def test_parse_empty_srt_file(self, tmp_path: Path) -> None:
        """Test parsing an empty SRT file."""
        srt_file = tmp_path / "empty.srt"
        srt_file.write_text("", encoding="utf-8")

        entries = SRTUtil.parse_srt_file(srt_file)

        assert len(entries) == 0

    def test_parse_nonexistent_file(self, tmp_path: Path) -> None:
        """Test parsing a file that doesn't exist raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.srt"

        with pytest.raises(FileNotFoundError):
            SRTUtil.parse_srt_file(nonexistent)


class TestSRTUtilEntriesToText:
    """Tests for SRTUtil.entries_to_text."""

    def test_entries_to_text_simple(self) -> None:
        """Test converting entries to plain text."""
        entries = [
            SRTEntry(
                index=1,
                start=timedelta(seconds=0),
                end=timedelta(seconds=5),
                content="Hello world",
                word_start=0,
                word_end=2,
            ),
            SRTEntry(
                index=2,
                start=timedelta(seconds=5),
                end=timedelta(seconds=10),
                content="This is a test",
                word_start=2,
                word_end=6,
            ),
        ]

        text = SRTUtil.entries_to_text(entries)

        assert text == "Hello world This is a test"

    def test_entries_to_text_empty_list(self) -> None:
        """Test converting empty entries list."""
        text = SRTUtil.entries_to_text([])
        assert text == ""


class TestSRTUtilWordPositionToTimestamp:
    """Tests for SRTUtil.word_position_to_timestamp."""

    def test_word_position_in_first_entry(self) -> None:
        """Test finding timestamp for word in first entry."""
        entries = [
            SRTEntry(
                index=1,
                start=timedelta(seconds=0),
                end=timedelta(seconds=5),
                content="Hello world",
                word_start=0,
                word_end=2,
            ),
            SRTEntry(
                index=2,
                start=timedelta(seconds=5),
                end=timedelta(seconds=10),
                content="Test content",
                word_start=2,
                word_end=4,
            ),
        ]

        timestamp = SRTUtil.word_position_to_timestamp(0, entries)
        assert timestamp == timedelta(seconds=0)

        timestamp = SRTUtil.word_position_to_timestamp(1, entries)
        assert timestamp == timedelta(seconds=0)

    def test_word_position_in_second_entry(self) -> None:
        """Test finding timestamp for word in second entry."""
        entries = [
            SRTEntry(
                index=1,
                start=timedelta(seconds=0),
                end=timedelta(seconds=5),
                content="Hello world",
                word_start=0,
                word_end=2,
            ),
            SRTEntry(
                index=2,
                start=timedelta(seconds=5),
                end=timedelta(seconds=10),
                content="Test content",
                word_start=2,
                word_end=4,
            ),
        ]

        timestamp = SRTUtil.word_position_to_timestamp(2, entries)
        assert timestamp == timedelta(seconds=5)

        timestamp = SRTUtil.word_position_to_timestamp(3, entries)
        assert timestamp == timedelta(seconds=5)

    def test_word_position_beyond_all_entries(self) -> None:
        """Test that position beyond all entries returns last entry's end."""
        entries = [
            SRTEntry(
                index=1,
                start=timedelta(seconds=0),
                end=timedelta(seconds=5),
                content="Hello",
                word_start=0,
                word_end=1,
            ),
        ]

        timestamp = SRTUtil.word_position_to_timestamp(10, entries)
        assert timestamp == timedelta(seconds=5)  # Last entry's end

    def test_word_position_with_empty_entries(self) -> None:
        """Test that position with empty entries returns zero."""
        timestamp = SRTUtil.word_position_to_timestamp(0, [])
        assert timestamp == timedelta(0)


class TestSRTUtilGetTotalWords:
    """Tests for SRTUtil.get_total_words."""

    def test_get_total_words(self) -> None:
        """Test getting total word count."""
        entries = [
            SRTEntry(
                index=1,
                start=timedelta(seconds=0),
                end=timedelta(seconds=5),
                content="Hello world",
                word_start=0,
                word_end=2,
            ),
            SRTEntry(
                index=2,
                start=timedelta(seconds=5),
                end=timedelta(seconds=10),
                content="Test content here",
                word_start=2,
                word_end=5,
            ),
        ]

        total = SRTUtil.get_total_words(entries)
        assert total == 5

    def test_get_total_words_empty(self) -> None:
        """Test getting total words for empty list."""
        total = SRTUtil.get_total_words([])
        assert total == 0
