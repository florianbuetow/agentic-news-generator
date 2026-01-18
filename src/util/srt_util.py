"""SRT file parsing utilities with word position tracking for timestamp mapping."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import srt


@dataclass(frozen=True)
class SRTEntry:
    """Single SRT subtitle entry with word position tracking.

    Attributes:
        index: Original SRT entry index (1-based).
        start: Start timestamp as timedelta.
        end: End timestamp as timedelta.
        content: Text content of the subtitle.
        word_start: Starting word position in the full concatenated text.
        word_end: Ending word position (exclusive) in the full concatenated text.
    """

    index: int
    start: timedelta
    end: timedelta
    content: str
    word_start: int
    word_end: int


class SRTUtil:
    """Utility class for SRT file parsing and timestamp mapping."""

    @staticmethod
    def parse_srt_file(srt_path: Path) -> list[SRTEntry]:
        """Parse SRT file into list of entries with word position mapping.

        Each entry tracks its word positions in the full concatenated text,
        enabling mapping from segment boundaries back to timestamps.

        Args:
            srt_path: Path to the SRT file.

        Returns:
            List of SRTEntry objects with word position tracking.

        Raises:
            FileNotFoundError: If the SRT file does not exist.
            srt.SRTParseError: If the SRT file is malformed.
        """
        content = srt_path.read_text(encoding="utf-8")
        subtitles = list(srt.parse(content))

        entries: list[SRTEntry] = []
        word_pos = 0
        for sub in subtitles:
            words = sub.content.split()
            word_count = len(words)
            # srt library's start/end are timedelta-compatible but typed as _Timecode
            start_td = timedelta(seconds=sub.start.total_seconds())
            end_td = timedelta(seconds=sub.end.total_seconds())
            entries.append(
                SRTEntry(
                    index=sub.index,
                    start=start_td,
                    end=end_td,
                    content=sub.content,
                    word_start=word_pos,
                    word_end=word_pos + word_count,
                )
            )
            word_pos += word_count
        return entries

    @staticmethod
    def entries_to_text(entries: list[SRTEntry]) -> str:
        """Concatenate SRT entries into plain text.

        Args:
            entries: List of SRTEntry objects.

        Returns:
            Single string with all entry contents joined by spaces.
        """
        return " ".join(e.content for e in entries)

    @staticmethod
    def word_position_to_timestamp(word_pos: int, entries: list[SRTEntry]) -> timedelta:
        """Map a word position back to the corresponding SRT timestamp.

        Finds which SRT entry contains the given word position and returns
        that entry's start timestamp.

        Args:
            word_pos: Word position in the concatenated text.
            entries: List of SRTEntry objects with word position tracking.

        Returns:
            The start timestamp of the SRT entry containing the word position.
            If word_pos is beyond all entries, returns the last entry's end timestamp.
            If entries is empty, returns timedelta(0).
        """
        for entry in entries:
            if entry.word_start <= word_pos < entry.word_end:
                return entry.start
        # If beyond all entries, return last entry's end
        return entries[-1].end if entries else timedelta(0)

    @staticmethod
    def get_total_words(entries: list[SRTEntry]) -> int:
        """Get the total word count across all entries.

        Args:
            entries: List of SRTEntry objects.

        Returns:
            Total number of words. Returns 0 if entries is empty.
        """
        if not entries:
            return 0
        return entries[-1].word_end
