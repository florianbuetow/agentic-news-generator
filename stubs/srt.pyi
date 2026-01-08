"""Type stubs for srt library."""

from collections.abc import Iterator
from datetime import timedelta
from typing import Protocol

class _Timecode(Protocol):
    """Protocol for timecode objects with to_timecode method."""

    def to_timecode(self) -> str: ...

class Subtitle:
    """SRT subtitle entry."""

    index: int
    start: _Timecode
    end: _Timecode
    content: str

    def __init__(
        self,
        index: int,
        start: timedelta,
        end: timedelta,
        content: str,
    ) -> None: ...

def parse(srt_content: str) -> Iterator[Subtitle]: ...
def compose(subtitles: list[Subtitle], reindex: bool = True, start_index: int = 1) -> str: ...
