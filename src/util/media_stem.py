"""Resolve a YouTube video's canonical file stem across pipeline artifact directories.

The pipeline names every artifact for a video ``<title> [VIDEO_ID].<ext>`` and
keys a video's metadata by that stem. When metadata has to be re-fetched, the
stem must be recovered from whatever artifact still exists on disk. For an
*archived* video that is no longer the WAV under ``downloads/audio`` — the
``archive-videos`` step removes it — but the archived video file under
``archive/videos`` or a transcript under ``downloads/transcripts``. Searching a
prioritized list of directories lets metadata be restored no matter how far
through the pipeline, or into the archive, a video has moved.
"""

from __future__ import annotations

import re
from pathlib import Path

# YouTube video IDs are exactly 11 characters of [A-Za-z0-9_-].
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def resolve_media_stem(search_dirs: list[Path], video_id: str) -> str | None:
    """Find a video's canonical ``<title> [VIDEO_ID]`` stem from its on-disk artifacts.

    Directories are searched in the given order; the first one holding a file
    whose name brackets ``video_id`` wins. The returned stem is the filename up
    to and including ``[VIDEO_ID]``, so any trailing format-code suffix
    (``.f251-11``) or extension is dropped — metadata files never carry those, so
    this yields the exact stem the rest of the pipeline expects.

    Args:
        search_dirs: Per-channel artifact directories, most authoritative first
            (e.g. audio, active video, archived video, transcripts). Directories
            that do not exist are skipped.
        video_id: The 11-character YouTube video ID.

    Returns:
        The canonical stem, or None when no artifact for the video exists in any
        of the directories.

    Raises:
        ValueError: If ``video_id`` is not a valid YouTube video ID.
    """
    if not _VIDEO_ID_RE.match(video_id):
        raise ValueError(f"Invalid YouTube video ID: {video_id!r}. Expected 11 characters of [A-Za-z0-9_-].")

    marker = f"[{video_id}]"
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"*[[]{video_id}[]]*")):
            if not path.is_file() or path.name.startswith("._"):
                continue
            index = path.name.find(marker)
            if index != -1:
                return path.name[: index + len(marker)]
    return None
