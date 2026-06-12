"""Reader for yt-dlp ``.info.json`` video metadata files.

yt-dlp writes ``upload_date`` in compact ``YYYYMMDD`` form; this reader
normalizes it to ISO ``YYYY-MM-DD``. All failures raise MetadataError with
the offending path — analytics fails fast.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import cast

from src.analytics.errors import MetadataError
from src.analytics.models import VideoMetadata

COMPACT_DATE_RE: re.Pattern[str] = re.compile(r"\d{8}")
COMPACT_DATE_FORMAT: str = "%Y%m%d"


def load_video_metadata(path: Path) -> VideoMetadata:
    """Load and validate one ``.info.json`` metadata file.

    Args:
        path: Path to the metadata file.

    Returns:
        VideoMetadata with a normalized ISO upload_date.

    Raises:
        MetadataError: If the file is missing, unreadable, not a JSON object,
            lacks a usable title, or carries a malformed upload_date.
    """
    if not path.is_file():
        raise MetadataError(f"Metadata file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise MetadataError(f"Metadata file unreadable or invalid JSON: {path}: {e}") from e

    if not isinstance(payload, dict):
        raise MetadataError(f"Metadata file is not a JSON object: {path}")
    fields = cast(dict[str, object], payload)

    title = fields.get("title")
    if not isinstance(title, str) or not title:
        raise MetadataError(f"Metadata file lacks a non-empty 'title': {path}")

    description = fields.get("description")
    if description is not None and not isinstance(description, str):
        raise MetadataError(f"Metadata 'description' is not a string: {path}")

    video_id = fields.get("id")
    if video_id is not None and (not isinstance(video_id, str) or not video_id):
        raise MetadataError(f"Metadata 'id' is not a non-empty string: {path}")

    return VideoMetadata(
        title=title,
        upload_date=_normalize_upload_date(fields.get("upload_date"), path),
        description=description,
        video_id=video_id,
    )


def _normalize_upload_date(raw_value: object, path: Path) -> str | None:
    """Normalize an upload_date value to ISO form, or None when absent.

    The shape is matched strictly before parsing because strptime alone is
    lenient (it accepts single-digit months/days, e.g. seven-digit dates).
    """
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise MetadataError(f"Metadata 'upload_date' is not a string: {path}")
    if COMPACT_DATE_RE.fullmatch(raw_value) is None:
        raise MetadataError(f"Metadata 'upload_date' is not compact YYYYMMDD: '{raw_value}' in {path}")
    try:
        return datetime.strptime(raw_value, COMPACT_DATE_FORMAT).date().isoformat()
    except ValueError as e:
        raise MetadataError(f"Metadata 'upload_date' is not a valid calendar date: '{raw_value}' in {path}") from e
