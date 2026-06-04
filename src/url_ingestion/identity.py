"""URL-derived processing identity helpers."""

import hashlib
import re

MAX_FILENAME_STEM_BYTES: int = 240
HASH_SUFFIX_LENGTH: int = 16


def sanitize_normalized_url_to_stem(normalized_url: str) -> str:
    """Convert a normalized URL to a readable filesystem-safe filename stem."""
    raw_stem = normalized_url.replace("://", "__").replace(".", "_")
    safe_characters: list[str] = []
    for character in raw_stem:
        if character.isascii() and (character.isalnum() or character in {"_", "-"}):
            safe_characters.append(character)
        else:
            safe_characters.append("_")

    collapsed_stem = re.sub("_+", "_", "".join(safe_characters)).strip("_")
    if not collapsed_stem:
        raise ValueError(f"Normalized URL produced an empty sanitized stem: {normalized_url}")
    if len(collapsed_stem.encode("utf-8")) <= MAX_FILENAME_STEM_BYTES:
        return collapsed_stem

    digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:HASH_SUFFIX_LENGTH]
    suffix = f"_{digest}"
    prefix_byte_limit = MAX_FILENAME_STEM_BYTES - len(suffix)
    prefix = _truncate_to_utf8_byte_length(collapsed_stem, prefix_byte_limit).rstrip("_")
    if not prefix:
        return digest
    return f"{prefix}{suffix}"


def _truncate_to_utf8_byte_length(text: str, byte_limit: int) -> str:
    """Return text truncated to a UTF-8 byte limit without splitting a character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text
    return encoded[:byte_limit].decode("utf-8", errors="ignore")
