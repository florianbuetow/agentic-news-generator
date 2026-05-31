"""URL-derived processing identity helpers."""

import re


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
    return collapsed_stem[:180]
