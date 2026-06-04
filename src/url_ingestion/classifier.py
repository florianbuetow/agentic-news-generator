"""Rule-based URL classification for URL ingestion."""

from typing import Literal
from urllib.parse import urlsplit

UrlContentType = Literal["pdf", "html", "markdown", "text", "unknown"]

_NON_DOCUMENT_SUFFIXES = {
    ".avi",
    ".css",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".png",
    ".svg",
    ".tar",
    ".webm",
    ".webp",
    ".zip",
}

_UNSUPPORTED_HOSTS = {
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtu.be",
    "youtube.com",
}


class UrlClassifier:
    """Classify normalized URLs by path suffix without network access."""

    def classify(self, normalized_url: str) -> UrlContentType:
        """Classify a normalized URL by inspecting its path suffix."""
        split_url = urlsplit(normalized_url)
        if self._is_unsupported_host(split_url.hostname):
            return "unknown"
        path = split_url.path
        path_lower = path.lower()
        if path_lower.endswith(".pdf"):
            return "pdf"
        if path_lower.endswith((".html", ".htm")):
            return "html"
        if path_lower.endswith(".md"):
            return "markdown"
        if path_lower.endswith(".txt"):
            return "text"
        suffix = self._suffix(path_lower)
        if suffix in _NON_DOCUMENT_SUFFIXES:
            return "unknown"
        return "html"

    def classification_order(self) -> tuple[UrlContentType, ...]:
        """Return deterministic output order for classification statistics."""
        return ("pdf", "html", "markdown", "text", "unknown")

    def _suffix(self, path: str) -> str:
        """Return the final path suffix when the last segment has an extension."""
        last_segment = path.rsplit("/", maxsplit=1)[-1]
        if "." not in last_segment:
            return ""
        return f".{last_segment.rsplit('.', maxsplit=1)[-1]}"

    def _is_unsupported_host(self, hostname: str | None) -> bool:
        """Return whether a host needs a non-generic future pipeline."""
        if hostname is None:
            return False
        normalized_hostname = hostname.lower().removeprefix("www.")
        return normalized_hostname in _UNSUPPORTED_HOSTS
