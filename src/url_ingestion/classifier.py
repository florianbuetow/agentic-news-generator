"""Rule-based URL classification for URL ingestion."""

from typing import Literal
from urllib.parse import urlsplit

UrlContentType = Literal["pdf", "html", "markdown", "text", "unknown"]


class UrlClassifier:
    """Classify normalized URLs by path suffix without network access."""

    def classify(self, normalized_url: str) -> UrlContentType:
        """Classify a normalized URL by inspecting its path suffix."""
        path = urlsplit(normalized_url).path
        if path.endswith(".pdf"):
            return "pdf"
        if path.endswith(".html") or path.endswith(".HTML"):
            return "html"
        if path.endswith(".md"):
            return "markdown"
        if path.endswith(".txt"):
            return "text"
        return "unknown"

    def classification_order(self) -> tuple[UrlContentType, ...]:
        """Return deterministic output order for classification statistics."""
        return ("pdf", "html", "markdown", "text", "unknown")
