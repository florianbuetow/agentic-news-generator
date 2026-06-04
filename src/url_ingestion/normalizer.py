"""Normalize URL inbox entries before URL ingestion processing."""

from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit, urlunsplit


@dataclass(frozen=True)
class NormalizedUrl:
    """A successfully normalized URL inbox entry."""

    original_url: str
    normalized_url: str


@dataclass(frozen=True)
class UnprocessableUrl:
    """A URL inbox entry that cannot be processed."""

    original_url: str
    reason: str


class UrlNormalizer:
    """Normalize raw URL lines according to the configured initial URL rules."""

    def normalize(self, raw_url: str) -> NormalizedUrl | UnprocessableUrl:
        """Normalize one raw URL line or return an unprocessable reason."""
        original_url = raw_url
        trimmed_url = raw_url.strip()
        if not trimmed_url:
            return UnprocessableUrl(original_url=original_url, reason="empty URL")
        if any(character.isspace() for character in trimmed_url):
            return UnprocessableUrl(original_url=original_url, reason="URL contains whitespace")

        candidate_url = self._ensure_scheme(trimmed_url)
        split_url = urlsplit(candidate_url)
        if split_url.scheme not in {"http", "https"}:
            return UnprocessableUrl(original_url=original_url, reason=f"unsupported URL scheme: {split_url.scheme}")
        if not split_url.netloc or split_url.hostname is None:
            return UnprocessableUrl(original_url=original_url, reason="URL is missing a host")
        if not self._is_valid_host(split_url.hostname):
            return UnprocessableUrl(original_url=original_url, reason="URL host is not valid")
        if not self._has_valid_port(split_url):
            return UnprocessableUrl(original_url=original_url, reason="URL port is not valid")

        normalized_split = self._normalize_split_url(split_url)
        normalized_split = self._normalize_arxiv_pdf_url(normalized_split)
        return NormalizedUrl(original_url=original_url, normalized_url=urlunsplit(normalized_split))

    def _ensure_scheme(self, raw_url: str) -> str:
        """Prepend http:// when a URL has no scheme."""
        split_url = urlsplit(raw_url)
        if split_url.scheme:
            return raw_url
        return f"http://{raw_url}"

    def _normalize_split_url(self, split_url: SplitResult) -> SplitResult:
        """Lowercase scheme and host while preserving path casing."""
        hostname = split_url.hostname
        if hostname is None:
            raise ValueError("URL is missing a host")

        normalized_host = hostname.lower()
        normalized_netloc = normalized_host
        if split_url.port is not None:
            normalized_netloc = f"{normalized_host}:{split_url.port}"

        path = "" if split_url.path == "/" else split_url.path
        return SplitResult(
            scheme=split_url.scheme.lower(),
            netloc=normalized_netloc,
            path=path,
            query=split_url.query,
            fragment="",
        )

    def _normalize_arxiv_pdf_url(self, split_url: SplitResult) -> SplitResult:
        """Convert arXiv abstract URLs to direct PDF URLs."""
        hostname = split_url.hostname
        if hostname is None:
            hostname = ""
        normalized_hostname = hostname.lower().removeprefix("www.")
        if normalized_hostname != "arxiv.org":
            return split_url

        split_url = split_url._replace(netloc="arxiv.org")
        path = split_url.path
        if path.startswith("/abs/"):
            arxiv_id = path.removeprefix("/abs/").rstrip("/")
            return split_url._replace(path=f"/pdf/{arxiv_id}.pdf")
        if path.startswith("/pdf/") and not path.lower().endswith(".pdf"):
            arxiv_id = path.removeprefix("/pdf/").rstrip("/")
            return split_url._replace(path=f"/pdf/{arxiv_id}.pdf")
        return split_url

    def _is_valid_host(self, hostname: str) -> bool:
        """Validate the minimal host shape needed to reject plain text inputs."""
        return hostname == "localhost" or "." in hostname

    def _has_valid_port(self, split_url: SplitResult) -> bool:
        """Return whether the URL port can be parsed."""
        try:
            parsed_port = split_url.port
        except ValueError:
            return False
        return parsed_port is None or parsed_port >= 0
