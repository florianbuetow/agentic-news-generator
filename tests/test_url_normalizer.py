"""Unit tests for URL normalization."""

from src.url_ingestion.normalizer import NormalizedUrl, UnprocessableUrl, UrlNormalizer


def test_normalizer_trims_adds_scheme_and_lowercases_scheme_and_host() -> None:
    """Normalize missing-scheme URLs while preserving path casing."""
    result = UrlNormalizer().normalize("  Example.COM/Some/Path.HTML  ")

    assert isinstance(result, NormalizedUrl)
    assert result.original_url == "  Example.COM/Some/Path.HTML  "
    assert result.normalized_url == "http://example.com/Some/Path.HTML"


def test_normalizer_preserves_https_and_path_casing() -> None:
    """Keep explicit HTTPS URLs while lowercasing only scheme and host."""
    result = UrlNormalizer().normalize("HTTPS://Example.COM/MixedCase/File.pdf")

    assert isinstance(result, NormalizedUrl)
    assert result.normalized_url == "https://example.com/MixedCase/File.pdf"


def test_normalizer_rejects_unsupported_scheme() -> None:
    """Reject non-http URL schemes."""
    result = UrlNormalizer().normalize("ftp://example.com/file.pdf")

    assert isinstance(result, UnprocessableUrl)
    assert result.original_url == "ftp://example.com/file.pdf"
    assert result.reason == "unsupported URL scheme: ftp"


def test_normalizer_rejects_query_string() -> None:
    """Reject query strings before classification."""
    result = UrlNormalizer().normalize("https://example.com/file.pdf?download=true")

    assert isinstance(result, UnprocessableUrl)
    assert result.original_url == "https://example.com/file.pdf?download=true"
    assert result.reason == "URL contains a query string"


def test_normalizer_rejects_non_url_text_with_whitespace() -> None:
    """Reject non-URL text that contains whitespace."""
    result = UrlNormalizer().normalize("not a url")

    assert isinstance(result, UnprocessableUrl)
    assert result.reason == "URL contains whitespace"
