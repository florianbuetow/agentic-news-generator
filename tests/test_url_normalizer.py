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


def test_normalizer_removes_root_path_slash() -> None:
    """Normalize equivalent site-root URLs to one identity."""
    result = UrlNormalizer().normalize("https://Example.COM/")

    assert isinstance(result, NormalizedUrl)
    assert result.normalized_url == "https://example.com"


def test_normalizer_rejects_unsupported_scheme() -> None:
    """Reject non-http URL schemes."""
    result = UrlNormalizer().normalize("ftp://example.com/file.pdf")

    assert isinstance(result, UnprocessableUrl)
    assert result.original_url == "ftp://example.com/file.pdf"
    assert result.reason == "unsupported URL scheme: ftp"


def test_normalizer_preserves_query_string_and_strips_fragment() -> None:
    """Preserve query strings for downloadable URLs and strip client-side fragments."""
    result = UrlNormalizer().normalize("https://example.com/file.pdf?download=true#section")

    assert isinstance(result, NormalizedUrl)
    assert result.original_url == "https://example.com/file.pdf?download=true#section"
    assert result.normalized_url == "https://example.com/file.pdf?download=true"


def test_normalizer_converts_arxiv_abstract_links_to_pdf_links() -> None:
    """Route arXiv abstract links to direct PDF downloads."""
    result = UrlNormalizer().normalize("https://arxiv.org/abs/2401.12345v2")

    assert isinstance(result, NormalizedUrl)
    assert result.normalized_url == "https://arxiv.org/pdf/2401.12345v2.pdf"


def test_normalizer_adds_pdf_suffix_to_arxiv_pdf_links_without_filename_suffix() -> None:
    """Normalize arXiv PDF links that omit the .pdf filename suffix."""
    result = UrlNormalizer().normalize("https://www.arxiv.org/pdf/2401.12345")

    assert isinstance(result, NormalizedUrl)
    assert result.normalized_url == "https://arxiv.org/pdf/2401.12345.pdf"


def test_normalizer_rejects_non_url_text_with_whitespace() -> None:
    """Reject non-URL text that contains whitespace."""
    result = UrlNormalizer().normalize("not a url")

    assert isinstance(result, UnprocessableUrl)
    assert result.reason == "URL contains whitespace"
