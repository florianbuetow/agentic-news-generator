"""Unit tests for URL identity helpers."""

from src.url_ingestion.identity import MAX_FILENAME_STEM_BYTES, sanitize_normalized_url_to_stem


def test_sanitized_url_filename_generation() -> None:
    """Generate readable filesystem-safe stems from normalized URLs."""
    stem = sanitize_normalized_url_to_stem("https://example.com/docs/My File.pdf")

    assert stem == "https_example_com_docs_My_File_pdf"


def test_sanitized_url_filename_generation_includes_query_identity() -> None:
    """Keep query parameters in the stem so distinct normalized URLs do not overwrite."""
    stem = sanitize_normalized_url_to_stem("https://example.com/search?q=llm&page=1")

    assert stem == "https_example_com_search_q_llm_page_1"


def test_sanitized_url_stem_keeps_long_readable_identity_when_filesystem_safe() -> None:
    """Avoid the old arbitrary 180-character truncation."""
    stem = sanitize_normalized_url_to_stem(f"https://example.com/{'a' * 190}.pdf")

    assert stem == f"https_example_com_{'a' * 190}_pdf"


def test_sanitized_url_stem_shortens_overlong_names_with_hash_suffix() -> None:
    """Stay within filesystem filename limits without creating false collisions."""
    stem = sanitize_normalized_url_to_stem(f"https://example.com/{'a' * 300}.pdf")

    assert len(stem.encode("utf-8")) <= MAX_FILENAME_STEM_BYTES
    assert stem.startswith("https_example_com_")
    assert stem.endswith("_eca4d5673e6d9db0")


def test_sanitized_url_stem_hash_suffix_distinguishes_long_urls_with_same_prefix() -> None:
    """Distinct overlong URLs sharing a long prefix should not collapse together."""
    first = sanitize_normalized_url_to_stem(f"https://example.com/{'a' * 300}.pdf?version=1")
    second = sanitize_normalized_url_to_stem(f"https://example.com/{'a' * 300}.pdf?version=2")

    assert first != second
    assert len(first.encode("utf-8")) <= MAX_FILENAME_STEM_BYTES
    assert len(second.encode("utf-8")) <= MAX_FILENAME_STEM_BYTES
