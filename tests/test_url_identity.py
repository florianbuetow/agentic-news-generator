"""Unit tests for URL identity helpers."""

from src.url_ingestion.identity import sanitize_normalized_url_to_stem


def test_sanitized_url_filename_generation() -> None:
    """Generate readable filesystem-safe stems from normalized URLs."""
    stem = sanitize_normalized_url_to_stem("https://example.com/docs/My File.pdf")

    assert stem == "https_example_com_docs_My_File_pdf"


def test_sanitized_url_stem_is_capped_at_180_characters() -> None:
    """Cap long URL-derived stems."""
    stem = sanitize_normalized_url_to_stem(f"https://example.com/{'a' * 300}.pdf")

    assert len(stem) == 180
