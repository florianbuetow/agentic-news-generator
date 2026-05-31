"""Unit tests for rule-based URL classification."""

from src.url_ingestion.classifier import UrlClassifier


def test_classifier_identifies_supported_suffixes() -> None:
    """Classify known URL path suffixes without network access."""
    classifier = UrlClassifier()

    assert classifier.classify("https://example.com/report.pdf") == "pdf"
    assert classifier.classify("https://example.com/article.html") == "html"
    assert classifier.classify("https://example.com/article.HTML") == "html"
    assert classifier.classify("https://example.com/readme.md") == "markdown"
    assert classifier.classify("https://example.com/notes.txt") == "text"


def test_classifier_inspects_path_not_whole_url() -> None:
    """Ignore non-path URL components when classifying."""
    classifier = UrlClassifier()

    assert classifier.classify("https://example.com/download#manual.pdf") == "unknown"


def test_classifier_case_handling_is_explicit() -> None:
    """Only the configured uppercase HTML suffix variant is classified specially."""
    classifier = UrlClassifier()

    assert classifier.classify("https://example.com/report.PDF") == "unknown"
    assert classifier.classify("https://example.com/readme.MD") == "unknown"
    assert classifier.classify("https://example.com/notes.TXT") == "unknown"
