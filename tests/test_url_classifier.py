"""Unit tests for rule-based URL classification."""

from src.url_ingestion.classifier import UrlClassifier


def test_classifier_identifies_supported_suffixes() -> None:
    """Classify known URL path suffixes without network access."""
    classifier = UrlClassifier()

    assert classifier.classify("https://example.com/report.pdf") == "pdf"
    assert classifier.classify("https://example.com/article.html") == "html"
    assert classifier.classify("https://example.com/article.HTML") == "html"
    assert classifier.classify("https://example.com/article.htm") == "html"
    assert classifier.classify("https://example.com/readme.md") == "markdown"
    assert classifier.classify("https://example.com/notes.txt") == "text"
    assert classifier.classify("https://example.com/articles/llm-research") == "html"
    assert classifier.classify("https://example.com/") == "html"


def test_classifier_inspects_path_not_whole_url() -> None:
    """Ignore non-path URL components when classifying."""
    classifier = UrlClassifier()

    assert classifier.classify("https://example.com/download#manual.pdf") == "html"


def test_classifier_handles_suffix_case_insensitively() -> None:
    """Classify supported document suffixes regardless of case."""
    classifier = UrlClassifier()

    assert classifier.classify("https://example.com/report.PDF") == "pdf"
    assert classifier.classify("https://example.com/readme.MD") == "markdown"
    assert classifier.classify("https://example.com/notes.TXT") == "text"


def test_classifier_keeps_obvious_non_documents_unknown() -> None:
    """Avoid routing obvious binary and media assets through the HTML downloader."""
    classifier = UrlClassifier()

    assert classifier.classify("https://example.com/image.png") == "unknown"
    assert classifier.classify("https://example.com/video.mp4") == "unknown"
    assert classifier.classify("https://example.com/archive.zip") == "unknown"


def test_classifier_keeps_future_social_and_video_hosts_unknown() -> None:
    """Avoid sending social/video URLs through the generic HTML downloader."""
    classifier = UrlClassifier()

    assert classifier.classify("https://x.com/hamelhusain/status/123?s=12") == "unknown"
    assert classifier.classify("https://www.instagram.com/p/example/") == "unknown"
    assert classifier.classify("https://www.youtube.com/watch?v=abc") == "unknown"
    assert classifier.classify("https://youtu.be/abc") == "unknown"
