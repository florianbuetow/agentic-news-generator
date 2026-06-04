"""Tests for curl-style URL reachability diagnostics."""

from src.url_ingestion.reachability import ReachabilityResult


def test_reachability_result_summary_reports_http_details() -> None:
    """Summarize successful curl diagnostics compactly."""
    result = ReachabilityResult(
        http_status="200",
        final_url="https://example.com/final",
        content_type="text/html",
    )

    assert result.summary() == "curl_http_status=200, curl_content_type=text/html, curl_final_url=https://example.com/final"


def test_reachability_result_summary_reports_curl_errors() -> None:
    """Summarize curl execution errors compactly."""
    result = ReachabilityResult(http_status=None, final_url=None, content_type=None, error="timeout")

    assert result.summary() == "curl_error=timeout"
