"""Curl-style URL reachability diagnostics for failed downloads."""

import subprocess
from dataclasses import dataclass
from typing import Protocol

from src.url_ingestion.downloader import CONTENT_FETCH_TIMEOUT_SECONDS


@dataclass(frozen=True)
class ReachabilityResult:
    """Result from a lightweight URL reachability check."""

    http_status: str | None
    final_url: str | None
    content_type: str | None
    error: str | None = None

    def summary(self) -> str:
        """Return a compact diagnostic string for logs and archives."""
        if self.error:
            return f"curl_error={self.error}"
        parts: list[str] = []
        if self.http_status:
            parts.append(f"curl_http_status={self.http_status}")
        if self.content_type:
            parts.append(f"curl_content_type={self.content_type}")
        if self.final_url:
            parts.append(f"curl_final_url={self.final_url}")
        return ", ".join(parts) if parts else "curl_no_diagnostic"


class ReachabilityProbe(Protocol):
    """Diagnostic probe used after a download failure."""

    def check(self, url: str) -> ReachabilityResult:
        """Check whether a URL is reachable outside the downloader implementation."""
        ...


class CurlReachabilityProbe:
    """Run curl HEAD diagnostics with redirects and the shared URL timeout."""

    def __init__(self) -> None:
        """Initialize the probe with a timeout."""
        self._timeout_seconds = CONTENT_FETCH_TIMEOUT_SECONDS

    def check(self, url: str) -> ReachabilityResult:
        """Run curl and parse HTTP status, final URL, and content type."""
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "-I",
                    "-L",
                    "--max-time",
                    str(self._timeout_seconds),
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}\n%{url_effective}\n%{content_type}",
                    url,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            return ReachabilityResult(http_status=None, final_url=None, content_type=None, error=str(exc))

        if completed.returncode != 0:
            error = completed.stderr.strip()
            if not error:
                error = f"curl exited {completed.returncode}"
            return ReachabilityResult(http_status=None, final_url=None, content_type=None, error=error)

        lines = completed.stdout.splitlines()
        return ReachabilityResult(
            http_status=lines[0].strip() if len(lines) >= 1 and lines[0].strip() else None,
            final_url=lines[1].strip() if len(lines) >= 2 and lines[1].strip() else None,
            content_type=lines[2].strip() if len(lines) >= 3 and lines[2].strip() else None,
        )
