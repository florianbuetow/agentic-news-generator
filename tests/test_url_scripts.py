"""Smoke tests for URL operational scripts."""

import py_compile
from pathlib import Path


def test_url_operational_scripts_compile() -> None:
    """Catch syntax errors in thin URL pipeline script entrypoints."""
    project_root = Path(__file__).parent.parent
    for relative_path in (
        "scripts/urls-download.py",
        "scripts/urls-cleancontent.py",
        "scripts/urls-requeue-unprocessed.py",
    ):
        py_compile.compile(str(project_root / relative_path), doraise=True)
