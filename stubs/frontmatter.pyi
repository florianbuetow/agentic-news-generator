"""Type stubs for python-frontmatter library."""

from typing import Any

class Post:
    """Frontmatter post object."""

    metadata: dict[str, Any]
    content: str

def load(fd: str) -> Post:
    """Load frontmatter from file path."""
    ...
