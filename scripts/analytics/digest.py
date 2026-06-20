#!/usr/bin/env python3
"""Run the full analytics chain: index, themes, timeline, research digest."""

from datetime import date

from src.analytics.digest import build_digest
from src.config import Config


def main() -> int:
    """Build all analytics artifacts; exceptions propagate (fail fast)."""
    config = Config.load_default()
    build_digest(config, date.today())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
