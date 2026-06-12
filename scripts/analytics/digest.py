#!/usr/bin/env python3
"""Run the full analytics chain: index, themes, timeline, research digest."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.analytics.digest import build_digest
from src.config import Config


def main() -> int:
    """Build all analytics artifacts; exceptions propagate (fail fast)."""
    config = Config(Path("config/config.yaml"))
    build_digest(config, date.today())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
