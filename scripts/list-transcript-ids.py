#!/usr/bin/env python3
"""Find all transcript files and extract video IDs from their filenames."""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config

ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


def main() -> None:
    """List transcript filenames and whether each filename includes a video ID."""
    project_root = Path(__file__).parent.parent
    config = Config(project_root / "config" / "config.yaml")
    transcripts_dir = config.getDataDownloadsTranscriptsDir()

    all_files = subprocess.run(
        ["find", str(transcripts_dir), "-type", "f"],
        capture_output=True,
        text=True,
    ).stdout

    with_id = 0
    without_id = 0

    for line in sorted(all_files.splitlines()):
        if not line:
            continue
        filename = Path(line).name
        match = ID_RE.search(filename)
        if match:
            print(f"{match.group(1)}  {filename}")
            with_id += 1
        else:
            print(f"NO_ID     {filename}")
            without_id += 1

    print(f"\n{'=' * 70}")
    print(f"With ID: {with_id}, Without ID: {without_id}, Total: {with_id + without_id}")


if __name__ == "__main__":
    main()
