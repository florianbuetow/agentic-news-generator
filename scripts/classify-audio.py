#!/usr/bin/env python3
"""Create speech/music/other classification maps for extracted audio files.

For every WAV file below the configured audio directory, this writes one JSON
map with per-class [start, end) second intervals on a 100 ms grid into the
configured audio classification directory. Existing maps are skipped; the two
ONNX models download once into the models directory on first run.
"""

from __future__ import annotations

import sys

from src.audio_classification.runner import run_with_onnx_models
from src.config import Config


def main() -> int:
    """Load project configuration and run the audio classification batch."""
    try:
        config = Config.load_default()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    try:
        audio_dir = config.get_data_downloads_audio_dir()
        config.get_data_downloads_audio_classification_dir()
        config.get_audio_classification_config()
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not audio_dir.is_dir():
        print(f"Error: audio directory not found: {audio_dir}", file=sys.stderr)
        return 1

    try:
        return run_with_onnx_models(config)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
