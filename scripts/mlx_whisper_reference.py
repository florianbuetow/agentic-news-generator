#!/usr/bin/env python3
"""Reference import for mlx_whisper dependency detection.

This file exists solely to ensure deptry can detect that mlx_whisper is used
in the project. The actual usage is in scripts/transcribe_audio.sh which calls
the mlx_whisper CLI via 'uv run mlx_whisper', but deptry cannot scan shell scripts.

This script is never executed - it only serves as a dependency marker.
"""

import mlx_whisper


def _reference_mlx_whisper() -> None:
    """Reference function to satisfy linters.

    This function is never called but ensures the import is considered used.
    """
    _ = mlx_whisper.__version__
