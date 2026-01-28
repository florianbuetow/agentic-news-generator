#!/usr/bin/env python3
"""Find and optionally delete empty files in the data folder."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config


def find_empty_files(data_dir: Path) -> list[Path]:
    """Find all empty files in the data directory.

    Args:
        data_dir: Root data directory to scan.

    Returns:
        List of paths to empty files.
    """
    empty_files: list[Path] = []

    # Files to ignore
    ignore_names = {".gitkeep", ".DS_Store"}

    for file_path in data_dir.rglob("*"):
        # Skip directories
        if not file_path.is_file():
            continue

        # Skip macOS metadata files
        if file_path.name.startswith("._"):
            continue

        # Skip ignored files
        if file_path.name in ignore_names:
            continue

        # Check if empty
        if file_path.stat().st_size == 0:
            empty_files.append(file_path)

    return sorted(empty_files)


def print_report(empty_files: list[Path], data_dir: Path) -> None:
    """Print a report of empty files grouped by extension.

    Args:
        empty_files: List of empty file paths.
        data_dir: Root data directory for computing relative paths.
    """
    # Group by extension
    by_extension: dict[str, list[Path]] = {}
    for f in empty_files:
        ext = f.suffix or "(no extension)"
        by_extension.setdefault(ext, []).append(f)

    print(f"Found {len(empty_files)} empty file(s):")
    print()

    # Print summary by extension
    for ext, files in sorted(by_extension.items(), key=lambda x: -len(x[1])):
        print(f"  {len(files):4d} {ext}")
    print()

    # Print all files (relative to data directory)
    print("Files:")
    for f in empty_files:
        print(f"  {f.relative_to(data_dir)}")
    print()


def delete_files(empty_files: list[Path]) -> None:
    """Delete the specified files with error handling.

    Args:
        empty_files: List of file paths to delete.
    """
    deleted = 0
    for f in empty_files:
        try:
            f.unlink()
            deleted += 1
        except OSError as e:
            print(f"  Error deleting {f}: {e}", file=sys.stderr)
    print(f"Deleted {deleted} file(s)")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    try:
        config = Config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data_dir = config.getDataDir()

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    print()
    print("=== Scanning for Empty Files ===")
    print(f"Data directory: {data_dir}")
    print()

    empty_files = find_empty_files(data_dir)

    if not empty_files:
        print("No empty files found.")
        return 0

    print_report(empty_files, data_dir)

    # Ask for confirmation
    try:
        confirm = input("Delete these files? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled")
        return 0

    if confirm == "y":
        delete_files(empty_files)
    else:
        print("Cancelled")

    return 0


if __name__ == "__main__":
    sys.exit(main())
