#!/usr/bin/env python3
"""Clean up all files associated with a YouTube video ID.

Finds every file in the data directory whose name contains `[<video_id>]`,
lists them, asks the user which to delete, and then offers to remove the
video ID from any `downloaded.txt` yt-dlp archive file that references it.

Intended for targeted cleanup of individual videos (e.g. corrupt transcripts)
without touching unrelated data. Every destructive action is explicitly
confirmed by the user.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config

# YouTube video IDs: exactly 11 chars of [A-Za-z0-9_-]
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def validate_video_id(video_id: str) -> str:
    """Validate a YouTube video ID string.

    Args:
        video_id: Candidate video ID.

    Returns:
        The validated video ID (unchanged).

    Raises:
        ValueError: If the video ID is not 11 chars of [A-Za-z0-9_-].
    """
    if not _VIDEO_ID_RE.match(video_id):
        raise ValueError(f"Invalid YouTube video ID: {video_id!r}. Expected 11 characters of [A-Za-z0-9_-].")
    return video_id


def find_files_for_video(data_dir: Path, video_id: str) -> list[Path]:
    """Find every file under data_dir whose name contains `[<video_id>]`.

    Skips macOS metadata sidecars (`._*`). Does not include `downloaded.txt`
    archive files — those are handled separately so they can be edited in
    place rather than deleted.

    Args:
        data_dir: Root directory to scan.
        video_id: YouTube video ID to search for.

    Returns:
        Sorted list of matching file paths.
    """
    marker = f"[{video_id}]"
    matches: list[Path] = []
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if marker in path.name:
            matches.append(path)
    return sorted(matches)


def find_archives_with_video(data_dir: Path, video_id: str) -> list[Path]:
    """Find every `downloaded.txt` archive file that references the video ID.

    Args:
        data_dir: Root directory to scan.
        video_id: YouTube video ID to search for.

    Returns:
        Sorted list of matching archive paths.
    """
    target = f"youtube {video_id}"
    matches: list[Path] = []
    for archive in data_dir.rglob("downloaded.txt"):
        if not archive.is_file():
            continue
        try:
            content = archive.read_text(encoding="utf-8")
        except OSError as e:
            print(f"  ⚠️  Could not read {archive}: {e}", file=sys.stderr)
            continue
        if any(line.strip() == target for line in content.splitlines()):
            matches.append(archive)
    return sorted(matches)


def parse_selection(selection: str, count: int) -> list[int] | None:
    """Parse a user selection string into a list of 0-based indices.

    Accepts:
      - "" / "none" / "n"  -> []
      - "all" / "a"        -> [0, 1, ..., count-1]
      - "1,3,5"            -> [0, 2, 4]
      - "1-3"              -> [0, 1, 2]
      - Mixed, e.g. "1,3-5"

    Args:
        selection: User-entered selection string.
        count: Total number of items available (1..count are valid).

    Returns:
        Sorted list of 0-based indices, or None on parse error.
    """
    text = selection.strip().lower()
    if text in ("", "none", "n"):
        return []
    if text in ("all", "a"):
        return list(range(count))

    indices: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                return None
            if start < 1 or end < 1 or start > count or end > count or start > end:
                return None
            indices.update(range(start - 1, end))
        else:
            try:
                i = int(token)
            except ValueError:
                return None
            if i < 1 or i > count:
                return None
            indices.add(i - 1)
    return sorted(indices)


def prompt_selection(count: int) -> list[int]:
    """Prompt the user for a selection of file indices to delete.

    Keeps prompting until a valid selection is entered or the user aborts
    via Ctrl-C / EOF (treated as "none").

    Args:
        count: Number of files available to choose from.

    Returns:
        Sorted list of 0-based indices to delete.
    """
    while True:
        try:
            raw = input("Which files should be deleted? [all / none / comma-separated 1-based indices, e.g. 1,3-5]: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return []
        parsed = parse_selection(raw, count)
        if parsed is None:
            print("  Invalid selection. Try again.")
            continue
        return parsed


def prompt_yes_no(question: str) -> bool:
    """Prompt the user with a yes/no question (default: no).

    Args:
        question: Question to display.

    Returns:
        True only if the user answered "y".
    """
    try:
        answer = input(f"{question} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer == "y"


def delete_files(files: list[Path], data_dir: Path) -> tuple[int, int]:
    """Delete files one by one, reporting each result.

    Args:
        files: Files to delete.
        data_dir: Root directory for relative-path display.

    Returns:
        Tuple of (deleted_count, failed_count).
    """
    deleted = 0
    failed = 0
    for f in files:
        try:
            rel = f.relative_to(data_dir)
        except ValueError:
            rel = f
        try:
            f.unlink()
            print(f"  ✓ Deleted: {rel}")
            deleted += 1
        except OSError as e:
            print(f"  ✗ Failed to delete {rel}: {e}", file=sys.stderr)
            failed += 1
    return deleted, failed


def remove_video_from_archive(archive: Path, video_id: str) -> int:
    """Remove all `youtube <video_id>` lines from an archive file.

    Preserves the original trailing-newline convention so diffs stay small.

    Args:
        archive: Path to the `downloaded.txt` archive.
        video_id: Video ID whose line(s) should be removed.

    Returns:
        Number of lines removed. 0 means the archive was not modified.
    """
    try:
        content = archive.read_text(encoding="utf-8")
    except OSError as e:
        print(f"  ✗ Could not read {archive}: {e}", file=sys.stderr)
        return 0

    target = f"youtube {video_id}"
    original_lines = content.splitlines()
    kept_lines = [line for line in original_lines if line.strip() != target]
    removed = len(original_lines) - len(kept_lines)

    if removed == 0:
        return 0

    new_content = "\n".join(kept_lines)
    if new_content and content.endswith("\n"):
        new_content += "\n"

    try:
        archive.write_text(new_content, encoding="utf-8")
    except OSError as e:
        print(f"  ✗ Could not write {archive}: {e}", file=sys.stderr)
        return 0

    return removed


def relative_display(path: Path, data_dir: Path) -> Path:
    """Return path relative to data_dir, or the path itself if outside it.

    Args:
        path: Absolute file path.
        data_dir: Root data directory.

    Returns:
        Relative path for display.
    """
    try:
        return path.relative_to(data_dir)
    except ValueError:
        return path


def handle_file_deletion(files: list[Path], video_id: str, data_dir: Path) -> bool:
    """List matching files, prompt the user for selection, and delete chosen files.

    Args:
        files: Files whose name contains `[<video_id>]`.
        video_id: The video ID being cleaned.
        data_dir: Root data directory for relative-path display.

    Returns:
        True if all selected deletions succeeded (or none selected), False on failure.
    """
    if not files:
        print(f"No files found containing [{video_id}] in their name.")
        return True

    print(f"Found {len(files)} file(s) containing [{video_id}]:")
    print()
    for i, f in enumerate(files, start=1):
        rel = relative_display(f, data_dir)
        size = f.stat().st_size
        print(f"  [{i:2d}] ({size:>12,d} B) {rel}")
    print()

    indices = prompt_selection(len(files))
    if not indices:
        print("No files selected for deletion.")
        return True

    to_delete = [files[i] for i in indices]
    print()
    print(f"Deleting {len(to_delete)} file(s):")
    deleted, failed = delete_files(to_delete, data_dir)
    print()
    print(f"Summary: {deleted} deleted, {failed} failed")
    return failed == 0


def handle_archive_cleanup(archives: list[Path], video_id: str, data_dir: Path) -> bool:
    """Offer to remove the video ID from each matching downloaded.txt archive.

    Args:
        archives: Archive files that reference the video ID.
        video_id: The video ID to remove.
        data_dir: Root data directory for relative-path display.

    Returns:
        True if all requested edits succeeded, False on failure.
    """
    if not archives:
        print(f"No downloaded.txt archives contain 'youtube {video_id}'.")
        return True

    print(f"Found {len(archives)} archive(s) referencing {video_id}:")
    for a in archives:
        print(f"  - {relative_display(a, data_dir)}")
    print()

    all_ok = True
    for archive in archives:
        rel = relative_display(archive, data_dir)
        if not prompt_yes_no(f"Remove 'youtube {video_id}' from {rel}?"):
            print(f"  ⊘ Skipped {rel}")
            continue
        removed = remove_video_from_archive(archive, video_id)
        if removed > 0:
            print(f"  ✓ Removed {removed} line(s) from {rel}")
        else:
            print(f"  ✗ No changes made to {rel}")
            all_ok = False

    return all_ok


def main() -> int:
    """Run the interactive cleanup flow.

    Returns:
        Exit code (0 on success, 1 on error).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Delete all files whose names contain [<video_id>] under the data "
            "directory, and optionally remove the video ID from the channel's "
            "downloaded.txt yt-dlp archive file(s)."
        ),
    )
    parser.add_argument(
        "video_id",
        help="YouTube video ID (11 chars, as it appears in filenames inside [])",
    )
    args = parser.parse_args()

    try:
        video_id = validate_video_id(args.video_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    try:
        config = Config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    data_dir = config.getDataDir().resolve()
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    print()
    print("=== Video File Cleaner ===")
    print(f"Data directory: {data_dir}")
    print(f"Video ID:       {video_id}")
    print()

    files = find_files_for_video(data_dir, video_id)
    if not handle_file_deletion(files, video_id, data_dir):
        return 1

    print()

    archives = find_archives_with_video(data_dir, video_id)
    if not handle_archive_cleanup(archives, video_id, data_dir):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
