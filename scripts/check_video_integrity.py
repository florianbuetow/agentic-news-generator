#!/usr/bin/env python3
"""Check video files for corruption using ffprobe integrity checks."""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.config import Config
from src.util.log_util import configure_root_logger, get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {"mp4", "mkv", "webm", "m4a", "mov", "m4v", "avi", "flv"}
MIN_BITRATE_BPS = 1000
MIN_DURATION_FOR_BITRATE_CHECK = 60
FRESHNESS_THRESHOLD_SECONDS = 60
CACHE_SAVE_INTERVAL_SECONDS = 30
CACHE_SAVE_SLEEP_SECONDS = 3
PROGRESS_INTERVAL = 100


def _load_cache(cache_file: Path) -> dict[str, dict[str, int]]:
    """Load the integrity cache from a JSON file."""
    if not cache_file.exists():
        return {}
    try:
        result: dict[str, dict[str, int]] = json.loads(cache_file.read_text())
        return result
    except (json.JSONDecodeError, KeyError):
        return {}


def _save_cache(cache_file: Path, cache: dict[str, dict[str, int]]) -> None:
    """Atomically write the integrity cache to a JSON file."""
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    tmp.rename(cache_file)


def _check_duration(input_file: Path, timeout_cmd: str | None) -> str | None:
    """Run ffprobe to extract the duration of a media file."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(input_file)]
    if timeout_cmd:
        cmd = [timeout_cmd, "30"] + cmd
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
        return result.stdout.strip() or None
    except (subprocess.TimeoutExpired, OSError):
        return None


def _detect_timeout_cmd() -> str | None:
    """Find an available timeout command (GNU coreutils)."""
    for cmd in ("timeout", "gtimeout"):
        if shutil.which(cmd):
            return cmd
    return None


def _check_file(input_file: Path, timeout_cmd: str | None, channel_name: str, file_size: int) -> str | None:
    """Run integrity checks on a single file. Returns an error message or None if OK."""
    duration_str = _check_duration(input_file, timeout_cmd)

    if not duration_str or duration_str == "N/A":
        return f"  CORRUPT (unreadable): {channel_name}/{input_file.name} [{file_size} bytes]"

    duration_match = re.match(r"^(\d+(?:\.\d+)?)$", duration_str)
    if not duration_match:
        return f"  CORRUPT (invalid duration '{duration_str}'): {channel_name}/{input_file.name} [{file_size} bytes]"

    duration = float(duration_match.group(1))
    if duration > MIN_DURATION_FOR_BITRATE_CHECK:
        bitrate = file_size / duration
        if bitrate < MIN_BITRATE_BPS:
            return f"  CORRUPT (incomplete download): {channel_name}/{input_file.name} [{file_size}B / {duration:.1f}s = {bitrate:.0f} B/s]"

    return None


def _print_corrupt_files(corrupt_files: list[Path]) -> None:
    """Log the list of corrupt files with video IDs if extractable."""
    logger.error("")
    logger.error("Corrupt files:")
    for f in corrupt_files:
        match = re.search(r"\[([A-Za-z0-9_-]{11})\]\.[a-zA-Z0-9]+$", f.name)
        video_id = match.group(1) if match else None
        if video_id:
            logger.error(f"  {f} (video_id: {video_id})")
        else:
            logger.error(f"  {f}")


class _Stats:
    """Mutable counters shared across the scan."""

    def __init__(self, cache_file: Path) -> None:
        self.checked = 0
        self.corrupt = 0
        self.cached = 0
        self.corrupt_files: list[Path] = []
        self.new_cache: dict[str, dict[str, int]] = {}
        self._last_save = time.monotonic()
        self._cache_file = cache_file

    def maybe_save(self) -> None:
        """Save the cache if enough time has elapsed since the last save."""
        elapsed = time.monotonic() - self._last_save
        if elapsed >= CACHE_SAVE_INTERVAL_SECONDS:
            _save_cache(self._cache_file, self.new_cache)
            self._last_save = time.monotonic()
            logger.info(f"Updated {self._cache_file}, sleeping for {CACHE_SAVE_SLEEP_SECONDS} seconds...")
            time.sleep(CACHE_SAVE_SLEEP_SECONDS)


def _process_channel(
    channel_dir: Path,
    data_dir: Path,
    cache: dict[str, dict[str, int]],
    timeout_cmd: str | None,
    stats: _Stats,
) -> None:
    """Process all video files in a single channel directory."""
    channel_name = channel_dir.name
    channel_corrupt = 0
    channel_checked = 0

    logger.info(f"Processing channel: {channel_name}")

    files = sorted(
        f
        for f in channel_dir.iterdir()
        if f.is_file() and not f.name.startswith("._") and f.suffix.lstrip(".").lower() in ALLOWED_EXTENSIONS
    )

    now = time.time()

    for input_file in files:
        stat = input_file.stat()
        file_mtime = int(stat.st_mtime)
        file_size = stat.st_size

        if (now - file_mtime) < FRESHNESS_THRESHOLD_SECONDS:
            continue

        rel_path = str(input_file.relative_to(data_dir))
        total_processed = stats.checked + stats.cached + 1

        if total_processed % PROGRESS_INTERVAL == 0:
            logger.info(
                f"  ... processed {total_processed} files so far ({stats.cached} cached, {stats.checked} checked, {stats.corrupt} corrupt)"
            )

        cached_entry = cache.get(rel_path)
        if cached_entry and cached_entry["mtime"] == file_mtime and cached_entry["size"] == file_size:
            stats.cached += 1
            stats.new_cache[rel_path] = cached_entry
            continue

        stats.checked += 1
        channel_checked += 1

        error = _check_file(input_file, timeout_cmd, channel_name, file_size)
        if error:
            logger.error(error)
            stats.corrupt += 1
            channel_corrupt += 1
            stats.corrupt_files.append(input_file)
        else:
            stats.new_cache[rel_path] = {"mtime": file_mtime, "size": file_size}

        stats.maybe_save()

    if channel_checked > 0 and channel_corrupt > 0:
        logger.error(f"  {channel_name}: {channel_corrupt} corrupt out of {channel_checked} files")


def main() -> int:
    """Check all video files for corruption, using a cache to skip unchanged files."""
    if not shutil.which("ffprobe"):
        print("ERROR: ffprobe is not installed (should come with ffmpeg)", file=sys.stderr)
        return 1

    timeout_cmd = _detect_timeout_cmd()

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config(config_path)
    configure_root_logger(config.getDataLogsDir())

    data_dir = config.getDataDir()
    videos_dir = config.getDataDownloadsVideosDir()
    cache_file = data_dir / "video_integrity_cache.json"
    cache = _load_cache(cache_file)
    logger.info(f"Loaded {len(cache)} entries from integrity cache")

    logger.info("Checking video file integrity")
    logger.info("==========================================")
    logger.info("")

    stats = _Stats(cache_file)
    channel_dirs = sorted(p for p in videos_dir.iterdir() if p.is_dir() and not p.name.startswith("."))

    for channel_dir in channel_dirs:
        _process_channel(channel_dir, data_dir, cache, timeout_cmd, stats)

    _save_cache(cache_file, stats.new_cache)

    total_files = stats.checked + stats.cached
    logger.info("")
    logger.info("==========================================")
    logger.info(f"Total files:   {total_files}")
    logger.info(f"Total cached:  {stats.cached} (unchanged since last check)")
    logger.info(f"Total checked: {stats.checked}")
    logger.info(f"Total corrupt: {stats.corrupt}")
    logger.info(f"Total clean:   {total_files - stats.corrupt}")

    if stats.corrupt > 0:
        _print_corrupt_files(stats.corrupt_files)
        logger.error(f"check-video-integrity failed: {stats.corrupt} corrupt file(s) found")
        return 1

    logger.info("check-video-integrity passed: all files OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
