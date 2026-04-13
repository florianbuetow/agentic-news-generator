#!/usr/bin/env python3
"""Add every downloaded file with no audio track or below a duration threshold to filefilter.json.

Scans both ``data_downloads_videos_dir/<channel>/`` and
``data_downloads_audio_dir/<channel>/`` for each channel (or a single
channel via ``--channel``).

For each video file:
- reads the matching ``.info.json`` under
  ``data_downloads_metadata_dir/<channel>/video/`` for duration
- runs ``ffprobe`` to check whether the container has any audio stream

For each ``.wav`` in the audio dir with no corresponding video file
(wav-only orphan), runs ``ffprobe`` on the wav for duration.

Any file whose duration is below ``--max-duration`` (default 120s), or any
video file with no audio stream, gets a ``<channel>/<video_id>`` entry
added to ``config/filefilter.json`` under ``data_downloads_audio_dir``.

Does **not** delete any files. Pair with
``scripts/remove-filtered-files.py --execute`` to sweep the newly-filtered
files off disk (together with their upstream copies).

Default is a dry-run that prints the entries it would add. Pass ``--write``
to actually modify ``filefilter.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config

_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]\.[^.]+$")
_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
_FILTER_KEY = "data_downloads_audio_dir"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--max-duration",
        type=int,
        default=120,
        help="Mark any file whose duration (seconds) is strictly less than this. Default: 120.",
    )
    ap.add_argument(
        "--channel",
        default=None,
        help="Restrict to a single channel name. Default: all channels.",
    )
    ap.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml",
    )
    ap.add_argument(
        "--filter",
        default="config/filefilter.json",
        help="Path to filefilter.json",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Actually write the updated filter file (default: dry-run).",
    )
    return ap.parse_args()


def _video_id_from_name(name: str) -> str | None:
    """Extract the 11-char YouTube id from a yt-dlp output filename."""
    m = _ID_RE.search(name)
    return m.group(1) if m is not None else None


def find_info_json(info_sub: Path, video_id: str) -> Path | None:
    """Return the ``.info.json`` path for ``video_id`` inside ``info_sub``, or None."""
    if not info_sub.is_dir():
        return None
    token = f"[{video_id}]"
    for p in info_sub.iterdir():
        if p.is_file() and token in p.name and p.name.endswith(".info.json"):
            return p
    return None


def _read_info_duration(info_path: Path) -> int | None:
    """Return the ``duration`` field from ``info.json``, or None on failure."""
    try:
        info = json.load(info_path.open())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: cannot read {info_path}: {exc}", file=sys.stderr)
        return None
    duration = info.get("duration")
    return duration if isinstance(duration, int) else None


def _ffprobe_duration(path: Path) -> int | None:
    """Return duration in whole seconds from ``ffprobe``, or None on failure."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"WARN: ffprobe duration failed on {path}: {exc}", file=sys.stderr)
        return None
    s = out.stdout.strip()
    try:
        return int(float(s))
    except ValueError:
        return None


def _ffprobe_has_audio(path: Path) -> bool:
    """Return True if ``path`` contains at least one audio stream.

    Returns True on probe failure so we never *add* a file to the filter just
    because ffprobe misbehaved.
    """
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"WARN: ffprobe audio-check failed on {path}: {exc}", file=sys.stderr)
        return True
    return bool(out.stdout.strip())


def iter_channel_dirs(root: Path, only: str | None) -> list[Path]:
    """Return every channel dir to scan under ``root``, or just ``only`` if given."""
    if only is not None:
        return [root / only]
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _classify_video(
    video_path: Path,
    info_sub: Path,
    max_duration: int,
) -> tuple[int | None, str | None, bool]:
    """Classify one video file as ``(duration, reason_or_None, missing_info)``."""
    video_id = _video_id_from_name(video_path.name)
    if video_id is None:
        return None, None, False
    info_path = find_info_json(info_sub, video_id)
    duration: int | None = None
    missing_info = False
    if info_path is None:
        missing_info = True
    else:
        duration = _read_info_duration(info_path)

    short = duration is not None and duration < max_duration
    no_audio = not _ffprobe_has_audio(video_path)
    if short and no_audio:
        return duration, "short+no_audio", missing_info
    if short:
        return duration, "short", missing_info
    if no_audio:
        return duration, "no_audio", missing_info
    return duration, None, missing_info


def _scan_video_channel(
    ch_dir: Path,
    metadata_dir: Path,
    max_duration: int,
) -> tuple[list[tuple[str, str, int | None, str]], set[str], int, int]:
    """Scan one channel's video dir.

    Returns ``(matches, seen_ids, scanned, missing_info)``. ``seen_ids`` is
    every 11-char id found in the video dir so the audio-dir pass can skip
    them.
    """
    matches: list[tuple[str, str, int | None, str]] = []
    seen_ids: set[str] = set()
    scanned = 0
    missing_info = 0
    channel = ch_dir.name
    info_sub = metadata_dir / channel / "video"

    for video in sorted(ch_dir.iterdir()):
        if not video.is_file() or video.name.startswith("._"):
            continue
        if video.suffix.lower() not in _VIDEO_EXTS:
            continue
        video_id = _video_id_from_name(video.name)
        if video_id is None:
            continue
        seen_ids.add(video_id)
        scanned += 1
        duration, reason, missing = _classify_video(video, info_sub, max_duration)
        if missing:
            missing_info += 1
        if reason is not None:
            matches.append((channel, video_id, duration, reason))
    return matches, seen_ids, scanned, missing_info


def _scan_audio_channel(
    ch_dir: Path,
    seen_ids: set[str],
    max_duration: int,
) -> tuple[list[tuple[str, str, int | None, str]], int]:
    """Scan one channel's audio dir for wav-only orphans below ``max_duration``."""
    matches: list[tuple[str, str, int | None, str]] = []
    scanned = 0
    if not ch_dir.is_dir():
        return matches, scanned
    channel = ch_dir.name
    for wav in sorted(ch_dir.iterdir()):
        if not wav.is_file() or wav.name.startswith("._"):
            continue
        if wav.suffix.lower() != ".wav":
            continue
        video_id = _video_id_from_name(wav.name)
        if video_id is None or video_id in seen_ids:
            continue
        scanned += 1
        duration = _ffprobe_duration(wav)
        if duration is not None and duration < max_duration:
            matches.append((channel, video_id, duration, "short_wav_orphan"))
    return matches, scanned


def collect_filter_matches(
    videos_dir: Path,
    audio_dir: Path,
    metadata_dir: Path,
    channel_only: str | None,
    max_duration: int,
) -> tuple[list[tuple[str, str, int | None, str]], dict[str, int]]:
    """Walk every channel and return ``(matches, counters)`` across both dirs.

    ``counters`` keys: ``channels``, ``videos_scanned``, ``audio_scanned``,
    ``missing_info``.
    """
    all_matches: list[tuple[str, str, int | None, str]] = []
    counters = {"channels": 0, "videos_scanned": 0, "audio_scanned": 0, "missing_info": 0}

    video_channels = iter_channel_dirs(videos_dir, channel_only)
    video_names_seen: set[str] = set()
    for v_ch in video_channels:
        if not v_ch.is_dir():
            print(f"WARN: channel dir not found: {v_ch}", file=sys.stderr)
            continue
        counters["channels"] += 1
        video_names_seen.add(v_ch.name)
        v_matches, seen_ids, v_scanned, v_missing = _scan_video_channel(v_ch, metadata_dir, max_duration)
        all_matches.extend(v_matches)
        counters["videos_scanned"] += v_scanned
        counters["missing_info"] += v_missing
        a_matches, a_scanned = _scan_audio_channel(audio_dir / v_ch.name, seen_ids, max_duration)
        all_matches.extend(a_matches)
        counters["audio_scanned"] += a_scanned

    for a_ch in iter_channel_dirs(audio_dir, channel_only):
        if a_ch.name in video_names_seen or not a_ch.is_dir():
            continue
        counters["channels"] += 1
        a_matches, a_scanned = _scan_audio_channel(a_ch, set(), max_duration)
        all_matches.extend(a_matches)
        counters["audio_scanned"] += a_scanned

    return all_matches, counters


def _print_report(
    matches: list[tuple[str, str, int | None, str]],
    counters: dict[str, int],
    max_duration: int,
    write_mode: bool,
    already: set[str],
    to_add: list[str],
) -> None:
    """Print dry-run / write summary and the new entries grouped by channel."""
    print("=" * 80)
    print(f"{'WRITE' if write_mode else 'DRY RUN'} — filter-short-videos")
    print("=" * 80)
    print(f"max_duration:        < {max_duration}s")
    print(f"channels scanned:    {counters['channels']}")
    print(f"videos scanned:      {counters['videos_scanned']}")
    print(f"audio orphans scan:  {counters['audio_scanned']}")
    print(f"missing info.json:   {counters['missing_info']}")
    print(f"total matches:       {len(matches)}")
    print(f"  already in filter: {len(already)}")
    print(f"  new entries:       {len(to_add)}")

    if not to_add:
        return

    per_key = {f"{ch}/{vid}": (dur, reason) for ch, vid, dur, reason in matches}
    grouped: dict[str, list[tuple[str, int | None, str]]] = {}
    for key in to_add:
        ch, vid = key.split("/", 1)
        dur, reason = per_key[key]
        grouped.setdefault(ch, []).append((vid, dur, reason))

    print()
    print("new entries to add:")
    for ch in sorted(grouped):
        print(f"  {ch}")
        for vid, dur, reason in sorted(grouped[ch], key=lambda t: (t[1] is None, t[1] or 0)):
            dur_s = f"{dur:>5}s" if dur is not None else "   ? s"
            print(f"    {dur_s}  {reason:<18}  {vid}")


def main() -> int:
    """Scan video + audio dirs, report matches, optionally update filter."""
    args = parse_args()

    config = Config(args.config)
    videos_dir = config.getDataDownloadsVideosDir()
    audio_dir = config.getDataDownloadsAudioDir()
    metadata_dir = config.getDataDownloadsMetadataDir()

    matches, counters = collect_filter_matches(
        videos_dir,
        audio_dir,
        metadata_dir,
        args.channel,
        args.max_duration,
    )
    if counters["channels"] == 0:
        print("No channels to scan.", file=sys.stderr)
        return 2

    filter_path = Path(args.filter)
    if not filter_path.is_file():
        print(f"ERROR: filter file not found: {filter_path}", file=sys.stderr)
        return 2
    filter_data = json.load(filter_path.open())
    existing = set(filter_data.get(_FILTER_KEY, []))

    new_keys = {f"{ch}/{vid}" for ch, vid, _, _ in matches}
    to_add = sorted(new_keys - existing)
    already = new_keys & existing

    _print_report(matches, counters, args.max_duration, args.write, already, to_add)

    if not args.write:
        print()
        print("Dry-run only. Re-run with --write to update the filter file.")
        return 0

    if not to_add:
        print()
        print("Nothing to write — no new entries.")
        return 0

    merged = sorted(existing | new_keys)
    filter_data[_FILTER_KEY] = merged
    filter_path.write_text(json.dumps(filter_data, indent=4) + "\n")
    print()
    print(f"Wrote {filter_path}: {len(existing)} → {len(merged)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
