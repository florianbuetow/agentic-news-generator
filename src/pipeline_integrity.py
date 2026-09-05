"""Read-only integrity checks for video pipeline bookkeeping and artifacts."""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from src.config import Config


@dataclass(frozen=True)
class DownloadArchiveEntry:
    """One channel-scoped video recorded in a yt-dlp download archive."""

    channel: str
    video_id: str
    archive_file: Path


def load_filtered_video_ids(filter_path: Path) -> set[str]:
    """Return video IDs listed across every key of ``filefilter.json``."""
    with filter_path.open(encoding="utf-8") as handle:
        filter_data = json.load(handle)

    video_ids: set[str] = set()
    for entries in filter_data.values():
        for entry in entries:
            if "/" not in entry:
                print(f"WARN: malformed filter entry '{entry}', skipping", file=sys.stderr)
                continue
            video_id = entry.split("/", 1)[1]
            if video_id:
                video_ids.add(video_id)
    return video_ids


def extract_video_id(filename: str) -> str | None:
    """Return the bracketed video ID immediately before the extension."""
    match = re.search(r"\[([A-Za-z0-9_-]+)\](?=\.[^/]+$)", filename)
    return match.group(1) if match is not None else None


def artifact_extensions() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Return supported video, audio, and transcript artifact extensions."""
    video_extensions = frozenset({".avi", ".flv", ".m4a", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
    audio_extensions = frozenset({".wav"})
    transcript_extensions = frozenset({".json", ".srt", ".tsv", ".txt", ".vtt"})
    return video_extensions, audio_extensions, transcript_extensions


def scan_data_dirs(config: Config) -> list[Path]:
    """Return every ID-convention data directory checked for filtered files."""
    return [
        config.get_data_downloads_videos_dir(),
        config.get_data_downloads_audio_dir(),
        config.get_data_downloads_transcripts_dir(),
        config.get_data_downloads_transcripts_hallucinations_dir(),
        config.get_data_downloads_transcripts_cleaned_dir(),
        config.get_data_downloads_transcripts_summaries_dir(),
        config.get_data_downloads_metadata_dir(),
        config.get_data_archive_videos_dir(),
    ]


def iter_channel_files(base_dir: Path) -> Iterable[Path]:
    """Yield files within channel subdirectories of ``base_dir``."""
    for channel_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        for file_path in sorted(channel_dir.rglob("*")):
            if file_path.is_file():
                yield file_path


def find_offenders(base_dir: Path, filtered_ids: set[str]) -> list[Path]:
    """Return files under ``base_dir`` whose video ID is filtered."""
    return [
        file_path
        for file_path in iter_channel_files(base_dir)
        if (video_id := extract_video_id(file_path.name)) is not None and video_id in filtered_ids
    ]


def find_filtered_file_offenders(
    config: Config,
    filtered_ids: set[str],
    announce_scan: Callable[[Path], None] | None,
) -> list[Path]:
    """Return all existing artifacts whose video ID is filtered."""
    offenders: list[Path] = []
    for base_dir in scan_data_dirs(config):
        if announce_scan is not None:
            announce_scan(base_dir)
        if base_dir.is_dir():
            offenders.extend(find_offenders(base_dir, filtered_ids))
    return offenders


def video_ids_from_artifacts(artifacts: list[Path]) -> set[str]:
    """Return unique video IDs represented by artifact paths."""
    return {video_id for artifact in artifacts if (video_id := extract_video_id(artifact.name)) is not None}


def relative_to_data_dir(path: Path, data_dir: Path) -> Path:
    """Return ``path`` relative to ``data_dir``, or the original path."""
    try:
        return path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        return path


def load_download_archive_entries(videos_dir: Path) -> list[DownloadArchiveEntry]:
    """Load unique, valid YouTube video IDs from channel ``downloaded.txt`` files."""
    entries: set[DownloadArchiveEntry] = set()
    for channel_dir in sorted(path for path in videos_dir.iterdir() if path.is_dir()):
        archive_file = channel_dir / "downloaded.txt"
        if not archive_file.is_file():
            continue
        try:
            lines = archive_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            print(f"WARN: cannot read download archive {archive_file}: {exc}", file=sys.stderr)
            continue
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 2 or re.fullmatch(r"[A-Za-z0-9_-]{11}", fields[1]) is None:
                if line.strip():
                    print(f"WARN: malformed download archive entry {archive_file}:{line_number}: {line}", file=sys.stderr)
                continue
            entries.add(DownloadArchiveEntry(channel_dir.name, fields[1], archive_file))
    return sorted(entries, key=lambda entry: (entry.channel, entry.video_id, str(entry.archive_file)))


def collect_artifact_keys(base_dir: Path, extensions: frozenset[str]) -> set[tuple[str, str]]:
    """Return ``(channel, video_id)`` keys for matching artifact files."""
    keys: set[tuple[str, str]] = set()
    if not base_dir.is_dir():
        return keys
    for channel_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        for file_path in channel_dir.rglob("*"):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() not in extensions:
                continue
            video_id = extract_video_id(file_path.name)
            if video_id is not None:
                keys.add((channel_dir.name, video_id))
    return keys


def find_stranded_download_entries(
    config: Config,
    excluded_video_ids: set[str],
) -> list[DownloadArchiveEntry]:
    """Return archive entries with no active/archived video, audio, or transcript."""
    videos_dir = config.get_data_downloads_videos_dir()
    entries = load_download_archive_entries(videos_dir)
    video_extensions, audio_extensions, transcript_extensions = artifact_extensions()

    covered: set[tuple[str, str]] = set()
    covered.update(collect_artifact_keys(videos_dir, video_extensions))
    covered.update(collect_artifact_keys(config.get_data_archive_videos_dir(), video_extensions))
    covered.update(collect_artifact_keys(config.get_data_downloads_audio_dir(), audio_extensions))
    covered.update(collect_artifact_keys(config.get_data_downloads_transcripts_dir(), transcript_extensions))

    return [entry for entry in entries if entry.video_id not in excluded_video_ids and (entry.channel, entry.video_id) not in covered]


def _unlink_with_normalization_fallback(path: Path) -> None:
    """Delete ``path``, retrying its NFC filename on normalization-sensitive filesystems."""
    try:
        path.unlink()
    except FileNotFoundError:
        normalized_path = path.with_name(unicodedata.normalize("NFC", path.name))
        if normalized_path == path:
            raise
        normalized_path.unlink()


def remove_filtered_artifacts(
    config: Config,
    filtered_ids: set[str],
    artifacts: list[Path],
) -> tuple[int, list[tuple[Path, str]]]:
    """Delete validated filtered artifacts and return ``(deleted, failures)``."""
    allowed_roots = [root.resolve() for root in scan_data_dirs(config)]
    deleted = 0
    failures: list[tuple[Path, str]] = []

    for artifact in artifacts:
        try:
            if artifact.is_symlink():
                raise ValueError("refusing to delete a symbolic link")
            if not artifact.is_file():
                raise ValueError("path is not a regular file")

            resolved_artifact = artifact.resolve(strict=True)
            if not any(resolved_artifact.is_relative_to(root) for root in allowed_roots):
                raise ValueError("path is outside configured pipeline data directories")

            video_id = extract_video_id(artifact.name)
            if video_id is None or video_id not in filtered_ids:
                raise ValueError("filename does not contain a currently filtered video ID")

            _unlink_with_normalization_fallback(artifact)
            deleted += 1
        except (OSError, ValueError) as exc:
            failures.append((artifact, str(exc)))

    return deleted, failures


def write_report(report_path: Path, lines: list[str]) -> None:
    """Overwrite a stage report with one finding per line."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    report_path.write_text(content, encoding="utf-8")


def run_pipeline_integrity(config: Config, filter_path: Path) -> int:
    """Run both integrity stages, aborting after the first failing stage."""
    reports_dir = config.get_reports_dir().resolve()

    print("Stage 1/2: checking for files belonging to filtered video IDs")
    filtered_ids = load_filtered_video_ids(filter_path)
    filtered_offenders = find_filtered_file_offenders(config, filtered_ids, None)
    offending_video_ids = video_ids_from_artifacts(filtered_offenders)
    filtered_files_report = reports_dir / "files-whose-video-id-is-listed-in-filefilter.txt"
    write_report(filtered_files_report, [str(offender.resolve()) for offender in filtered_offenders])
    if filtered_offenders:
        print(f"Stage 1 failed: {len(offending_video_ids)} videos ({len(filtered_offenders)} artifact files).")
        print(f"Report: {filtered_files_report}")
        return 1
    print(f"Stage 1 passed: 0 videos (0 artifact files). Report: {filtered_files_report}")
    print()

    print("Stage 2/2: checking downloaded.txt entries for pipeline artifacts")
    stranded_entries = find_stranded_download_entries(config, filtered_ids)
    stranded_entries_report = reports_dir / "downloaded-video-ids-without-video-audio-or-transcript.txt"
    write_report(
        stranded_entries_report,
        [f"{entry.archive_file.resolve()}\t{entry.video_id}" for entry in stranded_entries],
    )
    if stranded_entries:
        entry_str = "entry has" if len(stranded_entries) == 1 else "entries have"
        print(f"Stage 2 failed: {len(stranded_entries)} downloaded.txt {entry_str} no video, audio, or transcript file.")
        print(f"Report: {stranded_entries_report}")
        return 1
    print(f"Stage 2 passed: every downloaded.txt entry has a video, audio, or transcript file. Report: {stranded_entries_report}")
    return 0
