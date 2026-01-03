#!/usr/bin/env python3
"""Display processing status of downloaded content."""

import sys
from collections import defaultdict
from pathlib import Path


def count_files_by_extension(directory: Path, extensions: set[str]) -> int:
    """Count files with specific extensions in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*") if f.suffix in extensions and f.is_file())


def get_archived_video_names_per_channel(archive_dir: Path) -> dict[str, set[str]]:
    """Get archived video basenames organized by channel.

    Returns dict mapping channel names to sets of archived video basenames
    (without extensions).
    """
    videos_dir = archive_dir / "videos"
    if not videos_dir.exists():
        return {}

    archived_videos = defaultdict(set)
    video_extensions = {".mp4", ".webm", ".m4a", ".mov", ".m4v", ".avi", ".mkv", ".flv"}

    for channel_dir in videos_dir.iterdir():
        if channel_dir.is_dir():
            for f in channel_dir.iterdir():
                if f.is_file() and f.suffix in video_extensions and not f.name.startswith("._"):
                    # Extract basename without extension
                    archived_videos[channel_dir.name].add(f.stem)

    return dict(archived_videos)


def get_channel_stats(base_dir: Path, archived_videos: dict[str, set[str]] | None = None) -> dict[str, dict[str, int]]:
    """Get statistics for each channel, excluding archived content.

    Args:
        base_dir: Base directory containing videos/audio/transcripts
        archived_videos: Dict mapping channel names to sets of archived video basenames.
                        If provided, transcripts for archived videos will be excluded.
    """
    stats = defaultdict(lambda: {"videos": 0, "audio": 0, "transcripts": 0})

    # Count videos (.mp4 files only, not .info.json, and exclude macOS metadata files)
    videos_dir = base_dir / "videos"
    if videos_dir.exists():
        for channel_dir in videos_dir.iterdir():
            if channel_dir.is_dir():
                video_count = sum(1 for f in channel_dir.iterdir() if f.suffix == ".mp4" and not f.name.startswith("._"))
                stats[channel_dir.name]["videos"] = video_count

    # Count audio files (.wav, exclude macOS metadata files)
    audio_dir = base_dir / "audio"
    if audio_dir.exists():
        for channel_dir in audio_dir.iterdir():
            if channel_dir.is_dir():
                audio_count = sum(1 for f in channel_dir.iterdir() if f.suffix == ".wav" and not f.name.startswith("._"))
                stats[channel_dir.name]["audio"] = audio_count

    # Count transcripts (.srt files as indicator, exclude macOS metadata files)
    # EXCLUDE transcripts for archived videos
    transcripts_dir = base_dir / "transcripts"
    if transcripts_dir.exists():
        for channel_dir in transcripts_dir.iterdir():
            if channel_dir.is_dir():
                channel_name = channel_dir.name

                # Get set of archived video basenames for this channel
                archived_basenames = set()
                if archived_videos and channel_name in archived_videos:
                    archived_basenames = archived_videos[channel_name]

                # Count only transcripts that are NOT for archived videos
                transcript_count = 0
                for f in channel_dir.iterdir():
                    if f.suffix == ".srt" and not f.name.startswith("._"):
                        # Extract basename from transcript filename
                        transcript_basename = f.stem

                        # Only count if NOT archived
                        if transcript_basename not in archived_basenames:
                            transcript_count += 1

                stats[channel_name]["transcripts"] = transcript_count

    return dict(stats)


def get_archived_channel_stats(archive_dir: Path, transcripts_dir: Path, archived_videos: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    """Get statistics for archived videos by channel.

    Args:
        archive_dir: Archive base directory
        transcripts_dir: Directory containing transcript files
        archived_videos: Dict mapping channel names to sets of archived video basenames

    Returns:
        Dict mapping channel names to stats dicts with keys:
        - "videos": count of archived videos
        - "audio": always 0 (archived videos have audio deleted)
        - "transcripts": count of transcripts for archived videos
    """
    stats = defaultdict(lambda: {"videos": 0, "audio": 0, "transcripts": 0})
    video_extensions = {".mp4", ".webm", ".m4a", ".mov", ".m4v", ".avi", ".mkv", ".flv"}

    # Count archived videos
    archive_videos_dir = archive_dir / "videos"
    if archive_videos_dir.exists():
        for channel_dir in archive_videos_dir.iterdir():
            if channel_dir.is_dir():
                channel_name = channel_dir.name
                video_count = sum(
                    1 for f in channel_dir.iterdir() if f.is_file() and f.suffix in video_extensions and not f.name.startswith("._")
                )
                stats[channel_name]["videos"] = video_count
                stats[channel_name]["audio"] = 0  # Always 0 - audio files are deleted

    # Count transcripts for archived videos
    if transcripts_dir.exists():
        for channel_dir in transcripts_dir.iterdir():
            if channel_dir.is_dir():
                channel_name = channel_dir.name

                # Get set of archived video basenames for this channel
                if channel_name not in archived_videos:
                    continue  # No archived videos for this channel

                archived_basenames = archived_videos[channel_name]

                # Count only transcripts that ARE for archived videos
                transcript_count = 0
                for f in channel_dir.iterdir():
                    if f.suffix == ".srt" and not f.name.startswith("._"):
                        transcript_basename = f.stem

                        # Only count if it IS archived
                        if transcript_basename in archived_basenames:
                            transcript_count += 1

                stats[channel_name]["transcripts"] = transcript_count

    return dict(stats)


def main() -> int:
    """Main entry point."""
    # Get the downloads directory
    project_root = Path(__file__).parent.parent
    downloads_dir = project_root / "data" / "downloads"
    archive_dir = project_root / "data" / "archive"
    transcripts_dir = downloads_dir / "transcripts"

    if not downloads_dir.exists():
        print(f"Error: Downloads directory not found: {downloads_dir}")
        return 1

    # Step 1: Get mapping of archived videos
    archived_videos = get_archived_video_names_per_channel(archive_dir)

    # Step 2: Get active channel statistics (excludes archived transcripts)
    active_channel_stats = get_channel_stats(downloads_dir, archived_videos)

    # Step 3: Get archived channel statistics
    archived_channel_stats = get_archived_channel_stats(archive_dir, transcripts_dir, archived_videos)

    # Calculate totals for active content
    total_active_channels = len(active_channel_stats)
    total_active_videos = sum(stats["videos"] for stats in active_channel_stats.values())
    total_active_audio = sum(stats["audio"] for stats in active_channel_stats.values())
    total_active_transcripts = sum(stats["transcripts"] for stats in active_channel_stats.values())

    # Calculate totals for archived content
    total_archived_channels = len(archived_channel_stats)
    total_archived_videos = sum(stats["videos"] for stats in archived_channel_stats.values())
    total_archived_transcripts = sum(stats["transcripts"] for stats in archived_channel_stats.values())

    # Print overall summary
    print("=" * 70)
    print("PROCESSING STATUS")
    print("=" * 70)
    print()
    print("ACTIVE CONTENT (In Progress):")
    print(f"  Channels:       {total_active_channels}")
    print(f"  Videos:         {total_active_videos}")
    print(f"  Audio files:    {total_active_audio}")
    print(f"  Transcripts:    {total_active_transcripts}")
    print()
    print("ARCHIVED CONTENT (Completed):")
    print(f"  Channels:       {total_archived_channels}")
    print(f"  Videos:         {total_archived_videos}")
    print(f"  Transcripts:    {total_archived_transcripts}")
    print()

    # Calculate processing progress for active content
    if total_active_videos > 0:
        audio_pct = (total_active_audio / total_active_videos) * 100
        transcript_pct = (total_active_transcripts / total_active_videos) * 100
        print(f"Active Audio extraction:  {audio_pct:.1f}% complete")
        print(f"Active Transcription:     {transcript_pct:.1f}% complete")
    print()

    # TABLE 1: Active Downloads
    if active_channel_stats:
        print("=" * 90)
        print("ACTIVE DOWNLOADS (Not Yet Archived)")
        print("=" * 90)
        print()
        print(f"{'Channel':<50} {'Videos':>8} {'Audio':>8} {'Trans':>8} {'%':>7}")
        print("-" * 90)

        for channel_name in sorted(active_channel_stats.keys()):
            stats = active_channel_stats[channel_name]
            # Calculate completion percentage: (audio + transcripts) / (videos * 2) * 100
            completion_pct = ((stats["audio"] + stats["transcripts"]) / (stats["videos"] * 2)) * 100 if stats["videos"] > 0 else 0.0
            print(f"{channel_name:<50} {stats['videos']:>8} {stats['audio']:>8} {stats['transcripts']:>8} {completion_pct:>6.1f}%")

        # Add total row
        print("-" * 90)
        total_active_completion_pct = (
            ((total_active_audio + total_active_transcripts) / (total_active_videos * 2)) * 100 if total_active_videos > 0 else 0.0
        )
        print(
            f"{'TOTAL':<50} {total_active_videos:>8} {total_active_audio:>8} "
            f"{total_active_transcripts:>8} {total_active_completion_pct:>6.1f}%"
        )
        print()

    # TABLE 2: Archived Videos
    if archived_channel_stats:
        print("=" * 90)
        print("ARCHIVED VIDEOS (Processing Complete)")
        print("=" * 90)
        print()
        print(f"{'Channel':<50} {'Videos':>8} {'Audio':>8} {'Trans':>8} {'%':>7}")
        print("-" * 90)

        for channel_name in sorted(archived_channel_stats.keys()):
            stats = archived_channel_stats[channel_name]
            # Calculate completion percentage for archived videos
            # For archived: transcripts / videos * 100 (audio is always deleted)
            completion_pct = (stats["transcripts"] / stats["videos"]) * 100 if stats["videos"] > 0 else 0.0
            print(f"{channel_name:<50} {stats['videos']:>8} {'-':>8} {stats['transcripts']:>8} {completion_pct:>6.1f}%")

        # Add total row
        print("-" * 90)
        total_archived_completion_pct = (total_archived_transcripts / total_archived_videos) * 100 if total_archived_videos > 0 else 0.0
        print(f"{'TOTAL':<50} {total_archived_videos:>8} {'-':>8} {total_archived_transcripts:>8} {total_archived_completion_pct:>6.1f}%")
        print()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
