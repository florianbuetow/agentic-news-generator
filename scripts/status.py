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


def get_channel_stats(base_dir: Path) -> dict[str, dict[str, int]]:
    """Get statistics for each channel."""
    stats = defaultdict(lambda: {"videos": 0, "audio": 0, "transcripts": 0})

    # Count videos (.mp4 files only, not .info.json)
    videos_dir = base_dir / "videos"
    if videos_dir.exists():
        for channel_dir in videos_dir.iterdir():
            if channel_dir.is_dir():
                video_count = sum(1 for f in channel_dir.iterdir() if f.suffix == ".mp4")
                stats[channel_dir.name]["videos"] = video_count

    # Count audio files (.wav)
    audio_dir = base_dir / "audio"
    if audio_dir.exists():
        for channel_dir in audio_dir.iterdir():
            if channel_dir.is_dir():
                audio_count = sum(1 for f in channel_dir.iterdir() if f.suffix == ".wav")
                stats[channel_dir.name]["audio"] = audio_count

    # Count transcripts (.srt files as indicator)
    transcripts_dir = base_dir / "transcripts"
    if transcripts_dir.exists():
        for channel_dir in transcripts_dir.iterdir():
            if channel_dir.is_dir():
                transcript_count = sum(1 for f in channel_dir.iterdir() if f.suffix == ".srt")
                stats[channel_dir.name]["transcripts"] = transcript_count

    return dict(stats)


def count_archived_videos(archive_dir: Path) -> int:
    """Count archived videos."""
    videos_dir = archive_dir / "videos"
    if not videos_dir.exists():
        return 0
    return sum(1 for f in videos_dir.rglob("*.mp4") if f.is_file())


def main() -> int:
    """Main entry point."""
    # Get the downloads directory
    project_root = Path(__file__).parent.parent
    downloads_dir = project_root / "data" / "downloads"
    archive_dir = project_root / "data" / "archive"

    if not downloads_dir.exists():
        print(f"Error: Downloads directory not found: {downloads_dir}")
        return 1

    # Get channel statistics
    channel_stats = get_channel_stats(downloads_dir)

    # Count archived videos
    archived_videos = count_archived_videos(archive_dir)

    # Calculate totals
    total_channels = len(channel_stats)
    total_videos = sum(stats["videos"] for stats in channel_stats.values())
    total_audio = sum(stats["audio"] for stats in channel_stats.values())
    total_transcripts = sum(stats["transcripts"] for stats in channel_stats.values())

    # Print overall summary
    print("=" * 70)
    print("PROCESSING STATUS")
    print("=" * 70)
    print()
    print(f"Channels:       {total_channels}")
    print(f"Videos:         {total_videos}")
    print(f"Audio files:    {total_audio}")
    print(f"Transcripts:    {total_transcripts}")
    print(f"Archived:       {archived_videos}")
    print()

    # Calculate processing progress
    if total_videos > 0:
        audio_pct = (total_audio / total_videos) * 100
        transcript_pct = (total_transcripts / total_videos) * 100
        print(f"Audio extraction:  {audio_pct:.1f}% complete")
        print(f"Transcription:     {transcript_pct:.1f}% complete")
    print()

    # Print per-channel breakdown if there are channels
    if channel_stats:
        print("=" * 80)
        print("PER-CHANNEL BREAKDOWN")
        print("=" * 80)
        print()
        print(f"{'Channel':<50} {'Videos':>8} {'Audio':>8} {'Trans':>8}")
        print("-" * 80)

        for channel_name in sorted(channel_stats.keys()):
            stats = channel_stats[channel_name]
            print(f"{channel_name:<50} {stats['videos']:>8} {stats['audio']:>8} {stats['transcripts']:>8}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
