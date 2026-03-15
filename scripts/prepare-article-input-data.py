#!/usr/bin/env python3
"""Prepare article input data: create bundles from video source files.

Given one or more video IDs, finds the cleaned transcript, metadata, and topics
files, then copies them into a bundle directory with a manifest.json.

The source files (transcript, topics, metadata) are copied as-is — they are
immutable inputs from earlier pipeline stages. Only the manifest is created
by this script.
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, cast

from src.config import Config

logger = logging.getLogger(__name__)


def find_transcript(transcripts_cleaned_dir: Path, video_id: str) -> Path:
    """Find the cleaned transcript file matching a video ID.

    Searches for .txt files containing [<video_id>] in the filename.

    Raises:
        FileNotFoundError: If no matching transcript is found.
    """
    pattern = f"*[[]{video_id}[]]*.txt"
    matches = list(transcripts_cleaned_dir.rglob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No cleaned transcript found for video_id={video_id} in {transcripts_cleaned_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple transcripts found for video_id={video_id}: {matches}")
    return matches[0]


def find_metadata(metadata_dir: Path, channel_name: str, video_id: str) -> Path:
    """Find the video metadata .info.json file.

    Raises:
        FileNotFoundError: If no matching metadata is found.
    """
    video_metadata_dir = metadata_dir / channel_name / "video"
    if not video_metadata_dir.exists():
        raise FileNotFoundError(f"Video metadata directory not found: {video_metadata_dir}")
    pattern = f"*[[]{video_id}[]]*.info.json"
    matches = list(video_metadata_dir.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No metadata found for video_id={video_id} in {video_metadata_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple metadata files found for video_id={video_id}: {matches}")
    return matches[0]


def find_topics(topics_dir: Path, channel_name: str, video_id: str) -> Path:
    """Find the topics JSON file matching a video ID by filename pattern.

    Searches for files matching *[<video_id>]*_topics.json in the channel's
    topics directory. This matches the output pattern from extract-topics.py.

    Raises:
        FileNotFoundError: If no matching topics file is found.
    """
    channel_topics_dir = topics_dir / channel_name
    if not channel_topics_dir.exists():
        raise FileNotFoundError(f"Channel topics directory not found: {channel_topics_dir}")
    pattern = f"*[[]{video_id}[]]*_topics.json"
    matches = list(channel_topics_dir.glob(pattern))
    # Exclude macOS resource fork files
    matches = [m for m in matches if not m.name.startswith("._")]
    if len(matches) == 0:
        raise FileNotFoundError(f"No topics file found for video_id={video_id} in {channel_topics_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple topics files found for video_id={video_id}: {matches}")
    return matches[0]


def extract_metadata_fields(metadata_path: Path) -> dict[str, str]:
    """Extract required fields from a video metadata .info.json file."""
    with open(metadata_path, encoding="utf-8") as handle:
        raw: Any = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Metadata file root must be a JSON object: {metadata_path}")

    data: dict[str, object] = cast(dict[str, object], raw)

    title_raw = data.get("title", "")
    title = str(title_raw) if title_raw else ""
    if not title:
        raise ValueError(f"Missing or empty 'title' in metadata: {metadata_path}")

    upload_date = str(data.get("upload_date", ""))
    publish_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}" if len(upload_date) == 8 else upload_date

    uploader = str(data.get("uploader", ""))
    channel = str(data.get("channel", ""))
    uploader_id = str(data.get("uploader_id", ""))

    return {
        "title": title,
        "publish_date": publish_date,
        "author": uploader,
        "channel": channel if channel else uploader_id,
    }


def create_manifest(
    *,
    video_id: str,
    metadata_fields: dict[str, str],
    video_url: str,
) -> dict[str, object]:
    """Create manifest.json content."""
    return {
        "article_title": metadata_fields["title"],
        "slug": video_id,
        "publish_date": metadata_fields["publish_date"],
        "source_text_file": "transcript.txt",
        "topics_file": "topics.json",
        "references": [
            {
                "type": "video",
                "title": metadata_fields["title"],
                "url": video_url,
                "author": metadata_fields["author"],
                "channel": metadata_fields["channel"],
                "date": metadata_fields["publish_date"],
            }
        ],
    }


def prepare_bundle(*, config: Config, video_id: str) -> Path:
    """Create an article input bundle for a video ID.

    Finds the cleaned transcript, metadata, and topics files for the given
    video ID, copies them as-is into a bundle directory, and creates a
    manifest.json.

    Returns:
        Path to the created bundle directory.
    """
    transcripts_cleaned_dir = config.getDataDownloadsTranscriptsCleanedDir()
    metadata_dir = config.getDataDownloadsMetadataDir()
    topics_dir = config.getTopicDetectionOutputDir()
    articles_input_dir = config.getDataArticlesInputDir()

    logger.info("Finding source files for video_id=%s", video_id)

    transcript_path = find_transcript(transcripts_cleaned_dir, video_id)
    channel_name = transcript_path.parent.name
    logger.info("Found transcript: %s (channel=%s)", transcript_path, channel_name)

    metadata_path = find_metadata(metadata_dir, channel_name, video_id)
    logger.info("Found metadata: %s", metadata_path)

    topics_path = find_topics(topics_dir, channel_name, video_id)
    logger.info("Found topics: %s", topics_path)

    bundle_dir = articles_input_dir / video_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(transcript_path, bundle_dir / "transcript.txt")
    shutil.copy2(metadata_path, bundle_dir / "metadata.info.json")
    shutil.copy2(topics_path, bundle_dir / "topics.json")

    metadata_fields = extract_metadata_fields(metadata_path)
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    manifest = create_manifest(
        video_id=video_id,
        metadata_fields=metadata_fields,
        video_url=video_url,
    )

    manifest_path = bundle_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    logger.info("Bundle created: %s", bundle_dir)
    return bundle_dir


def main() -> int:
    """Prepare article input bundles for given video IDs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if len(sys.argv) < 2:
        logger.error("Usage: prepare-article-input-data.py <video_id> [<video_id> ...]")
        return 1

    video_ids = sys.argv[1:]

    config_path = Path("config/config.yaml")
    logger.info("Loading configuration from %s", config_path)
    try:
        config = Config(config_path)
    except Exception as exc:
        logger.error("Failed to load configuration: %s", exc)
        return 1

    success_count = 0
    failure_count = 0

    for video_id in video_ids:
        try:
            logger.info("=== Preparing article input data: %s ===", video_id)
            prepare_bundle(config=config, video_id=video_id)
            success_count += 1
        except Exception as exc:
            logger.exception("Failed to prepare bundle for video %s: %s", video_id, exc)
            failure_count += 1

    logger.info(
        "=== Preparation Complete === success=%d failures=%d total=%d",
        success_count,
        failure_count,
        len(video_ids),
    )

    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
