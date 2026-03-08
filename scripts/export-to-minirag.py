#!/usr/bin/env python3
"""Export leaf nodes from deterministic topic trees as .txt + .json pairs for mini-rag.

This reads `*_topic_tree.json` files produced by `scripts/topic-tree.py` and exports
only leaf nodes (time-contiguous segments) into a directory structure suitable for
mini-rag ingestion.

Usage:
    uv run python scripts/export-to-minirag.py --export-dir /path/to/mini-rag/input
    uv run python scripts/export-to-minirag.py --export-dir /path/to/mini-rag/input --force
    uv run python scripts/export-to-minirag.py --export-dir /path/to/mini-rag/input --file data/output/topics/..._topic_tree.json
"""

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config import Config
from src.topic_detection.hierarchy.schemas import MethodKeyphrasesData, TaxonomyLabelData, TopicTreeNodeData, TopicTreeOutput
from src.util.srt_util import SRTEntry, SRTUtil


class LeafSourceData(BaseModel):
    """YouTube source_data fields for mini-rag ingestion."""

    channel: str
    timestamp: str
    video_title: str
    upload_date: str | None
    exported_at: str
    source_file: str
    topic_tree_file: str
    embedding_model: str
    node_id: str
    depth: int = Field(..., ge=0)
    leaf_index: int = Field(..., ge=1)
    start_timestamp: str
    end_timestamp: str
    start_entry_idx: int = Field(..., ge=0)
    end_entry_idx_exclusive: int = Field(..., ge=0)
    taxonomy_labels: list[TaxonomyLabelData]
    keyphrases: list[MethodKeyphrasesData]

    model_config = ConfigDict(frozen=True, extra="forbid")


class LeafCommonData(BaseModel):
    """Common citation fields recognized by mini-rag."""

    title: str
    url: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class LeafCitation(BaseModel):
    """Nested citation format for mini-rag ingestion.

    Uses the nested format (citation_key, source_type, common, source_data)
    so the mini-rag ingester accepts it without field validation.
    """

    citation_key: str
    source_type: str
    common: LeafCommonData
    source_data: LeafSourceData

    model_config = ConfigDict(frozen=True, extra="forbid")


def sanitize_dirname(name: str) -> str:
    """Sanitize a string for use as a directory name."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip(". _")
    return sanitized if sanitized else "untitled"


def format_duration(seconds: float) -> str:
    """Format seconds as [Xh:Ym] duration with unit indicators."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"[{hours:02d}h:{minutes:02d}m]"


def calculate_eta(file_idx: int, total: int, start_time: float) -> str:
    """Calculate ETA string based on average time per processed file."""
    if file_idx <= 1:
        return ""
    elapsed = time.time() - start_time
    avg_time_per_file = elapsed / (file_idx - 1)
    remaining_files = total - file_idx + 1
    eta_seconds = avg_time_per_file * remaining_files
    return f" ETA {format_duration(eta_seconds)}"


def youtube_url_with_timestamp(*, webpage_url: str, start_seconds: int) -> str:
    """Append a `t=` timestamp parameter to a YouTube URL."""
    sep = "&" if "?" in webpage_url else "?"
    return f"{webpage_url}{sep}t={start_seconds}"


def find_info_json(*, metadata_dir: Path, channel_dir_name: str, video_stem: str, metadata_video_subdir: str) -> Path | None:
    """Find the `.info.json` metadata file for a video."""
    info_path = metadata_dir / channel_dir_name / metadata_video_subdir / f"{video_stem}.info.json"
    return info_path if info_path.exists() else None


def discover_topic_trees(args: argparse.Namespace, *, topics_dir: Path) -> tuple[list[Path], Path] | tuple[None, str]:
    """Discover `*_topic_tree.json` files.

    Returns (tree_files, base_dir) on success, or (None, error_message) on failure.
    """
    if not topics_dir.exists():
        return None, f"Error: Topics output directory not found: {topics_dir}"

    if args.file:
        file_path = args.file
        if not file_path.exists():
            return None, f"Error: File not found: {file_path}"
        if file_path.suffix.lower() != ".json" or not file_path.name.endswith("_topic_tree.json"):
            return None, f"Error: Expected *_topic_tree.json file, got: {file_path}"

        try:
            file_path.relative_to(topics_dir)
        except ValueError:
            return None, f"Error: --file must be inside topics output dir: {topics_dir}"

        return [file_path], topics_dir

    tree_files = sorted(topics_dir.rglob("*_topic_tree.json"))
    tree_files = [f for f in tree_files if not f.name.startswith("._")]
    return tree_files, topics_dir


def node_text(*, entries: list[SRTEntry], node: TopicTreeNodeData) -> str:
    """Join SRT entry contents for a node span into plain text."""
    parts = [e.content for e in entries[node.start_entry_idx : node.end_entry_idx_exclusive]]
    return " ".join(parts)


def extract_required_info_fields(info: dict[str, Any]) -> tuple[str, str, str, str | None]:
    """Extract required fields from yt-dlp `.info.json` metadata."""
    title_raw = info.get("title")
    channel_raw = info.get("channel")
    url_raw = info.get("webpage_url")

    title = title_raw if isinstance(title_raw, str) else None
    channel = channel_raw if isinstance(channel_raw, str) else None
    webpage_url = url_raw if isinstance(url_raw, str) else None

    if title is None or title.strip() == "":
        raise ValueError("Missing required metadata field: title")
    if channel is None or channel.strip() == "":
        raise ValueError("Missing required metadata field: channel")
    if webpage_url is None or webpage_url.strip() == "":
        raise ValueError("Missing required metadata field: webpage_url")

    upload_date_raw = info.get("upload_date")
    upload_date = upload_date_raw if isinstance(upload_date_raw, str) and upload_date_raw.strip() != "" else None

    return title, channel, webpage_url, upload_date


def export_video(
    *,
    topic_tree_path: Path,
    topics_base_dir: Path,
    data_dir: Path,
    metadata_dir: Path,
    metadata_video_subdir: str,
    export_dir: Path,
    channel_dir_name: str,
    video_stem: str,
) -> tuple[int, int]:
    """Export leaf nodes for a single video topic tree.

    Returns:
        (exported_leaf_count, total_leaf_count)
    """
    with open(topic_tree_path, encoding="utf-8") as f:
        tree = TopicTreeOutput.model_validate(json.load(f))

    source_srt_path = data_dir / tree.source_file
    if not source_srt_path.exists():
        raise FileNotFoundError(f"Referenced SRT not found: {source_srt_path}")

    entries = SRTUtil.parse_srt_file(source_srt_path)
    if not entries:
        raise ValueError(f"No SRT entries found in: {source_srt_path}")

    info_path = find_info_json(
        metadata_dir=metadata_dir,
        channel_dir_name=channel_dir_name,
        video_stem=video_stem,
        metadata_video_subdir=metadata_video_subdir,
    )
    if info_path is None:
        raise FileNotFoundError(f"Missing .info.json metadata for: {channel_dir_name}/{video_stem}")

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    video_title, channel, webpage_url, upload_date = extract_required_info_fields(info)

    sanitized_title = sanitize_dirname(video_title)
    video_export_dir = export_dir / channel_dir_name / sanitized_title
    video_export_dir.mkdir(parents=True, exist_ok=True)

    leaves = [n for n in tree.nodes if not n.children_ids]
    leaves_sorted = sorted(leaves, key=lambda n: (n.start_entry_idx, n.end_entry_idx_exclusive, n.node_id))

    now_iso = datetime.now(tz=UTC).isoformat()
    exported = 0
    total_leaves = len(leaves_sorted)

    relative_tree_path = str(topic_tree_path.relative_to(topics_base_dir))

    for idx, node in enumerate(leaves_sorted, start=1):
        if node.end_entry_idx_exclusive > len(entries):
            raise ValueError(
                f"Leaf span exceeds SRT entry count (leaf {node.node_id}, end={node.end_entry_idx_exclusive}, entries={len(entries)})"
            )

        start_seconds = int(entries[node.start_entry_idx].start.total_seconds())
        url_with_time = youtube_url_with_timestamp(webpage_url=webpage_url, start_seconds=start_seconds)

        leaf_filename = f"leaf_{idx:03d}"
        txt_path = video_export_dir / f"{leaf_filename}.txt"
        txt_path.write_text(node_text(entries=entries, node=node), encoding="utf-8")

        citation_key = f"{sanitized_title}_{leaf_filename}"
        citation = LeafCitation(
            citation_key=citation_key,
            source_type="youtube",
            common=LeafCommonData(
                title=video_title,
                url=url_with_time,
            ),
            source_data=LeafSourceData(
                channel=channel,
                timestamp=node.start_timestamp,
                video_title=video_title,
                upload_date=upload_date,
                exported_at=now_iso,
                source_file=tree.source_file,
                topic_tree_file=relative_tree_path,
                embedding_model=tree.embedding_model,
                node_id=node.node_id,
                depth=node.depth,
                leaf_index=idx,
                start_timestamp=node.start_timestamp,
                end_timestamp=node.end_timestamp,
                start_entry_idx=node.start_entry_idx,
                end_entry_idx_exclusive=node.end_entry_idx_exclusive,
                taxonomy_labels=node.taxonomy_labels,
                keyphrases=node.keyphrases,
            ),
        )

        json_path = video_export_dir / f"{leaf_filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(citation.model_dump(), f, indent=2, ensure_ascii=False)

        exported += 1

    return exported, total_leaves


def prescan_topic_trees(
    *,
    topic_tree_files: list[Path],
    topics_base_dir: Path,
    metadata_dir: Path,
    metadata_video_subdir: str,
    export_dir: Path,
    force: bool,
) -> tuple[list[tuple[Path, str, str]], int, int]:
    """Pre-scan topic trees to separate into files to process vs skip.

    Returns:
        (files_to_process, skipped_count, missing_metadata_count)
    """
    files_to_process: list[tuple[Path, str, str]] = []
    skipped = 0
    missing_metadata = 0

    for tree_path in topic_tree_files:
        relative_path = tree_path.relative_to(topics_base_dir)
        channel_dir_name = str(relative_path.parent)
        video_stem = tree_path.name.replace("_topic_tree.json", "")

        info_path = find_info_json(
            metadata_dir=metadata_dir,
            channel_dir_name=channel_dir_name,
            video_stem=video_stem,
            metadata_video_subdir=metadata_video_subdir,
        )
        if info_path is None:
            print(f"Skipping: {relative_path} (no .info.json found)")
            missing_metadata += 1
            continue

        if not force:
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            title, _, _, _ = extract_required_info_fields(info)
            sanitized_title = sanitize_dirname(title)
            check_dir = export_dir / channel_dir_name / sanitized_title
            if check_dir.exists() and any(check_dir.glob("leaf_*.txt")):
                print(f"Skipping: {relative_path} (already exported)")
                skipped += 1
                continue

        files_to_process.append((tree_path, channel_dir_name, video_stem))

    return files_to_process, skipped, missing_metadata


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export leaf nodes from topic trees to mini-rag format (.txt + .json pairs)")
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Directory to write exported .txt/.json files into",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Process a single *_topic_tree.json file instead of all files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-export even if output directory already has leaf_*.txt files",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config(config_path)

    td_config = config.get_topic_detection_config()
    topics_dir = config.getDataDir() / td_config.output_dir
    metadata_dir = config.getDataDownloadsMetadataDir()
    metadata_video_subdir = config.getTranscriptionMetadataVideoSubdir()
    export_dir = args.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    result = discover_topic_trees(args, topics_dir=topics_dir)
    if result[0] is None:
        print(result[1], file=sys.stderr)
        return 1
    topic_tree_files, topics_base_dir = result

    if not topic_tree_files:
        print("No *_topic_tree.json files found.")
        return 0

    print(f"Found {len(topic_tree_files)} topic tree file(s)")
    print(f"Topics directory: {topics_dir}")
    print(f"Export directory: {export_dir}")
    print()

    files_to_process, skipped_count, missing_metadata_count = prescan_topic_trees(
        topic_tree_files=topic_tree_files,
        topics_base_dir=topics_base_dir,
        metadata_dir=metadata_dir,
        metadata_video_subdir=metadata_video_subdir,
        export_dir=export_dir,
        force=args.force,
    )

    if not files_to_process:
        print()
        print("=" * 50)
        print(f"Completed: 0 exported, {skipped_count} skipped, {missing_metadata_count} missing metadata")
        return 0

    print()
    print(f"Processing {len(files_to_process)} video(s), {skipped_count} already exported, {missing_metadata_count} missing metadata")
    print()

    success_count = 0
    failure_count = 0
    total_leaves_exported = 0
    total_leaves_seen = 0
    total_to_process = len(files_to_process)
    start_time = time.time()

    for file_idx, (tree_path, channel_dir_name, video_stem) in enumerate(files_to_process, start=1):
        relative_path = tree_path.relative_to(topics_base_dir)
        progress_pct = (file_idx / total_to_process) * 100
        eta_str = calculate_eta(file_idx, total_to_process, start_time)
        print(f"Processing [{file_idx}/{total_to_process}] ({progress_pct:.0f}%){eta_str}: {relative_path}")

        try:
            exported, total_leaves = export_video(
                topic_tree_path=tree_path,
                topics_base_dir=topics_base_dir,
                data_dir=config.getDataDir(),
                metadata_dir=metadata_dir,
                metadata_video_subdir=metadata_video_subdir,
                export_dir=export_dir,
                channel_dir_name=channel_dir_name,
                video_stem=video_stem,
            )
            total_leaves_exported += exported
            total_leaves_seen += total_leaves
            print(f"  Exported {exported} leaf nodes (total leaves: {total_leaves})")
            success_count += 1
        except Exception as e:
            print(f"  Error: {e}")
            failure_count += 1

    total_elapsed = time.time() - start_time
    print()
    print("=" * 50)
    print(
        f"Completed: {success_count} videos exported, {skipped_count} skipped, "
        f"{failure_count} failed (elapsed: {format_duration(total_elapsed)})"
    )
    print(f"Leaves: {total_leaves_exported} exported, {total_leaves_seen - total_leaves_exported} not exported")

    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
