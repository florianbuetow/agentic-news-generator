"""Shared transcript summarization helpers."""

from __future__ import annotations

import re
from pathlib import Path

from src.util.fs_util import FSUtil


def strip_think_tags(text: str) -> str:
    """Strip <think> reasoning tags from model responses."""
    think_match = re.search(r"</think>\s*(.*)$", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return text


def collect_pending_files(
    cleaned_dir: Path,
    summaries_dir: Path,
    channel_filter: str,
) -> tuple[list[tuple[Path, Path]], int, int, int]:
    """Scan for transcript files and partition into pending/done/empty."""
    txt_files = FSUtil.find_files_by_extension(cleaned_dir, ".txt", recursive=True)
    txt_files = [f for f in txt_files if not f.name.startswith("._")]
    if channel_filter:
        txt_files = [f for f in txt_files if f.parent.name == channel_filter]

    pending: list[tuple[Path, Path]] = []
    already_done = 0
    empty_files = 0

    for txt_file in txt_files:
        channel_name = txt_file.parent.name
        output_file = summaries_dir / channel_name / (txt_file.stem + ".md")

        if output_file.exists():
            already_done += 1
            continue

        content = FSUtil.read_text_file(txt_file)
        if not content.strip():
            empty_files += 1
            continue

        pending.append((txt_file, output_file))

    pending_per_channel: dict[str, int] = {}
    for txt_file, _ in pending:
        channel_name = txt_file.parent.name
        if channel_name not in pending_per_channel:
            pending_per_channel[channel_name] = 0
        pending_per_channel[channel_name] += 1

    pending.sort(key=lambda pair: (pending_per_channel[pair[0].parent.name], pair[0].parent.name, pair[0].name))

    return pending, len(txt_files), already_done, empty_files
