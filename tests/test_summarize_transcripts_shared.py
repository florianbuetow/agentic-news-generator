"""Tests for shared transcript summarization helpers."""

from __future__ import annotations

from pathlib import Path

from src.summarize.transcripts import collect_pending_files, strip_think_tags


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_strip_think_tags_returns_content_after_reasoning_block() -> None:
    """Responses with reasoning tags keep only the final answer."""
    assert strip_think_tags("<think>hidden</think>\n\nVisible summary") == "Visible summary"


def test_strip_think_tags_leaves_plain_text_unchanged() -> None:
    """Responses without reasoning tags pass through unchanged."""
    assert strip_think_tags("Visible summary") == "Visible summary"


def test_strip_think_tags_returns_empty_when_response_is_only_a_think_block() -> None:
    """A response with only a <think> block and no trailing content yields an empty string."""
    assert strip_think_tags("<think>internal reasoning with no answer</think>") == ""


def test_collect_pending_files_partitions_and_sorts_by_channel_backlog(tmp_path: Path) -> None:
    """Pending files are counted, filtered, and ordered by smallest channel backlog."""
    cleaned_dir = tmp_path / "cleaned"
    summaries_dir = tmp_path / "summaries"

    write_text(cleaned_dir / "large" / "b.txt", "pending b")
    write_text(cleaned_dir / "large" / "a.txt", "pending a")
    write_text(cleaned_dir / "small" / "c.txt", "pending c")
    write_text(cleaned_dir / "empty" / "d.txt", "   ")
    write_text(cleaned_dir / "done" / "e.txt", "already summarized")
    write_text(cleaned_dir / "large" / "._ignored.txt", "metadata fork")
    write_text(summaries_dir / "done" / "e.md", "summary")

    pending, total, already_done, empty_files = collect_pending_files(cleaned_dir, summaries_dir, "")

    assert total == 5
    assert already_done == 1
    assert empty_files == 1
    assert [(txt.parent.name, txt.name, output.parent.name, output.name) for txt, output in pending] == [
        ("small", "c.txt", "small", "c.md"),
        ("large", "a.txt", "large", "a.md"),
        ("large", "b.txt", "large", "b.md"),
    ]


def test_collect_pending_files_applies_channel_filter(tmp_path: Path) -> None:
    """A channel filter limits counts and pending files to that directory."""
    cleaned_dir = tmp_path / "cleaned"
    summaries_dir = tmp_path / "summaries"

    write_text(cleaned_dir / "alpha" / "a.txt", "alpha")
    write_text(cleaned_dir / "beta" / "b.txt", "beta")

    pending, total, already_done, empty_files = collect_pending_files(cleaned_dir, summaries_dir, "beta")

    assert total == 1
    assert already_done == 0
    assert empty_files == 0
    assert [(txt.parent.name, txt.name) for txt, _ in pending] == [("beta", "b.txt")]
