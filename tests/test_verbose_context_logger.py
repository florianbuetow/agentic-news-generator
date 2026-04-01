"""Tests for the per-run artifact logger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.article_generation.chief_editor.verbose_context_logger import VerboseContextLogger


class _OverflowLogger(VerboseContextLogger):
    def force_overflow(self) -> None:
        self._sequence = 900


def test_log_writes_sequential_files(tmp_path: Path) -> None:
    logger = VerboseContextLogger(artifacts_dir=tmp_path)

    first_path = logger.log(agent_name="writer", step="prompt", content="hello", fmt="md")
    second_path = logger.log(agent_name="review", step="output", content="world", fmt="txt")

    assert first_path.name == "000_writer_prompt.md"
    assert second_path.name == "001_review_output.txt"
    assert first_path.read_text(encoding="utf-8") == "hello"
    assert second_path.read_text(encoding="utf-8") == "world"


def test_log_final_uses_900_range(tmp_path: Path) -> None:
    logger = VerboseContextLogger(artifacts_dir=tmp_path)

    first_final = logger.log_final(agent_name="final", step="article", content="done", fmt="md")
    second_final = logger.log_final(agent_name="final", step="report", content={"ok": True}, fmt="json")

    assert first_final.name == "900_final_article.md"
    assert second_final.name == "901_final_report.json"
    assert json.loads(second_final.read_text(encoding="utf-8")) == {"ok": True}


def test_log_parses_json_string_before_writing(tmp_path: Path) -> None:
    logger = VerboseContextLogger(artifacts_dir=tmp_path)

    path = logger.log(agent_name="source", step="metadata", content='{"channel":"Test"}', fmt="json")

    assert json.loads(path.read_text(encoding="utf-8")) == {"channel": "Test"}


def test_log_rejects_invalid_json_string(tmp_path: Path) -> None:
    logger = VerboseContextLogger(artifacts_dir=tmp_path)

    with pytest.raises(ValueError, match="not valid JSON"):
        logger.log(agent_name="source", step="metadata", content="{not-json}", fmt="json")


def test_log_rejects_unknown_format(tmp_path: Path) -> None:
    logger = VerboseContextLogger(artifacts_dir=tmp_path)

    with pytest.raises(ValueError, match="fmt must be one of"):
        logger.log(agent_name="writer", step="prompt", content="x", fmt="csv")


def test_log_raises_on_sequence_overflow(tmp_path: Path) -> None:
    logger = _OverflowLogger(artifacts_dir=tmp_path)
    logger.force_overflow()

    with pytest.raises(RuntimeError, match="sequence overflow"):
        logger.log(agent_name="writer", step="prompt", content="x", fmt="txt")
