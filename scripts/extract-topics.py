#!/usr/bin/env python3
"""Generate topic-boundary JSON for each hallucination-freed SRT transcript.

Drives `codex exec` directly with a rendered prompt per SRT. Codex writes the
result to a staged /tmp path; we validate it and move it to the final output.
Sequential by design — a single Codex CLI session is invoked per file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config

REASONING_EFFORT = "high"


class TopicExtractionError(RuntimeError):
    """Raised when topic extraction fails for a single SRT file."""


@dataclass
class ProgressTracker:
    """Track progress with ETA based on average time per processed file."""

    total_files: int
    current_index: int = 0
    start_time: float = 0.0

    def start(self) -> None:
        """Initialize the wall-clock timer at the first iteration."""
        self.start_time = time.time()
        self.current_index = 0

    def next_file(self) -> None:
        """Advance the 1-based index before producing the prefix for the next iteration."""
        self.current_index += 1

    def format_duration(self, seconds: float) -> str:
        """Format `seconds` as `[Xh:Ym]` matching the convention used elsewhere in this repo."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"[{hours:02d}h:{minutes:02d}m]"

    def get_progress_prefix(self) -> str:
        """Return `[N/total] (P%) ETA [Xh:Ym]`; placeholder ETA on the very first iteration."""
        if self.total_files == 0:
            return ""
        pct = (self.current_index / self.total_files) * 100
        completed = self.current_index - 1
        if completed > 0:
            elapsed = time.time() - self.start_time
            avg_time = elapsed / completed
            remaining = self.total_files - self.current_index + 1
            eta_str = f" ETA {self.format_duration(avg_time * remaining)}"
        else:
            eta_str = " ETA [--h:--m]"
        return f"[{self.current_index}/{self.total_files}] ({pct:.0f}%){eta_str}"


def _validate_chapter(chapter_value: object, index: int) -> None:
    if not isinstance(chapter_value, dict):
        raise TopicExtractionError(f"chapters[{index}] is not an object")
    chapter = cast("dict[str, object]", chapter_value)
    if "start_seconds" not in chapter:
        raise TopicExtractionError(f"chapters[{index}] missing 'start_seconds'")
    start: object = chapter["start_seconds"]
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        raise TopicExtractionError(f"chapters[{index}].start_seconds must be a number, got {type(start).__name__}")
    if start < 0:
        raise TopicExtractionError(f"chapters[{index}].start_seconds is negative ({start})")
    if "title" not in chapter:
        raise TopicExtractionError(f"chapters[{index}] missing 'title'")
    title: object = chapter["title"]
    if not isinstance(title, str) or not title.strip():
        raise TopicExtractionError(f"chapters[{index}].title must be a non-empty string")


def validate_topic_json(path: Path) -> None:
    """Raise TopicExtractionError describing the first structural problem found in `path`."""
    if not path.exists():
        raise TopicExtractionError("codex produced no output file")
    if path.stat().st_size == 0:
        raise TopicExtractionError("output file is empty")
    raw = path.read_text(encoding="utf-8")
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError as e:
        raise TopicExtractionError(f"output is not valid JSON: {e.msg} at line {e.lineno} col {e.colno}") from e
    if not isinstance(parsed, dict):
        raise TopicExtractionError("top-level JSON value is not an object")
    parsed_dict = cast("dict[str, object]", parsed)
    if "chapters" not in parsed_dict:
        raise TopicExtractionError("missing required key 'chapters'")
    chapters_value: object = parsed_dict["chapters"]
    if not isinstance(chapters_value, list):
        raise TopicExtractionError("'chapters' is not a list")
    chapters = cast("list[object]", chapters_value)
    if len(chapters) == 0:
        raise TopicExtractionError("'chapters' list is empty")
    for index, chapter_value in enumerate(chapters):
        _validate_chapter(chapter_value, index)


def _dump_codex_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print("--- codex stdout ---", file=sys.stderr)
        print(result.stdout.rstrip(), file=sys.stderr)
    if result.stderr:
        print("--- codex stderr ---", file=sys.stderr)
        print(result.stderr.rstrip(), file=sys.stderr)


def _parse_codex_banner(stdout: str) -> dict[str, str]:
    """Extract `model:`, `reasoning effort:`, and `session id:` lines from codex banner."""
    fields: dict[str, str] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        for key in ("model", "reasoning effort", "session id"):
            prefix = f"{key}:"
            if line.startswith(prefix):
                fields[key] = line[len(prefix) :].strip()
    return fields


def _assert_codex_banner(stdout: str, expected_model: str, expected_effort: str) -> dict[str, str]:
    """Raise if codex banner reports a model or reasoning effort that disagrees with what we asked for."""
    fields = _parse_codex_banner(stdout)
    actual_model = fields.get("model")
    actual_effort = fields.get("reasoning effort")
    if actual_model != expected_model:
        raise TopicExtractionError(f"codex banner reports model={actual_model!r}, expected {expected_model!r}")
    if actual_effort != expected_effort:
        raise TopicExtractionError(f"codex banner reports reasoning effort={actual_effort!r}, expected {expected_effort!r}")
    return fields


def _render_prompt(template: str, srt_file: Path, output_json: Path) -> str:
    return template.replace("{{INPUT_SRT_FILE}}", str(srt_file)).replace("{{OUTPUT_JSON_FILE}}", str(output_json))


def _process_one(
    srt_file: Path,
    output_json: Path,
    prompt_template: str,
    codex_model: str,
    tmp_output: Path,
    relative: Path,
) -> None:
    prompt = _render_prompt(prompt_template, srt_file, tmp_output)
    result = subprocess.run(
        [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "-m",
            codex_model,
            "-c",
            f"model_reasoning_effort={REASONING_EFFORT}",
            prompt,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _dump_codex_output(result)
        raise TopicExtractionError(f"codex exited with rc={result.returncode} for {relative}")
    try:
        _assert_codex_banner(result.stdout, expected_model=codex_model, expected_effort=REASONING_EFFORT)
        validate_topic_json(tmp_output)
    except TopicExtractionError:
        _dump_codex_output(result)
        raise
    output_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_output), output_json)


def main() -> None:
    """Drive Codex once per SRT, writing validated topic-boundary JSON per video."""
    project_root = Path(__file__).parent.parent
    config = Config(project_root / "config" / "config.yaml")

    topics_config = config.get_topics_config()
    data_dir = config.get_data_dir()
    input_dir = data_dir / topics_config.input_dir
    output_dir = data_dir / topics_config.output_dir
    prompt_template_path = project_root / topics_config.prompt_template
    codex_model = topics_config.codex_model

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not prompt_template_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")

    prompt_template = prompt_template_path.read_text(encoding="utf-8")

    srt_files = sorted(p for p in input_dir.rglob("*.srt") if not p.name.startswith("._"))
    total = len(srt_files)

    def _output_path_for(srt: Path) -> Path:
        return output_dir / srt.relative_to(input_dir).parent / f"{srt.stem}_topics.json"

    files_to_process = [srt for srt in srt_files if not _output_path_for(srt).exists()]
    pending = len(files_to_process)
    skipped = total - pending

    print(f"Found {total} SRT file(s) under {input_dir} — {pending} to process, {skipped} already done")

    tmp_root = Path(tempfile.gettempdir())
    processed = 0
    progress = ProgressTracker(total_files=pending)
    progress.start()

    for srt_file in files_to_process:
        progress.next_file()
        relative = srt_file.relative_to(input_dir)
        output_json = _output_path_for(srt_file)
        prefix = progress.get_progress_prefix()

        print(f"{prefix} Extracting topic boundaries for {relative} ...")
        tmp_output = tmp_root / f"extract_topics_{uuid.uuid4().hex}.json"

        try:
            _process_one(srt_file, output_json, prompt_template, codex_model, tmp_output, relative)
            processed += 1
        finally:
            if tmp_output.exists():
                tmp_output.unlink()

    print()
    print("=" * 50)
    print(f"Completed: {processed} processed, {skipped} skipped (total: {total})")


if __name__ == "__main__":
    main()
