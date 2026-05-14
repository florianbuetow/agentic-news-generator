#!/usr/bin/env python3
"""Generate topic-boundary JSON for each hallucination-freed SRT transcript.

Sequential by design: the launcher rewrites a shared PROMPT.generated.md, so
parallel invocations would race. The launcher runs Codex under a workspace-write
sandbox that can only write inside its workdir, /tmp, $TMPDIR, and
~/.codex/memories. We hand Codex a /tmp output path so it can write, then move
the result to the real destination outside the sandbox.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config


def main() -> int:
    """Drive the topic-boundary launcher once per SRT, writing JSON per video."""
    project_root = Path(__file__).parent.parent
    config = Config(project_root / "config" / "config.yaml")

    topics_config = config.get_topics_config()
    data_dir = config.getDataDir()
    input_dir = data_dir / topics_config.input_dir
    output_dir = data_dir / topics_config.output_dir
    launcher = Path(topics_config.launcher_script)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return 1
    if not launcher.is_file():
        print(f"Error: Launcher script not found: {launcher}", file=sys.stderr)
        return 1

    srt_files = sorted(p for p in input_dir.rglob("*.srt") if not p.name.startswith("._"))
    total = len(srt_files)
    print(f"Found {total} SRT file(s) under {input_dir}")

    tmp_root = Path(tempfile.gettempdir())
    processed = 0
    skipped = 0
    failed = 0

    for index, srt_file in enumerate(srt_files, start=1):
        relative = srt_file.relative_to(input_dir)
        output_json = output_dir / relative.parent / f"{srt_file.stem}_topics.json"

        if output_json.exists():
            skipped += 1
            print(f"[{index}/{total}] skip (exists): {relative}")
            continue

        print(f"[{index}/{total}] processing: {relative}")
        tmp_output = tmp_root / f"extract_topics_{uuid.uuid4().hex}.json"

        try:
            result = subprocess.run(
                ["bash", str(launcher), str(srt_file), str(tmp_output)],
                check=False,
            )
            if result.returncode == 0 and tmp_output.exists():
                output_json.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_output), output_json)
                processed += 1
            else:
                failed += 1
                print(f"  FAILED (rc={result.returncode}): {relative}", file=sys.stderr)
        finally:
            if tmp_output.exists():
                tmp_output.unlink()

    print(f"\nDone. processed={processed} skipped={skipped} failed={failed} total={total}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
