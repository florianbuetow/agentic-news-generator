# Troubleshooting

Catalogue of helper scripts in `scripts/` and `tools/` for finding files and diagnosing pipeline issues. All scripts read paths from `config/config.yaml` unless noted. Run from project root.

## Quick Reference

| Purpose | Command |
|---------|---------|
| Find everything for a video ID | `just find-files <VIDEO_ID>` |
| Check LM Studio + required models | `just status` |
| Pipeline data counts | `just stats` |
| Video file corruption scan | `just check-video-integrity` |
| Is audio track audible? | `just check-audio-track <CHANNEL> <VIDEO_ID>` |
| Transcripts ≤100 bytes | `just find-empty-transcripts` |
| All empty files in data dir | `just clean-empty-files` |
| Total transcribed hours | `just audio-hours` |
| Fetch missing `.info.json` | `just fetch-video-metadata <CHANNEL> <ID...>` |
| Nuke every file for a video ID | `just clean-video-files VIDEO_ID=<id>` |

---

## Finding Files

### `scripts/find-files.sh` — `just find-files <VIDEO_ID>`
Scan all data directories in `config.yaml` for filenames containing the video ID substring. Shortest-unique-prefix dedupe avoids duplicate hits from nested paths.

### `scripts/find-empty-transcripts.sh` — `just find-empty-transcripts`
List transcript `*.txt` files ≤100 bytes under `data_downloads_transcripts_dir`. Grouped by channel.

### `scripts/find-and-clean-empty-data-files.py` — `just clean-empty-files`
Walk the data root, find all 0-byte files (skipping `.gitkeep`, `.DS_Store`), list them, prompt to delete.

### `scripts/fetch-video-metadata.py` — `just fetch-video-metadata <CHANNEL> <ID...>`
For each video ID, look up the existing WAV stem in the channel audio dir and fetch `.info.json` via yt-dlp to match that stem. Fixes pipelines that key metadata by WAV stem.

---

## Identifying Corrupt / Broken Files

### `scripts/check_video_integrity.py` — `just check-video-integrity`
`ffprobe`-based integrity scan for video files. Flags corruption, bitrates <1000 bps, duration mismatches. Caches hashes under `.cache/` so reruns are fast. Reject-list output for downstream cleanup.

### `scripts/check-audio-track.sh` — `just check-audio-track <CHANNEL> <VIDEO_ID>`
Uses `ffprobe` + `ffmpeg volumedetect` to check a single video for (a) presence of audio stream, (b) non-empty samples, (c) mean volume above -40 dB. Exit 0 ok, 1 missing/quiet, 2 usage error. Useful when transcript is empty and you suspect silent source.

### `scripts/filter-short-videos.py`
Dry-run scan: for each channel, flag files with no audio stream or duration < `--max-duration` (default 120s). Adds `Channel/video_id` entries to `config/filefilter.json`. Pass `--write` to persist. Pair with `remove-filtered-files.py`.

### `scripts/remove-filtered-files.py`
Sweep files listed in `config/filefilter.json` plus upstream copies (transcripts → audio → videos) by `[<video_id>]` substring. Dry-run by default; pass `--execute` to unlink.

### `scripts/clean-video-files.py` — `just clean-video-files VIDEO_ID=<id>`
Interactive: lists every file containing `[<video_id>]`, asks which to delete, optionally removes the ID from yt-dlp `downloaded.txt` archive. Every destructive action confirmed.

---

## Identifying Transcript Quality Issues

### `scripts/transcript-hallucination-detection.py` — `just analyze-transcripts-hallucinations`
Repetition-based hallucination detection over SRT files. Writes per-file hallucination records. Run before `transcript-hallucination-removal.py` (LLM cleanup).

### `scripts/create-hallucination-digest.py`
Grouped digest of the above detector's JSON output. Summarises which files have how many suspect segments. Runs as part of `analyze-transcripts-hallucinations`.

### `scripts/transcript-language-analysis.py`
Language-detect each transcript in ~1000-word chunks via FastText. Per-channel markdown reports flag mixed-language or non-target-language files.

### `scripts/cleanup-empty-transcripts.sh`
Removes the whole transcript file group (srt/txt/vtt/tsv/json) whenever the `.srt` is 0 bytes. Ran automatically by `just transcribe`; invoke manually with `bash scripts/cleanup-empty-transcripts.sh <transcripts_dir>`.

---

## System & Pipeline Status

### `scripts/lmstudio_status.py` — `just status`
Probes LM Studio `/v1/models` at every `api_base` referenced in `config.yaml`. Reports which required models per section are loaded. Run first when LLM calls fail.

### `scripts/status.py` — `just stats`
Pipeline counters: videos downloaded, audio extracted, transcripts written, metadata present, articles compiled. Quick sanity check of stage completeness.

### `scripts/audio-hours.py` — `just audio-hours`
Sum last SRT timestamp per transcript → total corpus hours. Useful for progress tracking and capacity planning.

### `scripts/check/config_template.py` — `just code-config`
Compare `config/config.yaml` against `config/config.yaml.template`; fail on missing/extra keys. First thing to check after config edits.

### `scripts/validate_articles_dir.py`
Validate `data_input/newspaper/articles/` exists and contains markdown. Exit 1 on failure. Runs before `newspaper-generate` / `newspaper-serve`.

---

## AI-Powered Reviewers (`tools/`)

### `tools/fake_test_detector/detect_fake_tests.py` — `just ai-review-unit-tests` / `-nocache`
Autogen + local LLM scan of `tests/` via AST. Flags tests that don't actually exercise the code. Results cached by file hash in `.cache/unit_test_hashes.json`.

### `tools/shellscript_analyzer/shellscript_analyzer.py` — `just ai-review-shell-scripts` / `-nocache`
Autogen scan of every `.sh` file. Flags env-var reliance and missing CLI parameterisation. Cache at `.cache/shell_script_hashes.json`.

### `tools/llm-topic-performance/benchmark.py`
Queries LM Studio for all completion models, runs each against the production topic-extraction prompt, logs timing + output under `reports/llm-topic-extraction-comparison/`. Direct `uv run python tools/llm-topic-performance/benchmark.py`.

---

## Code Quality Diagnostics (grouped)

Run individually when CI fails:

| Target | Script |
|--------|--------|
| `just code-style` | ruff check + format |
| `just code-typecheck` | mypy on `src/` |
| `just code-lspchecks` | pyright strict (report under `reports/pyright/`) |
| `just code-security` | bandit (report under `reports/security/`) |
| `just code-deptry` | deptry dependency hygiene |
| `just code-spell` | codespell |
| `just code-semgrep` | semgrep with `config/semgrep/` rules |
| `just code-audit` | pip-audit |
| `just code-stats` | pygount → `reports/code-stats.txt` |

`just ci-quiet` runs them all, showing only the failing target's output.

---

## Playbook: Empty Transcript for a Video

1. `just find-files <VIDEO_ID>` — confirm files exist.
2. `just check-audio-track <CHANNEL> <VIDEO_ID>` — is source silent?
3. If silent → `uv run scripts/filter-short-videos.py --channel <CHANNEL> --write` then `uv run scripts/remove-filtered-files.py --execute`.
4. If audible but transcript empty → `just clean-video-files VIDEO_ID=<id>` and redownload (re-run `just download-videos`).

## Playbook: Corrupt Video Suspected

1. `just check-video-integrity` — flags all corrupt files.
2. `just clean-video-files VIDEO_ID=<id>` — interactive removal + archive cleanup.
3. `just download-videos` — redownload (archive entry removed in step 2 lets yt-dlp refetch).

## Playbook: Pipeline Stalls at LLM Step

1. `just status` — verify LM Studio + required models loaded.
2. `uv run python tools/llm-topic-performance/benchmark.py` — sanity-check model throughput.
3. Check `reports/` for most recent stage output.

---

## Temporary Debug Scripts

- All ad-hoc test/debug scripts go in `debug/` subfolder
- Makes cleanup trivial: scripts in `debug/` are disposable by definition
- `debug/` is gitignored — never commit debug scripts
- Never create Python files in the project root (use `src/`, `scripts/`, or `debug/`)

---

## Jupyter Notebook Validation

When modifying `.ipynb` files, validate before committing:

**JSON structure:**
```bash
uv run python -m json.tool notebook.ipynb > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
```

**Python syntax (no execution):**
```bash
uv run python -c "
import nbformat
nb = nbformat.read('notebook.ipynb', as_version=4)
for cell in nb.cells:
    if cell.cell_type == 'code':
        compile(cell.source or '', '<cell>', 'exec')
print('Valid Python syntax')
"
```

**Full execution:**
```bash
jupyter nbconvert --execute --to notebook --inplace --allow-errors notebook.ipynb --ExecutePreprocessor.timeout=-1
```
