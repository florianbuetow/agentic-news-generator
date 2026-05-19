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
| Find format-code artifacts | `find <audio_dir> -name '*.f[0-9]*.wav'` |

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

### `scripts/filter-short-videos.py` — `just filter-videos` (step 1 of 2)
For each channel, flags files with no audio stream or duration below `transcription.min_duration` (from `config.yaml`). Adds `Channel/video_id` entries to `config/filefilter.json`. Always writes; no CLI args. Run in tandem with `remove-filtered-files.py` via `just filter-videos`.

### `scripts/remove-filtered-files.py` — `just filter-videos` (step 2 of 2)
Sweeps files listed in `config/filefilter.json` plus upstream copies (transcripts → audio → videos) by `[<video_id>]` substring. Always deletes; no CLI args, no dry-run mode.

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

## Playbook: Non-English / Language Detection False Positive

When `transcript-language-analysis.py` flags a file as non-English, **check content quality first** before assuming genuine foreign language content.

Degenerate transcripts (Whisper hallucinations with repetitive tokens like "AI, AI, AI...") trigger false language classifications because short repeated tokens resemble words in other languages (e.g. "ai" is French for "have").

1. `just find-files <VIDEO_ID>` — confirm transcript file exists and find its path.
2. Check file size: `ls -lh <transcript.txt>` — suspiciously small or unexpectedly large files indicate problems.
3. Read first ~20 lines: look for repeated tokens, all-caps acronyms, or single words repeated thousands of times.
4. If degenerate/hallucinated → treat as corrupt transcript, follow **Playbook: Empty Transcript for a Video**.
5. If genuinely non-English → decide whether to keep, filter, or redownload with correct language settings.

**General rule:** content size and basic content sanity (empty files, missing files, degenerate repetition) are the **first things to check** for any transcript quality issue, before deeper analysis.

---

## Playbook: Empty Transcript for a Video

1. `just find-files <VIDEO_ID>` — confirm files exist.
2. `just check-audio-track <CHANNEL> <VIDEO_ID>` — is source silent?
3. If silent → `just filter-videos` (sweeps every channel for short / no-audio videos and unlinks them).
4. If audible but transcript empty → `just clean-video-files VIDEO_ID=<id>` and redownload (re-run `just download-videos`).

## Playbook: Corrupt Video Suspected

1. `just check-video-integrity` — flags all corrupt files.
2. `just clean-video-files VIDEO_ID=<id>` — interactive removal + archive cleanup.
3. `just download-videos` — redownload (archive entry removed in step 2 lets yt-dlp refetch).

---

## Playbook: Removing Download Artifacts — Re-download Decision

When deleting any artifact for a video ID (`.mp4`, `.mp4.part`, `.wav`, `.info.json`, `.silence_map.json`, transcript files), **always ask explicitly** whether to also remove the entry from `downloaded.txt` (yt-dlp archive).

**Decision rule:**

- **Re-download wanted** (remove from `downloaded.txt`): genuine corrupt download, network failure mid-download, transcribed but transcript was lost.
- **Re-download NOT wanted** (keep in `downloaded.txt`): video has no spoken word (music-only, silent, ambient), live broadcast / premiere downloaded by mistake, video filtered by content rules, video known to be unusable.

Re-downloading a video without spoken word wastes bandwidth and disk and will be filtered out again.

**Default: keep entry in `downloaded.txt`.** Only remove on explicit user confirmation that re-download is desired.

## Playbook: Transcription Fails With "Metadata file not found"

When `just transcribe` aborts with `Metadata file not found`, run `just find-files <VIDEO_ID>` first to identify which case applies.

### Case 1: `.info.json` exists in `videos/` but not in `metadata/<CHANNEL>/video/`

The video was downloaded but its metadata was never placed where the transcription script expects it. Fix immediately — no confirmation needed:

```bash
just fetch-video-metadata <CHANNEL> <VIDEO_ID>
```

Then re-run `just transcribe`.

### Case 2: Format-code artifact (`.f251-11.wav` or similar)

The root cause is yt-dlp intermediate files left behind from a failed format merge.

**How to identify:** The audio filename contains a format code like `.f251-11` between the video ID bracket and `.wav` — e.g. `Title [VIDEO_ID].f251-11.wav`. The corresponding `.wav` without the format code usually also exists. Metadata files never carry the format code, so the transcription script's stem-based lookup fails.

**Steps:**

1. `just find-files <VIDEO_ID>` — list all files for the video. Look for duplicate `.wav` entries with and without the format code.
2. Remove **both** the `.f251-11.wav` artifact and the normal `.wav` (the normal one may also be from the same broken download session).
3. Remove the video ID from `downloaded.txt` in the channel's videos directory:
   ```bash
   sed -i '' '/<VIDEO_ID>/d' /path/to/videos/<CHANNEL>/downloaded.txt
   ```
4. `just download-videos` — re-downloads the video cleanly.
5. `just extract-audio` — regenerates the `.wav` from the fresh `.mp4`.
6. Also clean up stale `.f251-11.silence_map.json` files under `metadata/<CHANNEL>/audio/` if they exist.
7. `just transcribe` — should now succeed.

**Why the normal `.wav` must also be removed:** The `.mp4` may still be intact on disk, so `just extract-audio` will skip extraction if a `.wav` already exists. Removing it forces re-extraction from the clean source.

**Why `just transcribe` may show 0 pending after extract-audio:** If transcription was started before `extract-audio` finished, the channel's new `.wav` files aren't visible to that run. Run `just transcribe` again after `extract-audio` completes.

---

## Playbook: YouTube Cookies Expired During Download

When `just download-videos` fails with `ERROR: YouTube cookies are expired or invalid. Aborting.`, Chrome's YouTube cookies have rotated mid-session. Channels processed before the rotation succeed; those after fail.

1. Open Chrome, visit youtube.com, ensure you're logged in.
2. Re-run `just download-videos` — yt-dlp re-extracts fresh cookies from Chrome on each channel.
3. If it keeps failing, see yt-dlp wiki: https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies

---

## Playbook: Pipeline Stalls at LLM Step

1. `just status` — verify LM Studio + required models loaded.
2. `uv run python tools/llm-topic-performance/benchmark.py` — sanity-check model throughput.
3. Check `reports/` for most recent stage output.

---

## Playbook: `just summarize-transcripts` Fails With "LLM returned empty response"

Symptom: `summarize-transcripts.py` aborts with `ValueError: LLM returned empty response` after `Attempt 1/3`…`Attempt 3/3 failed` warnings.

What the code does (verified at `scripts/summarize-transcripts.py:81-148`):

- `call_llm` raises `ValueError("LLM returned empty response")` when `response.choices[0].message.content` is `None` or whitespace-only.
- `process_single_file` retries `summarize_transcripts.llm.max_retries` times with `time.sleep(summarize_transcripts.llm.retry_delay)` between attempts; after the last attempt the exception propagates and the recipe exits 1.
- If consecutive attempt timestamps in the log are much further apart than `retry_delay`, the LLM call itself is taking that long to return — the script is not stuck in the sleep.
- The log line immediately before `Attempt 1/3 failed` is `[N/M] X.X% ETA ... <channel>/<file>.txt` — that path under `downloads/transcripts_cleaned/<channel>/` is the input that triggered the failure.

Possible causes (not ranked — verify each before acting):

- LM Studio is reachable but the loaded model is no longer responding (unloaded, crashed, or swapped). `just status` will show which models are currently loaded.
- The configured model returns content that is empty after LM Studio's own processing (for example, reasoning content is delivered out-of-band and the visible `content` field is empty). Confirm by inspecting LM Studio's server log or `reports/` for the actual response payload before claiming this is the cause.
- The specific transcript is degenerate (hallucinated repetition, near-empty after cleaning) and the model returns no output for it. The size threshold check in `process_single_file` only rejects oversized inputs, not low-quality ones.

Steps:

1. `just status` — record which models are loaded. Compare against `summarize_transcripts.llm.model` in `config/config.yaml`.
2. Read the log line preceding `Attempt 1/3 failed` to identify the offending transcript path.
3. Inspect that transcript: `ls -lh <path>` for size, then read the first ~20 lines. If it shows hallucination patterns (repeated tokens, single phrase repeated) → treat as corrupt transcript and follow **Playbook: Empty Transcript for a Video**. See **Playbook: Non-English / Language Detection False Positive** for the same content-sanity checks.
4. If the model from step 1 is not loaded or differs from config, load the configured one:
   ```bash
   lms unload --all
   lms load <model-name-from-config>
   ```
   Re-run `just summarize-transcripts`. The script skips files whose output already exists (`process_single_file` early-returns on `output_file.exists()`), so it resumes at the failed file.
5. If steps 1–4 do not identify a cause, capture the next failure with LM Studio's server log open (or add a temporary `logger.warning(response)` before the empty check in `call_llm`) to see what LM Studio actually returned. Decide remediation from that evidence rather than guessing.

Do not delete the source transcript to skip the file — the next run will hit it again. Either fix the transcript (step 3) or resolve the LLM-side cause (steps 4–5).

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
