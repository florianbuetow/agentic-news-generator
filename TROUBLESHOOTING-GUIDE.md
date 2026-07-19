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
| Find interrupted-merge leftovers | `find <videos_dir> -name '*.temp.mp4' -o -name '*.f[0-9]*.webm'` |
| Find macOS AppleDouble sidecars | `find <data_dir> -name '._*'` |

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

### `scripts/find-files-with-filtered-video-ids.py` — `just find-files-with-filtered-video-ids`
Read-only audit. Builds a set of every video ID in `config/filefilter.json`, then walks each ID-convention data folder channel-by-channel and flags any file whose `[<video_id>]` is in that set. Because `remove-filtered-files.py` only sweeps the `videos → audio → transcripts` chain, downstream artifacts (hallucination JSON, cleaned transcripts, summaries, metadata) of a filtered video can survive — this surfaces them. Prints one `ERROR:` line per offender; exits `1` if any are found, else `0`. No CLI args.

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

`just ci` runs them all, showing only the failing target's output.

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

## Playbook: Extract-Audio Skips a Channel Entirely (0 Files Processed)

When `just extract-audio` reports `0 to process` for a channel and its `downloads/videos/<channel>/` contains only `downloaded.txt`, **check whether the videos were already transcribed** before investigating further.

**Step 1 — Count existing transcripts:**

```bash
ls downloads/transcripts/<channel>/*.srt | wc -l
```

If the count matches the number of entries in `downloaded.txt`, the pipeline ran to completion for this channel. No action needed.

**Why this happens:** `just archive-videos` (step 6 of `video-all`) moves every transcribed MP4 from `downloads/videos/<channel>/` to `archive/videos/<channel>/` and deletes the corresponding WAV. After archival, `downloads/videos/<channel>/` is empty and `extract-audio` has nothing to process — which is correct behaviour.

**If transcripts are missing** (count is lower than expected), the videos were archived prematurely or the transcription step was skipped. In that case:

1. Identify which video IDs have an archived MP4 but no `.srt`: compare `ls archive/videos/<channel>/*.mp4` against `ls downloads/transcripts/<channel>/*.srt`.
2. For each missing transcript, move the MP4 back: `mv archive/videos/<channel>/<file>.mp4 downloads/videos/<channel>/`.
3. `just extract-audio` — re-extracts WAV from the restored MP4.
4. `just transcribe` — generates the missing transcript.
5. `just archive-videos` — re-archives once transcription is complete.

---

## Playbook: Extract-Audio Fails With "No Audio Stream Found"

When `just extract-audio` reports `❌ FAILED: No audio stream found (video-only file)`, first confirm the downloaded file has no audio stream, then determine whether YouTube had audio available.

**Step 1 — Confirm with ffprobe that the MP4 has no audio stream:**

```bash
ffprobe -v error -select_streams a:0 -show_entries stream=codec_name \
  -of default=noprint_wrappers=1:nokey=1 "path/to/videos/<CHANNEL>/<TITLE> [VIDEO_ID].mp4"
```

Empty output = no audio stream in the file. Non-empty = audio present (extract-audio should not have failed; investigate elsewhere).

**Step 2 — Inspect the `.info.json` to check what formats were available on YouTube:**

```python
import json, os
path = 'path/to/metadata/<CHANNEL>/video/<TITLE> [VIDEO_ID].info.json'
with open(path) as f:
    d = json.load(f)
audio_formats = [(fmt['format_id'], fmt['acodec'], fmt['ext'])
                 for fmt in d.get('formats', [])
                 if fmt.get('acodec') not in (None, 'none')]
print(len(audio_formats), 'audio formats available')
```

- **Audio formats present** → YouTube has audio; the download failed to merge it (yt-dlp picked a video-only stream or the merge step failed). Re-download:
  1. `just clean-video-files VIDEO_ID=<id>` — remove the video-only MP4 and WAV artifacts, remove from `downloaded.txt`.
  2. `just download-videos` — re-fetches with proper format selection.
  3. `just extract-audio` — regenerates the WAV.

- **No audio formats** → genuinely audio-free content (music video, silent, ambient). Treat as per **Playbook: Empty Transcript for a Video** step 3: `just filter-videos`.

---

## Playbook: Interrupted Merge — Unmerged Stream Artifacts in `videos/`

A variant of the above that reports the same "No audio stream found" error, but for a different reason: yt-dlp downloaded both streams successfully and was **interrupted during the ffmpeg merge**. No merged `.mp4` was ever produced.

**How to identify** — `just find-files <VIDEO_ID>` shows several files in `downloads/videos/<CHANNEL>/` where there should be exactly one `.mp4`:

| Artifact | Contents |
|---|---|
| `[…].f137.mp4` | video-only (h264) — the source of the "no audio stream" error |
| `[…].f251.webm` | audio-only (opus) |
| `[…].temp.mp4` | partial merge output — `ffprobe` reports `moov atom not found` |

Confirm with `ffprobe -v error -show_entries stream=codec_type,codec_name -of csv=p=0 <file>` on each.

**The trap:** `extract-audio` succeeds on the audio-only `.webm` and writes `[…].f251.wav` — a *complete, valid* WAV carrying a format-code suffix. It looks like a usable result and tempts you into keeping it. Do not. It is derived from the abandoned download, and keeping it prevents re-extraction from the clean merge (see the derived-WAV rule under **Playbook: Removing Download Artifacts**). It will also break metadata lookup — see **Case 2** of the "Metadata file not found" playbook.

**Steps:**

1. `just find-files <VIDEO_ID>` — enumerate every artifact.
2. `rm` all three video-dir artifacts (`.f*.mp4`, `.f*.webm`, `.temp.mp4`) — one literal `rm` per file.
3. `rm` the derived `[…].f*.wav` **and** `[…].f*.silence_map.json`.
4. Remove the ID from the archive so yt-dlp will refetch:
   ```bash
   sed -i '' '/<VIDEO_ID>/d' <videos_dir>/<CHANNEL>/downloaded.txt
   ```
   Verify with `grep -c <VIDEO_ID> …/downloaded.txt` → must print `0`.
5. `just download-videos` → `just extract-audio` → `just transcribe`.

`.info.json` and `.webp` can stay — yt-dlp overwrites both on re-download.

**Note the archive bug:** yt-dlp wrote the ID to `downloaded.txt` even though the merge never completed, so without step 4 the video is skipped forever while `extract-audio` fails on the leftovers **every single run**. That is the loop this playbook breaks. A permanent fix would verify the merged MP4 exists and has both streams before trusting the archive write.

---

## Playbook: Corrupt Video Suspected

1. `just check-video-integrity` — flags all corrupt files.
2. `just clean-video-files VIDEO_ID=<id>` — interactive removal + archive cleanup.
3. `just download-videos` — redownload (archive entry removed in step 2 lets yt-dlp refetch).

---

## Playbook: Removing Download Artifacts — Re-download Decision

### Rule: deleting a stream artifact means deleting its derived WAV

**Whenever you remove a partial or broken download artifact — a video-only stream (`.f137.mp4`), an audio-only stream (`.f251.webm`), a `.temp.mp4`, a `.part`, or a corrupt merged `.mp4` — you MUST also remove every artifact derived from it:**

- the `.wav` under `downloads/audio/<CHANNEL>/`
- the `.silence_map.json` under `downloads/metadata/<CHANNEL>/audio/`

**Why this is not optional:** `extract-audio` skips any video whose `.wav` already exists ("WAV already exists"). A WAV left behind from the broken download therefore **survives the re-download and silently wins** — the pipeline transcribes audio from the download you just decided was bad, and the fresh MP4 is never extracted. Nothing errors. You get a clean-looking run built on the discarded artifact.

The same applies in reverse: never delete the WAV and keep the stream artifacts, and never keep a WAV whose format-code suffix no longer matches anything on disk.

**Sequencing:** if the WAV is the only surviving copy of the audio, that is a reason to check the re-download will work (cookies valid, video still public) — **not** a reason to keep the WAV through the re-download. Keeping it defeats the entire operation.

### Archive entry decision

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

## Playbook: Transcription Fails With "UnicodeDecodeError ... invalid start byte"

Symptom: `just transcribe` aborts with a traceback ending in `metadata = json.load(f)` and:

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb0 in position 37: invalid start byte
```

Root cause: the transcription script reads a **macOS AppleDouble shadow file** (`._<name>.info.json`) as if it were the real metadata JSON. AppleDouble files are binary — they begin with the magic header `00 05 16 07 ... "Mac OS X"` — so `json.load` chokes on the first non-UTF-8 byte. The `0xb0` at position 37 is part of that binary header, not real content. (The byte/position may differ; any `invalid start byte` from `json.load` on a metadata file is the same problem.)

Why the wrong file gets picked (verified at `scripts/transcribe_audio.py:122-139`):

- `find_metadata_file` first tries the exact stem `<base_name>.info.json`. For a normal WAV this hits the real file and the bug never triggers.
- When the WAV carries a **format-code suffix** (e.g. `… [VIDEO_ID].f251-11.wav` — see **Case 2** of the previous playbook), the exact stem `… .f251-11.info.json` does not exist, so the code falls back to `sorted(metadata_video_dir.glob("*.info.json"))` and returns the first file whose bracketed video ID matches.
- On the external data drive (a non-APFS/HFS+ filesystem — exFAT/SMB/etc.), macOS writes an AppleDouble `._<name>.info.json` sidecar next to every real file. The glob returns **both**, and `sorted()` ranks `._…` first because `.` (`0x2E`) sorts before `A` (`0x41`). The script therefore opens the binary sidecar.

So two conditions must both hold: a format-code-suffixed WAV **and** AppleDouble sidecars on disk. WAV discovery already filters these out (`scripts/transcribe_audio.py:405,483` — `if not f.name.startswith(".")`); only the metadata glob fallback at `scripts/transcribe_audio.py:135` is missing the same filter.

**Immediate fix — remove the AppleDouble sidecars** (they carry no real content, only resource-fork/xattr metadata):

1. Dry run — list what would be removed under the metadata dir (narrow to one channel with `/<CHANNEL>`):
   ```bash
   find <metadata_dir> -name '._*' -print
   ```
2. Delete them:
   ```bash
   find <metadata_dir> -name '._*' -delete
   ```
3. Re-run `just transcribe`.

**This recurs.** macOS recreates `._` files whenever it touches that filesystem (Finder copy, Spotlight indexing, etc.), so deletion is a band-aid. The permanent fix is in code: `find_metadata_file` must skip dotfiles in its glob fallback, mirroring the WAV-discovery filter at `scripts/transcribe_audio.py:405,483`.

The underlying `.f251-11` format-code artifact is itself a broken-download symptom — see **Case 2** above for cleaning it up and re-downloading.

---

## Playbook: YouTube Cookies Expired During Download

When `just download-videos` fails with `ERROR: YouTube cookies are expired or invalid. Aborting.`, the browser cookies that yt-dlp extracted are no longer accepted by YouTube.

**Which browser is currently configured?**

Check `scripts/config.sh`:
```bash
grep 'BROWSER=' scripts/config.sh
```
The `BROWSER` variable sets the default (currently `firefox`). Override per-run with `BROWSER=chrome just download-videos`.

---

**Fix for Firefox (current default):**

Firefox stores cookies in plain SQLite (Chrome 127+ encrypts them), making extraction reliable.

1. Open Firefox and log into YouTube:
   ```bash
   # macOS
   open -a Firefox "https://www.youtube.com/watch?v=OQSNhk5ICTI"

   # Linux
   firefox "https://www.youtube.com/watch?v=OQSNhk5ICTI"
   ```
2. **Close Firefox completely** (Cmd+Q on macOS) — yt-dlp needs the SQLite WAL flushed before it can copy the file.
3. Run `just download-videos`.

---

**Fix for Chrome:**

Chrome 127+ encrypts its cookie store, so yt-dlp's `--cookies-from-browser chrome` is unreliable — Chrome also rotates session cookies when it detects external access. If you're on Chrome:

1. Open Chrome and visit youtube.com. Ensure you're logged in.
2. Close Chrome completely before running, for the same WAL-flush reason as Firefox.
3. Run `BROWSER=chrome just download-videos` immediately after closing.

If Chrome keeps rotating cookies and the download keeps failing, switch the default to Firefox in `scripts/config.sh` and follow the Firefox steps above.

---

**Last resort — switch browsers back and forth:**

YouTube's cookie rotation is sometimes session-specific. If both Firefox and Chrome fail independently, try switching browsers once:

1. Try Chrome: `BROWSER=chrome just download-videos`
2. If that fails, try Firefox: `BROWSER=firefox just download-videos`
3. If that also fails, go back to Chrome: `BROWSER=chrome just download-videos`

The act of switching forces yt-dlp to read a completely different cookie store, which can bypass a rotation that was triggered by the previous extraction attempt.

---

**Last resort — export cookies to a file:**

If all browser extraction methods fail:

1. Install the "Get cookies.txt LOCALLY" extension in Firefox or Chrome.
2. Visit youtube.com while logged in.
3. Click the extension → export `youtube.com` cookies → save as e.g. `~/.youtube-cookies.txt`.
4. Pass it directly to yt-dlp: set `--cookies ~/.youtube-cookies.txt` in `yt-downloader.sh` (remove `--cookies-from-browser`).

Reference: https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies

---

## Playbook: Pipeline Stalls at LLM Step

1. `just status` — verify LM Studio + required models loaded.
2. Check `reports/` for most recent stage output.

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
