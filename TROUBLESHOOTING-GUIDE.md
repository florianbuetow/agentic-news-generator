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
| Find unmerged format-code artifacts | `just find-partial [<CHANNEL>]` |
| Find macOS AppleDouble sidecars | `find <data_dir> -name '._*'` |
| Files stuck in a channel, no errors | See **Playbook: Files Stuck in a Channel** |
| `analyze-transcript-languages` exits 1 | See **Playbook: Language Analysis Exits 1** |

---

## Finding Files

### `scripts/find-files.sh` — `just find-files <VIDEO_ID>`
Scan all data directories in `config.yaml` for filenames containing the video ID substring. Shortest-unique-prefix dedupe avoids duplicate hits from nested paths.

### `scripts/find-empty-transcripts.sh` — `just find-empty-transcripts`
List transcript `*.txt` files ≤100 bytes under `data_downloads_transcripts_dir`. Grouped by channel.

### `scripts/find-partial-downloads.sh` — `just find-partial [<CHANNEL>]`
Read-only scan of every data directory in `config.yaml` for format-code artifacts (`*.f[0-9]*.*`) — the unmerged single streams yt-dlp leaves behind when an ffmpeg merge never completes, plus everything derived from them (`.f251-8.wav`, `.f251-11.silence_map.json`). Without a channel argument it scans all channels.

Results are grouped by video ID, since that is the unit the cleanup playbook operates on. Each group states whether a properly merged `.mp4` exists for that ID, which decides the remedy:

- **merged `.mp4` present** → the merge did complete; the format-code files are stale leftovers to delete along with their derived files.
- **merged `.mp4` MISSING** → the download never completed; the ID also needs removing from `downloaded.txt` and re-downloading.

AppleDouble sidecars (`._*`) are counted but not listed — they are macOS metadata shadows of the real artifacts, not downloads. Exits `1` if any artifacts are found, else `0`. See **Playbook: Interrupted Merge**.

Two guards exist because an empty result from this tool authorises deletions:

- **Unknown channel is an error, not a clean scan.** A channel name matching no directory under any configured data path exits `1` with `Error: no directory named …`. Channels live below the per-stage directories (`downloads/videos/<CHANNEL>`, `downloads/metadata/<CHANNEL>/video`, `archive/videos/<CHANNEL>`), never directly under the data root, so the scan matches the channel as a path component wherever it appears.
- **A running `yt-dlp` triggers a banner.** yt-dlp writes these exact artifacts while downloading and only merges at the end, so an in-flight download is indistinguishable from an abandoned one by filename alone — and the remedy for the abandoned case would destroy the live one. When a download is running the scan is only a snapshot: artifacts may still be in flight and more may appear. Wait for it to exit, then re-run.

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

### `scripts/check-pipeline-integrity.py` — `just check-pipeline-integrity`
Read-only two-stage audit. Stage 1 finds files whose video ID is listed in `config/filefilter.json`, writes their absolute paths to `reports/files-whose-video-id-is-listed-in-filefilter.txt`, and exits `1` before stage 2 when any are found. Stage 2 runs only after stage 1 passes; it finds `downloaded.txt` video IDs with no active/archived video, WAV, or transcript and writes the absolute archive path plus video ID to `reports/downloaded-video-ids-without-video-audio-or-transcript.txt`. The terminal prints only counts and report paths. No CLI args.

### `scripts/clean-filtered-video-artifacts.py` — `just clean-filtered-video-artifacts [--apply]`
Safe cleanup companion to the integrity check. With no argument it recomputes all files belonging to filtered video IDs, refreshes `reports/files-whose-video-id-is-listed-in-filefilter.txt`, and deletes nothing. `--apply` deletes only those freshly recomputed targets after validating that each is a regular non-symlink file inside a configured pipeline directory and still contains a currently filtered ID. It preserves `downloaded.txt`; filtered IDs are excluded from pipeline-integrity stage 2 so intentional exclusions do not become false missing-artifact failures.

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

## Playbook: Language Analysis Exits 1 — Non-English Files Detected

Symptom: `just analyze-transcript-languages` prints one `[WARNING]` line per file and the recipe fails:

```
[WARNING] |   [OpenAI] [fr] /…/downloads/transcripts/OpenAI/Réinventer l'expérience beauté grâce à l'IA [sLYXRA5Ay9g].txt
[WARNING] | Total non-English files: 11
error: Recipe `analyze-transcript-languages` failed on line 462 with exit code 1
```

**The exit code is a designed gate, not a crash.** `scripts/transcript-language-analysis.py:368-370` returns `1` whenever the run collected at least one file that failed the English check in a channel configured `language: en`. Nothing errored; the script is reporting a verdict.

**Never act on the warning list alone.** The warning states a *language*, not a *defect*. Two unrelated conditions produce byte-identical warnings and need opposite remedies:

- a genuinely foreign-language upload on an English channel — the detection is correct, and the decision is editorial;
- a failed transcription whose text is not language at all — the detection is meaningless, and the file needs re-transcribing or filtering.

Telling them apart requires the per-file evidence table below. File size alone does not separate them: a 96-word file of one repeated token and a 20-word real fragment look the same in `ls -lh`.

### Step 1 — Build the evidence table

Report every flagged file with these columns. Anything less and the class cannot be decided:

| Column | Where it comes from |
|---|---|
| File | the path from the `[WARNING]` line |
| Words | `wc -w < "<file>"` |
| Chunks | 1 if Words < 1000, else `ceil(Words / 1000)` (`transcript-language-analysis.py:95`) |
| Per-chunk verdict | `lang @ confidence` for every chunk |
| English ratio | `english_count / chunks` (`transcript-language-analysis.py:65`) |
| Excerpt | first ~20 words of the actual file content |
| Class | **A** (failed transcription) or **B** (genuine foreign language) — see Step 2 |

Words and Excerpt come straight from the file. The per-chunk verdicts and ratio require running the production logic — drop this in `debug/` (disposable, gitignored) and run `uv run debug/lang_report.py`:

```python
#!/usr/bin/env python3
"""Per-chunk language verdicts for flagged transcripts. Mirrors transcript-language-analysis.py."""
from pathlib import Path

from src.config import Config
from src.nlp.language_detector import LanguageDetector

FLAGGED = [  # paste the paths from the [WARNING] lines
    "OpenAI/Réinventer l'expérience beauté grâce à l'IA [sLYXRA5Ay9g].txt",
]

config = Config.load_default()
min_confidence = config.get_language_analysis_min_confidence()
detector = LanguageDetector(model_path=config.get_data_models_dir() / "fasttext" / "lid.176.ftz")

def chunk_text(text: str, target: int = 1000) -> list[str]:
    """Faithful copy of transcript-language-analysis.py:77-115, including the
    last-chunk expansion — omitting it changes the verdict (see warning below)."""
    words = text.split()
    if not words:
        return []
    if len(words) < target:
        return [text.strip()]
    chunks = [c for i in range(0, len(words), target) if (c := " ".join(words[i : i + target])).strip()]
    if len(chunks) > 1 and len(chunks[-1].split()) < target:
        chunks[-1] = " ".join(words[-target:])  # expand tail back to a full window
    return chunks


for rel in FLAGGED:
    path = Path(config.get_data_downloads_transcripts_dir()) / rel
    content = path.read_text(encoding="utf-8", errors="replace")
    words = content.split()
    chunks = chunk_text(content)

    labels = []
    for chunk in chunks:
        r = detector.detect(chunk, k=1)
        lang, conf = (r.language, r.confidence) if not isinstance(r, list) else (r[0].language, r[0].confidence)
        # same downgrade rule as transcript-language-analysis.py:165
        labels.append((lang, conf, "en" if lang != "en" and conf < min_confidence else lang))

    english = sum(1 for _, _, final in labels if final in {"en", "??"})
    print(f"{path.name}\n  words={len(words)} chunks={len(chunks)} ratio={english / len(labels):.2f}")
    for i, (lang, conf, final) in enumerate(labels):
        print(f"    chunk[{i}] {lang} @ {conf:.4f} -> {final}")
    print(f"    excerpt: {' '.join(words[:20])}\n")
```

**Do not simplify the chunker.** The last-chunk expansion at `transcript-language-analysis.py:104-113` is load-bearing: a file's trailing partial chunk is replaced by the *last full 1000 words*, not left short. Dropping it changes verdicts. Measured on `Réinventer l'expérience beauté grâce à l'IA [sLYXRA5Ay9g].txt` (1186 words):

| Chunker | chunk[0] | chunk[1] | Ratio |
|---|---|---|---|
| With expansion (production) | fr @ 0.9882 | fr @ 0.9798 | **0.00** |
| Without expansion | fr @ 0.9882 | en @ 0.4992 | **0.50** |

The raw 186-word tail scores 0.4992, below the 0.7 floor, so the guard relabels it `en` and the file reads as half English — a different class from the one production assigns.

### Step 2 — Classify each row

Decide from the **Excerpt**, never from the language code:

| Excerpt shows | Class | Meaning |
|---|---|---|
| One token repeated (`FireCrawl FireCrawl …`, `web2.com web2.com …`), a bare URL fragment, `【Music】`, or scriptless gibberish (`инегет инегет …`, kana loops) | **A** | Failed transcription. The language verdict is noise — there is no language in the file. |
| Continuous, varied prose in a real language (`Bonjour à tous, mais c'est Lydia…`) | **B** | Genuine foreign-language upload. Detection is correct. |

A strong Class A tell: **Words is far too small for the video's duration.** Cross-check against the metadata — a 970-second video that yielded 96 words did not get transcribed:

```bash
jq '.duration' "<metadata_dir>/<CHANNEL>/video/<TITLE> [<VIDEO_ID>].info.json"
```

**Do not use confidence to classify.** Confidence measures how strongly the text matches a language's character n-grams, not whether the text is meaningful. Repeated tokens match *more* consistently than real speech, so Class A files routinely score **higher** than Class B ones (observed: a kana loop at 0.9292 against genuine Spanish at 0.9463).

### Step 3 — Remedy per class

- **Class A** — the transcript is the defect. Confirm the audio was not the problem before blaming the download: read the video's `silence_map.json` under `metadata/<CHANNEL>/audio/`. `total_silence_removed_seconds: 0` with `silence_intervals: []` means the audio was continuous and audible for the full duration, so the fault is in transcription, not the media. Then treat as a corrupt transcript per **Playbook: Empty Transcript for a Video**. Check for downstream contamination first — Class A text propagates into `transcripts_cleaned/`, `transcripts_summaries/` and `transcripts-topics/` unchallenged, and those derived files must go too.
- **Class B** — nothing is broken. Decide editorially: keep the channel English-only and filter the video, or accept mixed-language content for that channel. Re-downloading changes nothing; the upload really is in that language.

### Why the two safeguards do not catch Class A

Both guards in `analyze_file` are inert against short degenerate files, and neither can be tuned to fix it:

- **Confidence floor** (`min_detection_confidence`, `config/config.yaml`; applied at `transcript-language-analysis.py:165-167`). A non-English chunk below the floor is relabelled `en`. Class A chunks score 0.77–0.93, far above the 0.7 default. Raising the floor breaks Class B detection long before it catches Class A.
- **80% majority vote** (`is_english_only`, `transcript-language-analysis.py:60-65`). A file passes if ≥80% of chunks are English. But `chunk_text` returns a *single* chunk for any file under 1000 words (`transcript-language-analysis.py:95`), and Class A files are typically 1–100 words. With one chunk the ratio can only be `0.00` or `1.00` — there is no majority to take, and one bad detection condemns the file. The safeguard only functions on files ≥1000 words.

**Consequence for reporting:** always record the chunk count. A row with `chunks=1` had no safeguard applied at all, and its verdict rests entirely on a single FastText call.

### Worked example (2026-08-11 run, 11 files)

| File | Words | Chunks | Verdicts | Ratio | Excerpt | Class |
|---|---|---|---|---|---|---|
| `…Finishing Tasks On Time [X-I2mKcs49s]` | 3 | 1 | ja @ 0.7824 | 0.00 | `Cal's role 【Music】` | A |
| `…Shabbat Rituals… [cgbWIKwbdfY]` | 56 | 1 | ru @ 0.8952 | 0.00 | `стакой онакантась срегданий инегет инегет…` | A |
| `…Build AI Podcasts… [ievgM928RBc]` | 96 | 1 | ru @ 0.8418 | 0.00 | `FireCrawl FireCrawl FireCrawl…` | A |
| `…Gary Vaynerchuk's… [gytWTM6ZY9M]` | 61 | 1 | pt @ 0.7715 | 0.00 | `web2.com web2.com web2.com…` | A |
| `…MUFG aims to become AI-native… [pE1ljLVY1Rg]` | 6 | 1 | pl @ 0.9292 | 0.00 | `保つべつりまけちちねねつねつねつ…` | A |
| `…Réinventer l'expérience beauté… [sLYXRA5Ay9g]` | 1186 | 2 | fr @ 0.9882, fr @ 0.9798 | 0.00 | `Applause On a l'oïlité client…` | B |
| `…Verso, l'entreprise qui ne dort jamais [mFwWax5pLTs]` | 1236 | 2 | fr @ 0.9937, fr @ 0.9962 | 0.00 | `Bonjour à tous, mais c'est Lydia…` | B |
| `…Demo en Español [-voWZLuFAZk]` | 2212 | 3 | es @ 0.9463, 0.9519, 0.9470 | 0.00 | `hola, soy Eric Desoner y en este vídeo…` | B |
| `…Demo en Español [uqz-tEZIrOQ]` | 2287 | 3 | es @ 0.9573, 0.9483, 0.9490 | 0.00 | `Hola, soy Eric Desoner y en este video…` | B |
| `…AI is Making Us Dumber… [yZqbLNi9fhQ]` | 1 | 1 | de @ 0.7809 | 0.00 | `ers.com.au` | A |
| `…How Decoder-Only Transformers… [baykhvFS_e4]` | 20 | 1 | el @ 0.8838 | 0.00 | `www.kirillestem.com kirillestem.com…` | A |

Seven Class A, four Class B, one exit code. Every ratio was `0.00` — no file had even one English chunk — and the lowest confidence across all 20 chunks was 0.7715, so the confidence floor rescued nothing.

**General rule:** content sanity (empty, missing, degenerate repetition, word count versus video duration) is the **first** thing to check for any transcript quality issue, before deeper analysis.

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

**Step 0 — Is `just download-videos` still running?** If the reported filename carries a format code (`… [VIDEO_ID].f137.mp4`), this is the most likely cause and nothing is wrong with the data. yt-dlp writes the video-only stream first and merges last, so a concurrent `extract-audio` reads a file that is still being assembled and correctly reports it has no audio.

```bash
ps -Ao lstart,command | grep '[y]t-dlp'
```

Any hit → let the download finish, then re-run `just extract-audio`; the merged `.mp4` will have both streams. Delete nothing, and do not touch `downloaded.txt`. Confirm afterwards with the ffprobe in Step 1: `h264,video` **and** an audio line means it resolved itself. The two stages are not safe to run concurrently against the same channel.

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

**How to identify** — `just find-partial` sweeps every channel for these artifacts without needing a video ID up front, and reports for each ID whether the merged `.mp4` exists. Use it when you do not yet know which videos are affected; use `just find-files <VIDEO_ID>` when you do. Either way you see several files in `downloads/videos/<CHANNEL>/` where there should be exactly one `.mp4`:

| Artifact | Contents |
|---|---|
| `[…].f137.mp4` | video-only (h264) — the source of the "no audio stream" error |
| `[…].f251.webm` | audio-only (opus) |
| `[…].temp.mp4` | partial merge output — `ffprobe` reports `moov atom not found` |

Confirm with `ffprobe -v error -show_entries stream=codec_type,codec_name -of csv=p=0 <file>` on each.

**The trap:** `extract-audio` succeeds on the audio-only `.webm` and writes `[…].f251.wav` — a *complete, valid* WAV carrying a format-code suffix. It looks like a usable result and tempts you into keeping it. Do not. It is derived from the abandoned download, and keeping it prevents re-extraction from the clean merge (see the derived-WAV rule under **Playbook: Removing Download Artifacts**). It will also break metadata lookup — see **Case 2** of the "Metadata file not found" playbook.

**First, rule out a download that is still running.** A live `yt-dlp` produces artifacts identical to this playbook's symptoms — including `extract-audio` failing with "No audio stream found" on a `.f137.mp4` that merges cleanly minutes later. Running the steps below mid-download deletes work in progress:

```bash
ps -Ao lstart,command | grep '[y]t-dlp'
```

Any hit → stop. Wait for it to exit, then re-scan; the artifact is very likely transient. `just find-partial` prints a banner in this case, but check directly when working from `just find-files` instead.

**Steps:**

1. `just find-partial <CHANNEL>` (or `just find-files <VIDEO_ID>`) — enumerate every artifact.
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

## Playbook: Files Stuck in a Channel — Nothing Errors, Nothing Progresses

Symptom: a channel's `downloads/videos/<CHANNEL>/`, `downloads/audio/<CHANNEL>/` or `archive/videos/<CHANNEL>/` holds files that survive run after run. No recipe fails. `just stats` looks fine. The files simply never become transcripts.

Stuck files are the residue of a stage that **failed without raising**. Three mechanisms produce them, and they need opposite remedies — so classify before deleting anything. The cost of guessing is asymmetric: the data volume has no backup, and a wrong deletion of a good WAV is unrecoverable if the disk is too full to re-download (see step 2).

### Step 0 — Rule out a live download

yt-dlp produces artifacts indistinguishable from abandoned ones. Everything below is invalid while it runs:

```bash
ps -Ao lstart,command | grep '[y]t-dlp'
```

Any hit → stop, wait, re-scan.

### Step 1 — Inventory the channel across all stage dirs

```bash
ls -la <videos_dir>/<CHANNEL>/ <audio_dir>/<CHANNEL>/ <archive_videos_dir>/<CHANNEL>/
```

Then, for each suspicious video ID, get the full cross-stage picture — this is the single most informative command in this playbook:

```bash
just find-files <VIDEO_ID>
```

Read it for two things: (a) which stage the ID stops at, and (b) whether a **format-code twin** (`.f251`, `.f251-8`, `.f399`) exists beside a clean file. A twin means two parallel artifact sets for one video.

### Step 2 — Check free disk before forming any theory

```bash
df -h <data_dir>
```

This governs both diagnosis and remedy. `MIN_FREE_DISK_GB = 20` in `scripts/yt-downloader.py`; `just download-videos` refuses to start below it. **If free space is under that threshold, deleting a good MP4/WAV strands the video** — it cannot be re-fetched until space is freed. A full disk is also a prime cause of the silent failure in Case A.

### Step 3 — Classify each stuck ID

| Evidence | Case |
|---|---|
| ID present in `config/filefilter.json` | **A** — condemned as "no speech" |
| Format-code twin alongside a clean set | **B** — interrupted-merge duplicate |
| `.info.json` in `videos/` but absent from `metadata/<CHANNEL>/video/` | **C** — metadata never landed |

---

### Case A — Silently condemned as "no speech"

`scripts/transcribe_audio.py` treats a **0-byte SRT** from Whisper as "no speech detected": it appends to `logs/empty_transcripts.log`, calls `add_no_speech_to_filefilter` to write the ID into `config/filefilter.json`, and **returns success**. Nothing reaches `error.log`. The video is now queued for permanent deletion by `remove-filtered-files.py`, and stays on disk until `just filter-videos` next runs.

A 0-byte SRT means Whisper produced nothing — **not** that the audio is silent. A full disk, an unloaded model, or a crashed run all yield the same empty file, and all get recorded as a content verdict.

**Confirm the overlap** (the two files should be read together — `filefilter.json` says *condemned*, the log says *why*):

```bash
jq -r '.. | strings | select(test("<CHANNEL>"))' config/filefilter.json
grep '<CHANNEL>/' <logs_dir>/empty_transcripts.log
```

**Never trust the verdict — verify the audio directly.** Probe the WAV, not the MP4:

```bash
ffprobe -v error -show_entries stream=codec_name,sample_rate,channels -of default=nw=1 "<audio_dir>/<CHANNEL>/<TITLE> [<VIDEO_ID>].wav"
ffmpeg -hide_banner -nostats -i "<audio_dir>/<CHANNEL>/<TITLE> [<VIDEO_ID>].wav" -af volumedetect -f null - 2>&1 | grep -E 'mean_volume|max_volume'
```

- `pcm_s16le` / `16000` / `1 channel` with `mean_volume` around **−25 to −35 dB** → **audible speech. The verdict is a false positive.**
- `mean_volume` at or below **−91 dB** → genuinely silent; the verdict stands.

Cross-check duration against the threshold (`transcription.min_duration`, 90 s) using the metadata rather than the filter's say-so:

```bash
jq '.duration' "<metadata_dir>/<CHANNEL>/video/<TITLE> [<VIDEO_ID>].info.json"
```

**Remedy for a false positive — repair, do not delete.** The WAV is already correct; only the verdict is wrong:

1. Remove the offending `<CHANNEL>/<VIDEO_ID>` entries from `config/filefilter.json`. Leave `downloaded.txt` alone — the media is on disk and must not be re-fetched.
2. Re-run `just transcribe`. It costs zero bytes of download and works regardless of free space.

Do **not** run `just filter-videos` before doing this: step 2 of that recipe (`remove-filtered-files.py`) will sweep the still-listed files off disk, along with their upstream MP4s.

**Confirming the historical cause.** `error.log` is retained far longer than `app.log` (which rotates). Correlate the `mtime` of `config/filefilter.json` with the log:

```bash
ls -la config/filefilter.json
grep -n 'disk space' <logs_dir>/error.log
grep -nE '<VIDEO_ID_1>|<VIDEO_ID_2>' <logs_dir>/error.log
```

A `🚨 Less than 20 GB disk space remaining (0 MB available)` entry near the filter's write time is the signature of Whisper being starved of space. Note that the condemning run itself leaves **no** error line — only the earlier, noisier failures for the same IDs appear.

---

### Case B — Interrupted-merge duplicates in `archive/`

An audio-only stream (`[…].f251.webm`) that was transcribed before the merge was repaired gets archived by `just archive-videos` as though it were a video. The re-download then produces a **second, clean** set. The corpus now counts the video twice — duplicate transcripts, cleaned transcripts and summaries — which skews every downstream aggregate.

```bash
just find-partial <CHANNEL>
```

Groups by video ID and lists every derived artifact, including ones easy to miss by hand (`transcripts-hallucinations/`, `metadata/<CHANNEL>/transcript/`). Exits `1` when anything is found.

**Before deleting, prove the clean twin is complete** — the format-code set is only redundant if a non-suffixed `.srt`, `.txt`, cleaned transcript *and* summary all exist. `just find-files <VIDEO_ID>` shows both sets side by side.

**Remedy:** delete the `.webm` and every format-code-suffixed derivative, and **keep the ID in `downloaded.txt`**. The clean transcripts already exist; removing the archive entry only triggers a pointless re-download of content that is fully processed.

`find-partial` reporting `merged .mp4 MISSING` is **not** by itself grounds for re-download. If clean transcripts and a summary exist, the pipeline completed and the MP4 was simply reclaimed. Judge by the derived text artifacts, not by the absence of the video.

**Variant — the orphan is the only copy.** Case B above assumes the repair already happened and left a clean twin behind. The interrupted download can also be archived and never repaired: `archive/videos/<CHANNEL>/` then holds `[…].fNNN.webm` and **no `.mp4` at all**, and the only transcripts are format-code-suffixed. The remedy inverts — re-download *is* warranted, and the archive entry must go. Classify before touching anything:

| Signal | Case B — clean twin exists | Variant — orphan is the only copy |
|---|---|---|
| `archive/videos/<CHANNEL>/` | merged `.mp4` alongside the `.webm` | `[…].fNNN.webm` only |
| Transcripts | both `[…].srt` and `[…].fNNN.srt` | `[…].fNNN.srt` only |
| Remedy | delete the orphan set, **keep** the archive entry | re-download, **remove** the archive entry |

Confirm the orphan is genuinely audio-only first — one `opus` stream, no video stream:

```bash
ffprobe -v error -show_entries stream=codec_type,codec_name -of csv=p=0 <file>
```

**Archiving does not imply the ID was recorded.** `downloaded.txt` is inconsistent across such IDs — in one observed channel only one of five archived orphans was present, and the other four re-downloaded with no archive edit at all. Check per ID instead of assuming:

```bash
grep -c <VIDEO_ID> <videos_dir>/<CHANNEL>/downloaded.txt
```

**Re-downloading specific IDs.** `just download-videos <CHANNEL>` re-scans the whole channel. To refetch individual videos with identical flags, cookies and archive bookkeeping, call the downloader directly — the single-video form documented in `scripts/yt-downloader.sh`:

```bash
scripts/yt-downloader.sh 'https://www.youtube.com/watch?v=<VIDEO_ID>' '<videos_dir>/<CHANNEL>' '1'
```

Any `[…].fNNN.mp4.part` still in `videos/<CHANNEL>/` is consumed as a resume point and deleted once the merge succeeds. Verify both streams landed before trusting the result, then run `scripts/move-metadata.sh <CHANNEL>` — `just download-videos` runs it after the download, so skipping it leaves `.info.json` and `.webp` stranded in the videos dir. **Judge that move by the filesystem, not by its summary line:** a run that moved five `.info.json` and five `.webp` reported `0 metadata file(s) moved, 0 thumbnail(s) moved`.

Delete the superseded `[…].fNNN.*` transcripts only **after** the replacements exist. Until `extract-audio` and `transcribe` have re-run, they are the video's only coverage, and removing them early opens a gap in the corpus.

---

### Case C — Metadata never landed in `metadata/<CHANNEL>/video/`

`just transcribe` aborts per-file with `Metadata file not found` and the video never advances. Unlike Case A this leaves no `filefilter.json` entry and no `empty_transcripts.log` line — the ID is invisible to every audit and retries identically forever.

```bash
just find-files <VIDEO_ID>
```

`.info.json` sitting in `downloads/videos/<CHANNEL>/` with nothing under `metadata/<CHANNEL>/video/` confirms it. A missing `.silence_map.json` under `metadata/<CHANNEL>/audio/` is corroborating evidence that the file never got past extraction.

**Remedy** — see **Case 1** of "Transcription Fails With Metadata file not found":

```bash
just fetch-video-metadata <CHANNEL> <VIDEO_ID>
```

Then re-run `just transcribe`. Delete nothing; the MP4 and WAV are sound.

---

### Why these hide so well

Each mechanism converts a failure into something that reads as a normal outcome — a content verdict (A), a successful archive (B), or a per-file skip (C). None sets a non-zero exit code, so `just` reports success and the next stage finds nothing to do. Treat "a file that is still here after two full runs" as the primary signal; the logs will not raise it for you.

**Stray files that are not stuck.** A lone `.webp`, or a channel-level `<CHANNEL> - Videos [UC…].info.json`/`.jpg` (a `UC…` playlist ID, not an 11-char video ID) left in `videos/<CHANNEL>/`, is yt-dlp residue with no pipeline stage waiting on it. Harmless; do not treat it as a stuck video.

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

**Fixed in code (commit `6a099b5`, 2026-06-10).** `find_metadata_file` now skips dotfiles in its glob fallback, so this failure can no longer be triggered by AppleDouble sidecars. The mechanism below is retained because the error text still appears in older logs, and because the same class of bug can return if the filter is ever dropped. If you hit this error on current code, the cause is something else — do not assume sidecars.

Why the wrong file used to get picked (`scripts/transcribe_audio.py:119-136`):

- `find_metadata_file` first tries the exact stem `<base_name>.info.json`. For a normal WAV this hits the real file and the bug never triggers.
- When the WAV carries a **format-code suffix** (e.g. `… [VIDEO_ID].f251-11.wav` — see **Case 2** of the previous playbook), the exact stem `… .f251-11.info.json` does not exist, so the code falls back to `sorted(metadata_video_dir.glob("*.info.json"))` and returns the first file whose bracketed video ID matches.
- On the external data drive (a non-APFS/HFS+ filesystem — exFAT/SMB/etc.), macOS writes an AppleDouble `._<name>.info.json` sidecar next to every real file. The glob returns **both**, and `sorted()` ranks `._…` first because `.` (`0x2E`) sorts before `A` (`0x41`). The script therefore opens the binary sidecar.

Two conditions had to hold together: a format-code-suffixed WAV **and** AppleDouble sidecars on disk. WAV discovery already filtered these out (`scripts/transcribe_audio.py:401,479` — `if not f.name.startswith(".")`); the metadata glob fallback lacked the same filter until `6a099b5` added it at `scripts/transcribe_audio.py:133`.

**Cleaning up the sidecars** (they carry no real content, only resource-fork/xattr metadata):

1. Dry run — list what would be removed under the metadata dir (narrow to one channel with `/<CHANNEL>`):
   ```bash
   find <metadata_dir> -name '._*' -print
   ```
2. Delete them:
   ```bash
   find <metadata_dir> -name '._*' -delete
   ```
3. Re-run `just transcribe`.

**Sidecars still recur** — macOS recreates `._` files whenever it touches that filesystem (Finder copy, Spotlight indexing, etc.), so deletion remains a band-aid. That no longer matters for transcription, which now ignores them, but they still add noise to any directory listing or glob written without a dotfile filter.

The underlying `.f251-11` format-code artifact is itself a broken-download symptom — see **Case 2** above for cleaning it up and re-downloading.

---

## Playbook: Metadata `timestamp` Is Missing (Publication Timeseries Fails)

Symptom: `just publication-timeseries` reports one line per offending file and exits non-zero:

```
❌ Metadata 'timestamp' is missing or not a number: <…>/metadata/<CHANNEL>/video/<TITLE> [VIDEO_ID].info.json
```

The check lives in `src/analytics/publication_timeseries.py` (`_require_number`, called from `read_publication_record`). The timeseries derives each video's publication instant from the `.info.json` **`timestamp`** field (epoch seconds, which carries a time of day). It deliberately does **not** fall back to `upload_date`: that field is date-only and would collapse every video published on a given day onto the same instant. So a file whose `timestamp` is `null` fails the check even though `upload_date` is present.

**Root cause:** yt-dlp wrote `"timestamp": null`. Older yt-dlp versions (and some extraction paths) populate only the date-only `upload_date` for a regular video and leave `timestamp` null. A newer yt-dlp usually extracts the full epoch timestamp for the same video, so re-fetching the metadata fixes it. `live_status: not_live` / `was_live: false` in the file confirms this is an ordinary upload, not a premiere/livestream (whose publication time can legitimately be modelled differently).

**How to identify** — inspect the flagged file (read-only):

```bash
jq '{timestamp, upload_date, release_timestamp, live_status, ytdlp: ._version.version}' \
  '<…>/metadata/<CHANNEL>/video/<TITLE> [VIDEO_ID].info.json'
```

`timestamp: null` with a non-null `upload_date` confirms this playbook. Compare the file's `._version.version` against the installed `yt-dlp --version`; a newer installed version is a strong sign a re-fetch will populate `timestamp`.

**Before deleting anything, confirm a fresh fetch actually returns a timestamp.** YouTube does not expose a precise time for every video — if it doesn't, deleting and re-fetching only loses the file. This probe writes nothing (`firefox` = the browser set in `scripts/config.sh`):

```bash
yt-dlp --skip-download --no-warnings --cookies-from-browser firefox \
  --print "timestamp=%(timestamp)s upload_date=%(upload_date)s" \
  "https://www.youtube.com/watch?v=<VIDEO_ID>"
```

A numeric `timestamp=` (not `NA`) means the re-fetch will fix the file. `NA` means it won't — stop and decide separately (keep the file, or teach the analytics to accept a date-only fallback).

**Fix:**

1. Delete the stale `.info.json` — one literal `rm` per file.
2. Re-fetch the metadata:
   ```bash
   just fetch-video-metadata <CHANNEL> <VIDEO_ID>
   ```
   This resolves the video's stem from whichever artifact survives — the audio WAV for an active video, or the archived MP4 / transcript for an **archived** one — so it works whether or not the WAV is still on disk. (`just check-missing-metadata` also re-fetches, but only for videos whose WAV still exists — see the trap below.)
3. Validate:
   ```bash
   jq '.timestamp' '<…>/metadata/<CHANNEL>/video/<TITLE> [VIDEO_ID].info.json'
   ```
   A number (not `null`) means the `just publication-timeseries` check will now pass.

**The `check-missing-metadata` trap:** `just check-missing-metadata` discovers work only by matching an existing **WAV** on disk (`scripts/check-missing-metadata.py`). For an archived video the WAV is gone, so it reports "all metadata present" and re-fetches **nothing** — a silent no-op, not an error. That is why this playbook uses `just fetch-video-metadata` instead: its stem lookup also searches the archived video and the transcript (`src/util/media_stem.py`), so it restores metadata for archived videos too. `scripts/fetch-video-metadata.sh '<…>/video/<TITLE> [VIDEO_ID]' <VIDEO_ID>` remains the low-level escape hatch — it takes the exact output stem with no extension.

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
3. Inspect that transcript: `ls -lh <path>` for size, then read the first ~20 lines. If it shows hallucination patterns (repeated tokens, single phrase repeated) → treat as corrupt transcript and follow **Playbook: Empty Transcript for a Video**. See **Playbook: Language Analysis Exits 1** for the same content-sanity checks.
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
