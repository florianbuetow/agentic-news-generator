# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## 2026-06

### 2026-06-07

#### Added

- The clean-content pipeline now detects documents whose prompt exceeds the model output window (or the context-window threshold) and records them in `data_cleaned/uncleanable.json`, deleting any partial output and skipping the file on later runs until the output window grows; `LiteLlmClient` raises `OutputWindowExceededError` on a `length` finish reason.
- URL clean-content progress now shows a per-item completion percentage and running-average ETA (mirroring `summarize-transcripts`).
- Added a Semgrep rule (`justfile-no-error-suppression.yml`) banning just's error-ignoring recipe-line prefixes (`-`, `-@`, `@-`), wired into the `code-semgrep` scan target.

#### Changed

- Data pipelines (`url-all`, `video-all`, `pipelines-all`) now continue past a failed step and print a red/green per-step summary, exiting non-zero if any step failed; `ci`, `ci-verbose`, and `checks-all` remain fail-fast.
- The URL download pipeline now pre-filters already-downloaded URLs (non-empty raw file on disk) before the loop, reports how many were skipped, and counts only pending URLs in the `[n/total]` progress.
- Raindrop fetcher now dedupes bookmarks against a persisted seen-set (`raindrop-fetched-urls.json`), emitting only URLs not seen in a prior run; new `--force` re-emits every bookmark.
- Reduced `url_clean_content` max retries from 3 to 1.

#### Removed

- Archived two duplicate transcript files (without YouTube ID) that were superseded by a properly-versioned counterpart already on disk. Preserved in the external archive.

#### Security

- Upgraded `pip` to `26.1.2` to patch PYSEC-2026-196.

---

### 2026-06-06

#### Added

- `urls-requeue-unprocessed` now skips URLs that are definitively gone (DNS-unresolvable or HTTP 404/410) using a curl reachability pre-filter, so permanently dead URLs are no longer re-queued every run (new `unreachable_count`).
- Added a durable cleaning processing-error log at `data/urls/data_cleaned/errors/<YYYY-MM-DD>.txt`: raw files that fail, produce empty/invalid output, or are oversized during clean-content processing are appended (one line each) for human review.
- Added `find-files-without-youtube-id`: a read-only finder that reports data files lacking a bracketed YouTube ID with no ID-bearing sibling in the same folder, grouped by folder category (per-channel `downloaded.txt` archives excluded).

#### Changed

- HTML render wait strategy changed from `load` to `domcontentloaded`, so otherwise-reachable pages no longer time out waiting on images and trackers.

#### Removed

- Removed all topic-processing code: agent-critic topic segmentation, the LLM topic-experiment script, Codex-driven topic extraction, the `topic_detection` module, and their config models, tests, and LM Studio status checks. The code is preserved in the external archive.

#### Fixed

- HTML downloader now falls back to a direct HTTP fetch when the headless renderer is blocked, errors, or returns empty/blocked content, recovering reachable pages that fail browser rendering (e.g. bot-detection 403s).
- HTML-classified URLs that actually serve PDF content (e.g. extensionless `/pdf?id=` links) are re-routed to the PDF downloader instead of failing.
- Download `unprocessed` archive entries are now written as a single line; multi-line Playwright error text previously polluted the line-oriented log and was mis-read as separate entries by the requeue scan.

---

## 2026-05

### 2026-05-31

#### Added

- Added URL ingestion pipeline: queue reader, normalizer, classifier, type-routed downloaders, raw-content processing, and clean-content Markdown formatter.
- Added Raindrop.io integration: fetcher reads token via `Config` (`integrations.raindrop_io.token`) and writes categorized bookmark URLs into the configured URL inbox under `data_dir`.
- Inbox queue reader now extracts real URLs from category-prefixed lines (`Category->Subcategory:url`) and rejects lines with no `http` URL as unprocessable.
- Added `pipelines-all` and `checks-all` justfile targets grouping pipeline and check steps.
- Added `Data-Import` justfile section with `urls-download` and `urls-cleancontent` targets.

#### Changed

- Nested LLM settings for `agentic_unit_test_reviews` and `agentic_shell_script_reviews` under `.llm` key (consistent with `summarize_transcripts` and `url_clean_content`). **BREAKING:** existing configs must move `model`/`base_url`/`api_key` under `llm:` block.
- Extended `just status` to cover all LM Studio endpoints in a unified table (Config Key | Model | Available | Loaded); normalizes `localhost` and `127.0.0.1` to the same server group; adds `url_clean_content.llm` to the check.
- Replaced topic-detection justfile targets (`topics-all`, `extract-topics`) with URL pipeline targets; renamed help section from "Topic Detection" to "Data-Export".

#### Removed

- Removed `all-quiet` justfile target (superseded by `pipelines-all`).

---

### 2026-05-30

#### Added

- Added between-channel disk-space guard to `yt-downloader.py`: stops downloading if free space drops below 20 GB before starting each channel batch (threshold clears the largest single observed video at ~10.8 GB).
- Added `ARCHIVE-GUIDE.md` documenting procedures for archiving stale code and its data artifacts.
- Added optional transcription limit parameter to the `transcribe` justfile target.

#### Changed

- Renamed `TROUBLESHOOTING.md` to `TROUBLESHOOTING-GUIDE.md`; expanded YouTube cookie playbook with browser-specific instructions (Firefox SQLite-based vs Chrome encrypted session), troubleshooting steps, and last-resort approaches; switched default browser config from Chrome to Firefox.
- Updated README to reflect current pipeline state.

#### Removed

- Removed dead topic boundary detection implementation (script, prompt, config, build targets) and archived via `ARCHIVE-GUIDE.md`.
- Removed stale `topics_experiment`, `topic_segmentation`, and `topics` sections from config template.

#### Fixed

- `yt-downloader.py` now exits immediately on download failure so the error message stays as the last visible output; previously, post-processing steps (move-metadata, members-only skip list) pushed the failure message up the scrollback.

---

### 2026-05-28

#### Added

- `yt-downloader.py` now pulls the best available thumbnail alongside each `.info.json`.
- `move-metadata.sh` now relocates sibling thumbnails (`.jpg`/`.webp`/`.png`) into the metadata directory alongside their `.info.json`.
- Added `just fetch-video-thumbnails` target to backfill missing thumbnails across all channels, a single channel, or specific video IDs; collects per-video failures and exits 1 on any failure.

---

### 2026-05-26

#### Fixed

- Added `--` argument separator to yt-dlp invocations so that video IDs beginning with `-` are not misinterpreted as yt-dlp options.

---

### 2026-05-24

#### Added

- `just status` now checks LM Studio models for `summarize_transcripts` and `topic_boundaries` in a single unified table (Config Key | Model | Available | Loaded).
- `just summarize-transcripts` now accepts an optional channel argument to restrict summarization to one channel (`CHANNEL_FILTER` env var internally).
- Added Semgrep rules for env var usage detection, early loop exit detection, and non-`snake_case` naming enforcement in `src/` and `scripts/`.
- Extended Ruff banned-api rules to ban `os.environ`/`os.getenv`; messages now point directly to `CONVENTIONS.md`.

#### Changed

- Config accessors renamed to `snake_case` throughout `src/config.py` and all call sites (required for Semgrep naming rule compliance).

#### Fixed

- Partial transcription reruns now heal correctly: both `.txt` and `.srt` outputs must be present before a file is considered complete; video metadata lookup now resolves by YouTube ID when audio stems include format suffixes (e.g. `.f401`).

---

### 2026-05-23

#### Added

- `fetch-video-metadata` now has a scan mode: when invoked with no arguments it auto-discovers all non-archived video files missing metadata and batch-fetches them without requiring an explicit video ID list.

---

### 2026-05-22

#### Added

- Transcript summarization now measures token count with tiktoken before issuing LLM requests and skips oversized inputs as warnings; threshold configured via required `skip_transcripts_above_context_window_pct` field (no implicit default allowed).

#### Security

- Pinned `idna>=3.15` to resolve GHSA-65pc-fj4g-8rjx (DoS via `idna.encode`).
- Suppressed `torch` PYSEC-2026-139 with documented justification (local-only deserialization; torch is never imported at runtime — `torch_whisper.py` is dead code).

---

### 2026-05-21

#### Added

- Added troubleshooting playbook for `extract-audio` no-audio-stream errors to `TROUBLESHOOTING-GUIDE.md`.
- Added maintenance recipe for read-only data pipeline health checks to justfile.

---

### 2026-05-19

#### Changed

- `summarize-transcripts` context window threshold made fully configurable via `skip_transcripts_above_context_window_pct` in `SummarizeTranscriptsConfig`; token count is now measured against the transcript text (not the assembled prompt) for more accurate gating; field is required in config to prevent hidden defaults.

---

### 2026-05-18

#### Added

- Added `check-missing-metadata` script: scans all channels for WAV files whose `.info.json` metadata is absent, reports per-channel counts, then batch-fetches all missing files via `fetch-video-metadata.py`; yt-dlp output suppressed, only success/failure shown.

#### Fixed

- `summarize-transcripts` now re-raises the original exception after max retries instead of swallowing it, ensuring Python exits with code 1 and a traceback on persistent failures.
- `TopicBoundaryResponse.validate_root` docstring added; wrapped lines collapsed; `split_sentences` simplified to treat each non-empty TXT line as one sentence.

---

### 2026-05-17

#### Added

- Added topic boundary detection: `detect-topic-boundaries.py` script, `detect-topic-boundaries-from-txt` and `extract-topic-boundaries` prompts, and supporting config block.

#### Changed

- `summarize-transcripts` now processes channels by fewest pending files first (mirrors ordering in `transcribe_audio.py`).
- fzf search results are now colorized with query match highlighting in the preview window.

---

### 2026-05-16

#### Added

- Added `just find`: interactive transcript search via fzf with bat syntax-highlighted preview window.
- Added `just search`: search transcript summaries with fzf preview.
- Added optional transcript time-totals view: `SHOW_TIME`-gated Time column and `just totals` target; long totals runs now show live per-channel progress.
- Added `SKIP_EXISTING` flag to hallucination detection and removal scripts — when set, scripts skip files whose output already exists and log only new activity.

#### Changed

- Updated `summarize-transcripts` model to `qwen3.6-35b-a3b` in config template.

---

### 2026-05-15

#### Changed

- Topic extraction updated to drive Codex exec directly instead of through a launcher script; adds JSON validation, progress tracking with ETA, and explicit error handling; config schema and supporting scripts updated accordingly.

---

### 2026-05-14

#### Added

- Added `extract-topics` pipeline: iterates hallucination-freed SRTs, invokes the Codex topic-boundary launcher, routes output through `/tmp` to stay within Codex's workspace-write sandbox, then moves results to `output_dir`.

#### Removed

- Removed the entire `topic_detection` module: embedding agents, sliding window segmenter, taxonomy mapper, ACM CCS 2012 loader, boundary detection scripts, visualizations, tests, benchmarks, and associated config sections.

---

### 2026-05-12

#### Changed

- Updated `uv` dependency lock file.

---

### 2026-05-11

#### Changed

- Updated troubleshooting documentation with improved guidance.

---

### 2026-05-09

#### Added

- `yt-downloader.py` now tracks and reports failed channels with per-channel failure reasons in the final summary.
- `filter-videos.py` now emits per-channel progress during the ffprobe scan every 100 files (mirrors integrity checker pattern; scan no longer looks hung on large channels).
- Added `.claudeignore` to exclude caches, `node_modules`, `uv.lock`, `.omx`, and ephemeral spec/plan files under `docs/superpowers/` from AI context.
- Extracted `CONVENTIONS.md` from `AGENTS.md`: coding conventions, git rules, and project standards now live in a dedicated file; `AGENTS.md` reduced from 141 to 35 lines.

#### Fixed

- Added `Field(min_length=2, max_length=5)` to `LLMTopicLabelData.topic_labels` as defense-in-depth alongside LLM JSON schema enforcement.
- Language analysis non-English detection now filtered to configured English-language channels only (prevents false positives on intentionally non-English channels).
- Bumped `gitpython` and `python-multipart` to address known vulnerabilities; removed outdated `--ignore-vuln` flags from pip-audit.

---

### 2026-05-08

#### Fixed

- `transcribe_audio.py` now filters hidden WAV files (any filename starting with `.`) at both scan sites, covering both macOS `._*` resource forks and in-progress `.{base}.temp.wav` temp files written by `convert_to_audio.sh`.

---

### 2026-05-06

#### Changed

- Language analysis extracted into a standalone pipeline task, fully decoupled from the transcription step.

#### Fixed

- Language detector now detects and excludes degenerate repetition patterns (e.g. Whisper hallucinations like `"Uiuiuiuiui..."`) from language detection via `_is_repetitive()` check in `filter_alpha_words()`.
- `stats` script: replaced `[[ -n "" ]] && ...` one-liner with `if`-block to prevent false non-zero exit code when the period argument is unset.

---

### 2026-05-05

#### Added

- Language analysis now supports a configurable minimum confidence threshold: low-confidence non-English detections (e.g. German proper nouns in English text) fall back to English, reducing false positives.

#### Security

- Pinned transitive dependencies to resolve known vulnerabilities: `gitpython>=3.1.47`, `jupyter-server>=2.18.0`, `jupyterlab>=4.5.7`, `notebook>=7.5.6`.

---

### 2026-05-04

#### Changed

- `yt-downloader.py`: consolidated all `--match-filter` flags into a single combined expression.

---

### 2026-05-01

#### Changed

- Added commit convention: author credits and source URLs are prohibited in commit messages (allowed only in guides, docs, and code comments).

---

## 2026-04

### 2026-04-30

#### Added

- Added `prompts/` and `scripts/` folders with transcript summarization prompt templates.
- Added `SummarizeTranscriptsConfig` Pydantic model and `data_downloads_transcripts_summaries_dir` path configuration.

#### Fixed

- `just stats` completion percentage now uses transcript count as the denominator (previously used video file count, which caused 50750%+ overflows on channels with archived videos); added `% OF TRANSCRIPTS` summary row showing each pipeline stage as a percentage of total transcripts.

---

### 2026-04-27

#### Added

- `just stats hour|day`: always displays current stats but only updates the JSON cache (diff baseline) once per the given period via `--no-update-cache`; status footer shows `Stats from <baseline time> — diff until now <current time>`.

---

### 2026-04-26

#### Security

- Bumped `litellm` to `>=1.83.7` (GHSA-xqmj-j6mv-4862) and forced `pip>=26.1` (GHSA-58qw-9mgm-455v) via `uv` override.
- Added `[tool.uv] override-dependencies` to resolve `jsonschema` conflict between `litellm` (pins `==4.23.0`) and `semgrep` (requires `>=4.25.1`).

---

### 2026-04-23

#### Changed

- Updated `lmstudio_status.py` script.

---

### 2026-04-22

#### Added

- Added YAML syntax validation to CI pipeline: `yamllint` dependency, `check-config-syntax` justfile target, integrated into both `ci` and `ci-quiet` targets.

#### Security

- Bumped `nbconvert` to `7.17.1` and `python-dotenv` to `1.2.2` to fix security vulnerabilities.

---

### 2026-04-20

#### Added

- Centralized logging module `src/util/log_util`: exposes `configure_root_logger` and `get_logger`; root logger writes to stdout, `app.log`, and `error.log` under the configured `logs_dir`; all scripts migrated off ad-hoc `print`/`logging.basicConfig` calls.

#### Fixed

- Missing-SRT error messages now include the directory path for easier diagnosis.

---

### 2026-04-18

#### Security

- Bumped `python-multipart` to `>=0.0.26` to address a security vulnerability.

---

### 2026-04-16

#### Added

- Added `topics-experiment` module: LLM-based topic segmentation experiment using LM Studio; selects model by `effective_context_length` (uses `loaded_context_length`, not architectural ceiling); skips files exceeding 90% of the loaded model's effective context to avoid runtime "Context size exceeded" errors.
- `LMStudioModel`: new `loaded_context_length` field and `effective_context_length` property.
- `select_best_model` ranks and filters by effective context length.
- Extracted `sanitize_channel_name` to shared `src/util` module.

---

### 2026-04-15

#### Fixed

- `topics-experiment` pre-scans pending files into already-processed/empty/pending buckets before the main loop; only pending items count toward `[n/total]`, percentage, and ETA display, eliminating ETA distortion from near-instant skips.

---

### 2026-04-14

#### Added

- LLM topic labeling now passes strict `response_format` JSON schema to `litellm.completion` so the model returns valid `LLMTopicLabelData` shape directly; uses `strict=False` parsing to tolerate control characters in responses.
- Added short/silent/no-audio sweep toolset:
  - `check-audio-track.sh`: ffprobe-based audio stream probe + mean/max volume, flags `LOW_VOLUME` below `LOW_VOLUME_THRESHOLD_DB` (default −40 dB).
  - `filter-short-videos.py`: scans videos and audio dirs for files <120s or with no audio stream; writes results to `filefilter.json`.
  - `remove-filtered-files.py`: deletes filtered files along with all upstream copies (videos → audio → transcripts).
  - `yt-downloader.sh`: rejects videos <120s at download time via `--match-filter "duration >= 120"`.

#### Changed

- `topic-tree.py` processing now pre-computes the work list, fails fast on matcher build errors (drop `try/except` swallowers), prints `[i/N]` progress per file, and tracks running total node count.

#### Security

- Upgraded Pillow `12.1.1`→`12.2.0` (GHSA-whj4-6x5x-4v2j) and pytest `9.0.2`→`9.0.3` (GHSA-6w46-j5rx-g56g).

---

### 2026-04-13

#### Added

- Added `fetch-video-metadata` helper: re-fetches `.info.json` for specific video IDs via `yt-dlp --skip-download --write-info-json` without re-downloading the video; preserves existing WAV stem for transcription pipeline compatibility.
- `just fetch-video-metadata CHANNEL +VIDEO_IDS` justfile target.

---

### 2026-04-11

#### Added

- `FileProcessingFilter` class: uses `channel/video_id` format (extracted from `[VIDEO_ID].ext` filenames) for O(1) set-based lookup; resolves base directories back to config keys with path normalization; added `Config.get_paths_config()` public accessor.
- `find-empty-transcripts.sh`: detects empty or incomplete transcript files across all formats (`.txt`, `.vtt`, `.tsv`, `.srt`).
- `find-files` tool: locates video files across all configured data directories.

#### Changed

- `transcribe_audio.py` now fails fast on first error instead of accumulating errors; detects empty WAV files; logs empty transcripts to `data_logs_dir`.
- `filefilter.json` entries migrated to `channel/video_id` format to avoid yt-dlp Unicode path mismatches.

---

### 2026-04-10

#### Added

- Per-channel transcription limiter: new config field caps new transcriptions per channel per pipeline run.
- `clean-video-files` script: interactive deletion of all files associated with a given YouTube video ID, with optional `downloaded.txt` archive cleanup.

#### Security

- Upgraded `cryptography` `46.0.6`→`46.0.7` (GHSA-p423-j2cm-9vmq).

---

### 2026-04-06

#### Changed

- Transcription channel ordering is now globally flat across all language groups (fewest-pending-first across the full channel list, not just within each language group); `channels_by_pending_count()` helper extracted.

---

### 2026-04-05

#### Added

- Transcription output now shows pending file count per channel in the header line.

#### Changed

- Channels within each language group are sorted by ascending pending file count so smaller backlogs complete first.

#### Fixed

- CI: added missing docstring to `audio-hours.py:main()`; upgraded `litellm` `1.80.11`→`1.83.0` (GHSA-53mr-6c8q-9789, GHSA-jjhc-v7c2-5hh6).

---

### 2026-04-04

#### Fixed

- Transcription pipeline now detects 0-byte SRT files produced by Whisper on speechless videos and skips them instead of moving them to the transcripts directory (which caused downstream hallucination-removal failures); stale empty transcript file groups are cleaned up automatically.

---

### 2026-04-03

#### Fixed

- `yt-downloader.sh` now uses `--merge-output-format mp4` to properly merge video and audio streams into a single MP4; replaces the previous `-f "b"` single-stream selection that omitted the video track.

---

### 2026-04-02

#### Fixed

- `convert-to-audio.sh` now exits with code 1 when any video file has no audio stream (previously silently skipped).
- Fixed `find`-pipe-`while` subshell bug that caused the failure counter to be lost in a subshell, preventing non-zero exit codes.
- Upgraded `aiohttp` `3.13.3`→`3.13.5` (vulnerability audit).

---

### 2026-04-01

#### Added

- `extract-audio` aborts when the target device has less than 2 GB free (checks `df -k` before each file conversion to prevent partial writes and disk-full failures mid-pipeline).

#### Changed

- `stats` displays zero values as dark grey dashes (`—`) instead of `0` for improved visual scanning of the pipeline status table.

---

## 2026-03

### 2026-03-31

#### Added

- `extract-audio` now formats silence duration removed as human-readable time (e.g. `13m 40.0s` instead of raw `819.97s`).

#### Changed

- `convert-to-audio` now shows an accurate processing file count by excluding already-converted files from the total.

#### Fixed

- Minimum silence duration increased from 1 s to 2 s to reduce the number of speech segments generated and avoid over-segmentation of natural speech.

---

### 2026-03-30

#### Added

- `check-video-integrity` rewritten from Bash to Python with a per-file result cache (`video_integrity_cache.json`); detects two corruption classes: unreadable/missing-duration files and low-bitrate files (< 1000 B/s for videos > 60 s); integrated into `all`, `all-ingestion`, and `all-quiet` pipelines between `extract-audio` and `transcribe`.
- `check-video-integrity` prints channel name and a progress line every 100 files.
- `extract-audio` now shows `[current/total]` file progress counter and total file count per channel header line.
- `stats` now shows inline colored change deltas (`+N`/`−N`) compared to the previous run; values capped at ±99 and fixed to 3 chars for column alignment; cache stored in `.cache/stats_previous.json`.

#### Fixed

- `extract-audio` now discards speech intervals shorter than 10 ms to prevent ffmpeg errors from degenerate `aselect` ranges.
- `extract-audio` now extracts each speech segment individually and concatenates with ffmpeg instead of using a single `aselect` filter expression, preventing `Cannot allocate memory` errors on videos with many segments.
- ffmpeg concat file paths with single quotes (apostrophes in filenames) are now properly escaped.

---

### 2026-03-28

#### Added

- Added project-specific `/delegate` command for agentic workflow task delegation (sourced from `DELEGATE.md`).

---

### 2026-03-20

#### Fixed

- `yt-downloader.sh` now monitors yt-dlp stdout and stderr in real-time via a named pipe; kills the yt-dlp process immediately on "Sign in to confirm you're not a bot" or expired-cookie detection instead of waiting for all queued items to fail.
- Hallucination digest script now correctly treats zero hallucinations as a success result (was returning exit code 1 on empty input).

---

### 2026-03-19

#### Security

- Upgraded `pyjwt` `2.10.1`→`2.12.1` and `tornado` `6.5.4`→`6.5.5` to fix known vulnerabilities.
- Changed yt-dlp format selection from `-f "best"` to `-f "b"` to suppress the deprecated format selection warning.

---

### 2026-03-09

#### Added

- Hierarchical topic detection system:
  - `topic-tree.py`: builds deterministic hierarchical topic trees using TreeSeg-style divisive SSE segmentation.
  - `lmstudio_status.py`: checks LM Studio connectivity and which models are currently loaded.
  - `export-to-minirag.py`: exports topic tree leaf nodes as `.txt` + `.json` pairs.
  - ACM CCS 2012 taxonomy loader with embedding cache for deterministic topic classification.
  - Added `defusedxml` (safe XML parsing for ACM taxonomy), `keybert`, and `yake` dependencies for keyphrase extraction.
  - `topic-tree`, `export-to-minirag`, and `status` justfile targets.
  - `min_duration` config option to skip videos shorter than the threshold before transcription.

#### Changed

- SRT output now uses plain text (all timestamps stripped) instead of simplified timestamp format; hallucination removal now outputs a `.txt` file alongside the cleaned `.srt`.
- Transcription pipeline skips videos shorter than configured `min_duration`.
- Help target reworked with grouped sections and corrected typo in header comment.

---

## 2026-02

### 2026-02-03

#### Fixed

- Topic extraction now catches `BadRequestError` with "No models loaded" from LM Studio and surfaces an actionable error message including the `lms load` command with the expected model name.

#### Security

- Upgraded `pip` from `25.3` to `26.0` (GHSA-6vgw-5pg2-w6jp).

---

### 2026-02-01

#### Added

- Pipeline status now shows total storage size in GB per channel (tracked via `total_size_bytes` field).

---

## 2026-01

### 2026-01-30

#### Added

- Pipeline status display completely redesigned as a unified table showing all pipeline stages in a single view: Videos, Archived Videos, Audio, Transcripts, Hallucination Analysis, Cleaned Transcripts, Topics Embeddings, Topics Segmentations, Topics Extracted, Topics Visualizations; two-row column headers; line width calculated dynamically by column count.

#### Fixed

- ETA duration format changed from `[HH:MM]` to `[XXh:YYm]` with explicit unit indicators for clarity; applied to both `transcribe_audio.py` and `extract-topics.py`.

---

### 2026-01-29

#### Added

- `ProgressTracker` class added to transcription pipeline: pre-scans files needing transcription, shows `[current/total] (percentage%) ETA [hh:mm]` per file, displays total elapsed time in final summary.
- Progress tracking and ETA display added to `extract-topics.py`; added `llm-topic-performance` benchmark tool for comparing LLM model throughput and quality on topic extraction.

#### Fixed

- CI: reduced `main()` complexity in `extract-topics.py` by extracting helper functions; added return type annotations and docstrings to `benchmark.py`; excluded `tools/llm-topic-performance` from deptry checks.

---

### 2026-01-28

#### Added

- `find-and-clean-empty-data-files.py`: scans the data folder for empty files (failed transcriptions, corrupt embeddings, etc.), shows a summary by extension, lists all files, and asks for confirmation before deleting; replaces a buggy bash version with issues on spaces in filenames.

#### Fixed

- Video download format changed from `"bestvideo+bestaudio/best"` to `"best"` (single pre-merged file), eliminating orphaned intermediate files (`.f401.mp4`, `.f251.webm`) when downloads are interrupted.
- English language detection now uses 80% majority vote instead of requiring all chunks to be English, handling spurious false positives from technical jargon (e.g. Cebuano detection from isolated code-heavy chunks).
- Pipeline status percentage now uses `max(videos, audio)` as denominator; prevents division-by-zero and incorrect percentages when archived videos still have audio files remaining.

---

### 2026-01-27

#### Security

- Pinned `python-multipart>=0.0.22` to fix GHSA-wp53-j4wj-2cfg.

---

### 2026-01-25

#### Added

- Added `just all-ingestion` target to run the full ingestion pipeline (download → audio → transcribe → archive → hallucination detection + removal) without topic detection.

---

### 2026-01-24

#### Security

- Documented and suppressed protobuf GHSA-7gcm-g887-7qv7 (DoS via uncontrolled recursion in `ParseDict` with nested `Any` messages; no fix available up to v6.33.4; project does not parse untrusted protobuf JSON from external sources).

---

### 2026-01-18

#### Added

- Embedding-based topic detection pipeline replacing the agent/critic architecture:
  - `SlidingWindowTopicSegmenter`: tokenizer → chunk encoder → similarity calculator → boundary detector → segment assembler pipeline.
  - LM Studio embedding generator with factory pattern for provider abstraction.
  - `TopicExtractionAgent`: LLM-based topic extraction (high/mid/specific topic levels) from detected segments.
  - Three-step decoupled scripts: `generate-embeddings.py` (SRT → `_embeddings.json`), `detect-boundaries.py` (`_embeddings.json` → `_segmentation.json`), `extract-topics.py` (`_segmentation.json` → `_topics.json`).
  - `visualize-embeddings.py`: stacked JPG visualizations of cosine similarity at multiple word distances; overlays segment backgrounds, topic labels, boundary markers, and threshold lines.
  - `EmbeddingsOutput`, `SegmentationOutput`, `WindowData` Pydantic schemas for portable JSON serialization.
  - `srt_util.py`: SRT parsing with word position tracking for timestamp mapping.
  - Justfile targets: `topics-embed`, `topics-boundaries`, `topics-extract`, `topics-all`, `topics-visualize`.
  - `TopicDetectionConfig`, `TopicDetectionEmbeddingConfig`, `TopicDetectionSlidingWindowConfig` config models.
  - `max_retries` and `retry_delay` fields on `LLMConfig`; retry logic with configurable delay and `<think>` tag handling for reasoning models.
  - All intermediate JSON files store paths relative to `data_dir` for portability across machines.

---

### 2026-01-15

#### Added

- FastText language detection pipeline:
  - `LanguageDetector` class using FastText `lid.176.ftz` compressed model (176 languages); `DetectionResult` dataclass; single/batch detection methods.
  - Per-transcript language analysis script: detects language distribution across all `.txt` transcript files, generates per-channel markdown reports, highlights non-English files, exits 1 if non-English transcripts detected.
  - Integrated into transcription pipeline: runs after transcription and metadata move; fails pipeline on non-English output.
  - `download-fasttext-models.sh` added to `just init` to download the 917 KB model on first setup.
  - `filter_alpha_words()` made public on `LanguageDetector`; `??` language code (no alphabetic words) accepted as non-error in pipeline output.
  - FastText and `numpy<2.0.0` (compatibility pin) added as dependencies.
- Shellscript analyzer notebook and tooling:
  - Renamed tool from `shellscript_env_var_args_detector` → `shellscript_analyzer`.
  - Jupyter notebook (`shellscript_analyzer.ipynb`): performance analysis of 10 LLM models on shell script classification; accuracy/precision/recall/F1 metrics; confusion matrices; misclassification patterns; speed vs accuracy Pareto frontier; 9 high-resolution (300 DPI) visualizations.
  - Automated model benchmark runner (`shellscript_analyzer_benchmark.sh`): sequential testing of 11 models, interactive LM Studio prompts, CSV parsing from log output.
  - `analyze_model_performance.py` moved from `scripts/` to `notebooks/` (analysis tool, not production script).
  - `reports/analysis/` directory structure with hallucination analysis and shellscript model performance reports.
- Hallucination classifier training tools in `notebooks/`:
  - `hallucination_analysis.ipynb`: interactive analysis and SVM classifier training.
  - Training scripts for linear, quadratic, and RBF SVM classifiers.
  - Classifier implementations with decision boundary visualizations.
- `reports_dir` path added to `Config`; `notebooks_gfx_dir` removed (notebooks now self-contained via `Path(__file__).parent`).
- `ci-quiet` now includes `code-config` validation step.

#### Changed

- `seaborn` added as direct dependency (used in notebooks).

#### Removed

- `notebooks_gfx_dir` configuration key removed from `PathsConfig` and all test fixtures.

---

### 2026-01-14

#### Added

- Multi-language transcription with automatic translation:
  - `language` field on `ChannelConfig` validated against 100 Whisper-supported language codes (required, single string).
  - `WhisperLanguages` utility class providing supported language codes.
  - Language grouping minimizes model switching: `medium.en` for English channels, `medium` for others; non-English content automatically translated to English.
  - `group-channels-by-language.py` helper script.
  - **BREAKING:** `channels.languages` (list) renamed to `channels.language` (single string).
- Shell script environment variable violation detector:
  - `detect_env_violations.py`: binary pass/fail check for scripts that use env vars without passing them as CLI arguments or reading from config files.
  - Hash-based caching (`.cache/shell_script_hashes.json`) to skip unchanged files.
  - Markdown report generation at `reports/shell_env_var_violations.md`.
  - `example_tests/` directory with pass/fail sample scripts.
  - Uses Autogen v0.7.5 with local LM Studio.
- Improved video download pipeline:
  - Download logging to `reports/video-download.log`.
  - `parse-and-archive-membersonly.py`: parses download log and appends members-only video IDs to the archive file with duplicate detection.
  - `--ignore-errors` flag added to yt-dlp for resilience against unavailable videos; exit code 1 treated as success (expected for private/unavailable videos).
- `TranscriptionConfig` Pydantic model: model repos, anti-hallucination thresholds, metadata usage, processing options; added to config template.
- `TranscriptionArgumentHelper`: builds context-aware initial prompts for Whisper (title + description from `.info.json`; empty prompt for translation tasks to avoid language confusion).

#### Changed

- Replaced 230-line bash transcription script with Python orchestrator (`transcribe_audio.py`): language-specific handling (omits `--language` for translate tasks), better error handling and progress tracking, cross-device file moves via `shutil.move()`.
- Added 5-second cooldown pause after each successful transcription job to reduce GPU/thermal pressure.
- `autogen-ext[openai]`, `autogen-agentchat`, `requests` added as dependencies; `mlx-whisper` added to `pyproject.toml`.

#### Removed

- Removed old 230-line bash transcription script and channel grouping script.

---

### 2026-01-13

#### Added

- Extended topic segmentation and transcript processing pipeline with additional stages.

#### Fixed

- Corrected path construction in the hallucination analysis notebook.

---

### 2026-01-12

#### Added

- Added `just all-quiet` target for silent end-to-end pipeline execution with minimal output.
- Updated hallucination analysis notebook to load the data directory from `config.yaml` via `Config` class.

---

### 2026-01-11

#### Changed

- Newspaper dev server port changed from 3000 to 12000 in both Nuxt config and justfile.
- Replaced inline bash validation in the newspaper justfile target with `validate_articles_dir.py` — a dedicated Python script that uses `config.yaml` as the single source of truth for the articles directory path.

---

### 2026-01-08

#### Added

- Nuxt.js-based newspaper frontend (`frontend/newspaper/`):
  - Renamed from `ai-times-nuxt`; updated all references in justfile, README, and package.json.
  - `ArticleParser` + `ArticleCompiler` (Pydantic models: `ArticleFrontmatter`, `MarkdownArticle` with slug support) for transforming markdown articles into newspaper format.
  - Migrated to `@nuxt/content` module: direct markdown querying via `layout.json`; full article content on individual pages; build step copies markdown files to `content/` directory.
  - Article preprocessing script: strips redundant H1/H2/separator after YAML frontmatter before copying to frontend.
  - Clickable headlines and masthead title (all article headlines in hero/featured/secondary sections link to full article pages with hover effects).
  - Article pages use full container width (removed 800px constraint and white background).
  - `python-frontmatter` and `markdown-it-py` dependencies added.
  - Nuxt updated to 3.19.0; `better-sqlite3` added for `@nuxt/content` SQLite storage.
  - `just compile-articles` and `just newspaper` justfile targets.
- Full topic segmentation and transcript processing pipeline (PR #4):
  - `TopicSegmentationAgent` + `TopicSegmentationCritic` with agent/critic/orchestrator pattern (litellm, srt deps).
  - `TopicBlock` Pydantic schema: `id`, `start`/`end` in SRT format, single `topic` string slug.
  - Prompts moved to Python modules (`agent_prompts.py`, `critic_prompts.py`) alongside agents.
  - `FSUtil` module: find files by extension, read/write text files, write JSON with parent directory creation.
  - SRT preprocessing pipeline: batch converts SRT files to plain text organized by channel.
  - Main pipeline: multi-channel processing, tiktoken token counting, skip-already-processed logic, per-file success/failure tracking with summary statistics.
  - `scripts/config.sh`: centralized shell configuration for all paths, silence detection (−40 dB / 1 s min), transcription settings; settings overridable via environment variables; sourced by `archive-videos.sh`, `convert_to_audio.sh`, `yt-downloader.sh`.
  - `move-metadata.sh`: organizes YouTube `.info.json` metadata files into `metadata/<channel>/video/` subdirectories.
  - `move-transcript-metadata.sh`: relocates transcript JSON files to `metadata/<channel>/transcript/`.
  - YouTube metadata-based Whisper prompting: extracts title + description from `.info.json`; fallback to generic AI/ML prompt when metadata unavailable; `hallucination_silence_threshold=2.0`, `compression_ratio_threshold=2.0`.
  - Word-level timestamps (`--word-timestamps`) enabled in Whisper output.
  - `token_validator.py`: validates token count via tiktoken before every LLM call; raises `ContextWindowExceededError` when 90% threshold exceeded; 44 tests.
  - Pipeline status command (`scripts/status.py`): per-channel stats (videos, audio, transcripts, archived), completion percentages, active/archived split tables.
  - Per-channel download limiter config field (0 = skip, −1 = unlimited, N = exact limit).
  - `RepetitionDetector`: suffix array analysis, consecutive-repetition filter (reduces false positives by 97%); processes SRT entries with sliding window; outputs JSON reports; 117 true hallucinations detected on test data.
  - SVM `HallucinationClassifier`: loads coefficients from `config.yaml`; linear decision function; no ML library at inference time; training tools (scikit-learn, pandas, matplotlib) added as dependencies.
  - `just analyze-transcripts-hallucinations`: detection + digest generation in one command.
  - Deterministic hallucination removal script: string replacement using detection patterns; validates cleaned output with `RepetitionDetector`; 100% success on 31/31 test files.
  - Config template validation script (`scripts/check/config_template.py`): ensures `config.yaml` and `config.yaml.template` have matching key structures; integrated into CI as first check.
  - `PathsConfig` Pydantic model: validates all project directory paths; required in `config.yaml`; getter methods for all path fields.
  - `init-directories.sh`: reads all directory paths from `config.yaml` and creates them; replaces hardcoded `mkdir` commands in justfile.
  - `just all` target: complete pipeline automation (`ci-quiet` → `download-videos` → `extract-audio` → `transcribe` → `archive-videos` → `analyze-transcripts-hallucinations` → `transcripts-remove-hallucinations`).
  - `just notebooks` target: launches Jupyter server without token requirement; auto-opens browser on macOS/Linux.

---

### 2026-01-07

#### Added

- AI-powered fake test detector tool (`tools/fake_test_detector/`):
  - Detects 12 patterns of fraudulent unit tests: empty bodies, missing assertions, trivial assertions, exception swallowing, mock-only tests with no business logic validation, and more.
  - Hash-based caching (`.cache/unit_test_hashes.json`) to skip unchanged test files.
  - AST-based test case extraction from Python test files.
  - LLM analysis via Autogen v0.7.5 + LM Studio (OpenAI-compatible API).
  - Markdown report generation.
  - Integrated into CI pipeline via `just ci-ai` command.
  - Dependencies added: `autogen-ext[openai]`, `autogen-agentchat`, `requests`.

---

## 2025-12

### 2025-12-30

#### Added

- YouTube video downloader (`scripts/yt-downloader.sh`): uses yt-dlp with Chrome cookie authentication, downloads videos from the past day with lazy playlist loading, maintains a download archive to prevent re-downloading, supports both channel URLs and individual video URLs.
- YAML-based configuration system: `config.yaml.template` (channel configuration schema), `src/config.py` (`Config` class + `ChannelConfig` Pydantic model with validation for `url`, `name`, `category`, `description`; methods `get_channels()`, `get_channel(index)`, `get_channel_by_name(name)`; 40+ test cases).
- Comprehensive CI/CD justfile with targets: `init`, `help`, `destroy`, `code-style`, `code-format`, `code-typecheck`, `code-lspchecks`, `code-security`, `code-deptry`, `code-stats`, `code-spell`, `code-audit`, `code-semgrep`, `test`, `test-coverage`, `ci`, `ci-quiet`.
- Development infrastructure: `.python-version` (Python 3.12), `.pre-commit-config.yaml` (runs `just ci-quiet` on commit), `.semgrepignore`, `pyrightconfig.json` (strict mode).
- Semgrep rules enforcing project conventions: no default parameter values, no `# noqa` or `# type: ignore` suppression, no sneaky fallbacks (`or`/`getattr`/ternary), no module-level constants.
- `AGENTS.md`: consolidated development rules (git commit guidelines, testing rules, Python execution via `uv run`, project structure, API key management); `CLAUDE.md` now redirects to `AGENTS.md`.
- `pyproject.toml`: `pydantic>=2.0.0`, `pyyaml>=6.0` as core deps; `pytest`, `ruff`, `mypy`, `pyright`, `bandit`, `semgrep`, `deptry`, `codespell`, `pip-audit`, `pygount`, `pre-commit` as dev deps; full tool configuration (ruff line-length 120, mypy strict, pytest asyncio, 80% coverage threshold).
- Comprehensive README: system workflow description, repository structure, channel configuration instructions, environment variables, common commands, code quality tools, project status checklist.

---

### 2025-12-27

#### Added

- Created initial project repository for the agentic news generator.
