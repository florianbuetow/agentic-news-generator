# Data Directory I/O Patterns

**Snapshot date:** 2026-03-15
**Scope:** All justfile targets and their backing scripts — how data flows through the pipeline via input and output directories.

---

## Central Path Resolution

All paths are defined in `config/config.yaml` under the `paths:` key and loaded via `src/config.py` (`Config` / `PathsConfig`). Shell scripts source `scripts/config.sh`, which reads the same YAML via a Python one-liner and exports Bash variables. The actual data root on the production machine is an external volume; the template (`config/config.yaml.template`) uses relative `./data/` paths.

---

## Input Folder Structure

```
{data_dir}/
├── downloads/                              # Raw ingested content (staging area)
│   ├── videos/{channel_name}/              # Downloaded YouTube videos (.mp4, .mkv, .webm, etc.)
│   ├── audio/{channel_name}/               # Extracted WAV audio files
│   ├── transcripts/{channel_name}/         # Raw SRT transcripts from Whisper
│   ├── transcripts-hallucinations/         # Hallucination detection JSON results
│   │   └── {channel_name}/
│   ├── transcripts_cleaned/                # Cleaned SRT + TXT (hallucinations removed)
│   │   └── {channel_name}/
│   ├── transcripts-topics/                 # (Legacy) topic transcripts
│   └── metadata/{channel_name}/            # Organized metadata
│       ├── video/                          # .info.json from YouTube
│       ├── audio/                          # .silence_map.json from ffmpeg
│       └── transcript/                     # Transcript metadata JSON
├── input/                                  # Static / curated input data
│   ├── taxonomies/                         # ACM CCS taxonomy XML + embedding cache
│   └── newspaper/articles/                 # Curated markdown articles for newspaper frontend
├── articles/input/                         # Programmatically generated article bundles
│   └── {bundle_dir}/                       # manifest.json + copied source files per bundle
├── models/                                 # FastText language detection models
├── knowledgebase/                          # Knowledge base data for article generation
├── knowledgebase_index/                    # Pre-built KB embedding index
└── institutional_memory/                   # Article generation institutional memory
    ├── fact_checking/
    └── evidence_finding/
```

## Output Folder Structure

```
{data_dir}/
├── output/
│   ├── topics/{channel_name}/              # Topic detection pipeline outputs
│   │   ├── *_embeddings.json               # Step 1: generate-embeddings.py
│   │   ├── *_segmentation.json             # Step 2: detect-boundaries.py
│   │   ├── *_topics.json                   # Step 3: extract-topics.py
│   │   └── *_topic_tree.json               # topic-tree.py (TreeSeg hierarchical)
│   ├── articles/{bundle_slug}/             # Final generated articles (markdown)
│   ├── article_editor_runs/                # Editor run artifacts (drafts, reviews, logs)
│   ├── newspaper/                          # Generated static website (HTML/CSS/JS)
│   └── hallucination_digest.md             # Hallucination summary report
├── archive/
│   └── videos/{channel_name}/              # Archived (fully processed) video files
├── temp/                                   # Temporary processing files
└── logs/                                   # Application logs
```

---

## Pipeline Flow: Per-Target I/O Map

| Justfile Target | Reads From | Writes To | Path Source |
|---|---|---|---|
| `download-videos` | YouTube (network) | `downloads/videos/{channel}/` | config.yaml `paths.data_downloads_videos_dir` |
| `extract-audio` | `downloads/videos/{channel}/` | `downloads/audio/{channel}/` + `downloads/metadata/{channel}/audio/` | config.sh |
| `transcribe` | `downloads/audio/{channel}/` + `downloads/metadata/{channel}/video/` | `downloads/transcripts/{channel}/` | config.yaml `paths.data_downloads_audio_dir`, `paths.data_downloads_transcripts_dir`, `paths.data_downloads_metadata_dir` |
| `archive-videos` | `downloads/videos/{channel}/` + `downloads/audio/{channel}/` | `archive/videos/{channel}/` (moves video, deletes audio) | config.sh |
| `analyze-transcripts-hallucinations` | `downloads/transcripts/{channel}/` | `downloads/transcripts-hallucinations/{channel}/` + `output/hallucination_digest.md` | config.yaml `paths.data_downloads_transcripts_dir`, `paths.data_downloads_transcripts_hallucinations_dir`, `paths.data_output_dir` |
| `transcripts-remove-hallucinations` | `downloads/transcripts/{channel}/` + `downloads/transcripts-hallucinations/{channel}/` | `downloads/transcripts_cleaned/{channel}/` (.srt + .txt) | config.yaml `paths.data_downloads_transcripts_dir`, `paths.data_downloads_transcripts_hallucinations_dir`, `paths.data_downloads_transcripts_cleaned_dir` |
| `topics-embed` | `downloads/transcripts_cleaned/{channel}/` | `output/topics/{channel}/*_embeddings.json` | config.yaml `paths.data_downloads_transcripts_cleaned_dir`, `topic_detection.output_dir` |
| `topics-boundaries` | `output/topics/{channel}/*_embeddings.json` | `output/topics/{channel}/*_segmentation.json` | config.yaml `topic_detection.output_dir` |
| `topics-extract` | `output/topics/{channel}/*_segmentation.json` | `output/topics/{channel}/*_topics.json` | config.yaml `topic_detection.output_dir` |
| `topics-tree` | `downloads/transcripts_cleaned/{channel}/*.srt` | `output/topics/{channel}/*_topic_tree.json` | config.yaml `paths.data_downloads_transcripts_cleaned_dir`, `topic_detection.output_dir` |
| `prepare-article-input` | `downloads/transcripts_cleaned/` + `downloads/metadata/` + `output/topics/` | `articles/input/{bundle}/manifest.json` | config.yaml `paths.data_downloads_transcripts_cleaned_dir`, `paths.data_downloads_metadata_dir`, `topic_detection.output_dir`, `paths.data_articles_input_dir` |
| `generate-articles` | `articles/input/{bundle}/manifest.json` | `output/articles/` + `output/article_editor_runs/` | config.yaml `article_generation.editor.output.final_articles_dir`, `article_generation.editor.output.run_artifacts_dir` |
| `compile-articles` | `input/newspaper/articles/*.md` | `input/newspaper/articles.js` | config.yaml `article_compiler.input_dir`, `article_compiler.output_file` |
| `newspaper-generate` | `input/newspaper/articles/*.md` | `output/newspaper/` | config.yaml `paths.data_input_dir` + hardcoded frontend paths |
| `newspaper-serve` | `input/newspaper/articles/*.md` | (dev server, no persistent output) | config.yaml `paths.data_input_dir` |
| `export-to-minirag` | `output/topics/{channel}/*_topic_tree.json` | custom `--export-dir` (CLI arg) | CLI argument + config.yaml for metadata |
| `stats` | all data directories (read-only scan) | console output only | config.yaml (multiple path getters) |
| `clean-empty-files` | `{data_dir}/**` (recursive scan) | deletes empty files (with `--delete` flag) | config.yaml `paths.data_dir` |

---

## Key Observations

1. **Left-to-right flow:** `downloads/ → output/` with `archive/` as a side-effect for fully processed videos.

2. **`downloads/` is both input and output.** It serves as a staging area. Raw downloads land there, and intermediate processing (hallucination detection, cleaning) writes back into `downloads/` subdirectories. The cleaned transcripts in `downloads/transcripts_cleaned/` are the primary input for all downstream analysis.

3. **`output/topics/` is both output and input.** The topic pipeline writes there, then `prepare-article-input` reads from it to build article bundles.

4. **Two separate "input" directories exist with different roles:**
   - `data/input/` — static/curated data (taxonomy files, newspaper articles)
   - `data/articles/input/` — programmatically generated article bundles

5. **Channel-based partitioning is universal.** Almost every directory uses `{channel_name}/` subdirectories to organize files per YouTube channel.

6. **Newspaper reads from curated input, not generated output.** The newspaper pipeline reads from `data/input/newspaper/articles/` (human-curated), not from `data/output/articles/` (LLM-generated). These are separate flows.

7. **Topic detection has two paths:**
   - The old step-by-step pipeline (`topics-embed` → `topics-boundaries` → `topics-extract`) produces `_embeddings.json` → `_segmentation.json` → `_topics.json`.
   - The newer `topics-tree` target runs the full TreeSeg hierarchical pipeline in one step, producing `_topic_tree.json`.
   - `topics-all` currently calls only `topics-tree`.

8. **Config keys that control output directories outside `paths:`:**
   - `topic_detection.output_dir` — relative to `data_dir`, resolves to `output/topics`
   - `hallucination_detection.output_dir` — relative to `data_dir`, resolves to `downloads/transcripts-hallucinations`
   - `article_generation.editor.output.final_articles_dir` — absolute path
   - `article_generation.editor.output.run_artifacts_dir` — absolute path
   - `article_compiler.input_dir` / `article_compiler.output_file` — relative paths

---

## How This Document Was Created

This snapshot was produced by an AI-assisted codebase analysis on 2026-03-15. The process:

1. **Read `justfile`** to identify all pipeline targets and the scripts they invoke.
2. **Read `config/config.yaml` and `config/config.yaml.template`** to catalog all declared paths.
3. **Read `src/config.py`** to understand how paths are resolved programmatically.
4. **Read each script** in `scripts/` (both Python and shell) to trace which config paths it reads from and writes to — including how it discovers files (glob patterns, `rglob`, channel iteration).
5. **Read `scripts/config.sh`** to understand the shell-side path resolution.
6. **Cross-referenced** justfile target ordering with script I/O to build the per-target flow table.

### How to Re-Create or Check for Changes

To verify this snapshot is still accurate or produce an updated version:

```bash
# 1. List all justfile targets and their script invocations
grep -E '^\w|uv run|bash scripts/' justfile

# 2. Check config path keys haven't changed
grep '_dir:' config/config.yaml.template

# 3. Check each script's path resolution — look for config getters and Path() usage
grep -rn 'getData\|get_topic_detection\|get_article_generation\|get_article_compiler\|data_dir\|output_dir' scripts/ src/config.py

# 4. Check shell scripts for config.sh sourcing and variable usage
grep -rn 'source.*config.sh\|DATA_\|DOWNLOADS_\|OUTPUT_\|ARCHIVE_' scripts/*.sh

# 5. Diff the current config template against the snapshot's path list
diff <(grep '_dir:' config/config.yaml.template | sort) <(grep '_dir:' config/config.yaml | sort)
```

**Signs this snapshot needs updating:**
- New `*_dir` keys added to `config/config.yaml.template`
- New scripts added to `scripts/` or new justfile targets
- Changes to `src/config.py` path resolution (new getters, renamed fields)
- Changes to `scripts/config.sh` variable exports

**Recommended cadence:** Re-verify after any PR that adds pipeline stages, modifies `config.yaml.template`, or changes `src/config.py`.
