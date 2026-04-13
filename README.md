![News Generator](./docs/news_generator.png)

# Agentic News Generator

An AI-powered YouTube news aggregator that crawls AI-focused YouTube channels, transcribes video content, segments transcripts by topic, and generates a weekly newspaper-style HTML digest using autonomous AI agents.


## Overview

This system automatically:
1. Downloads videos from pre-configured YouTube channels
2. Extracts audio from videos to WAV format
3. Transcribes audio using MLX Whisper (medium.en for English, medium for other languages) with automatic translation to English and multiple output formats
4. Segments transcripts into topic-based sections using AI analysis
5. Aggregates related topic segments across multiple videos
6. Generates news articles from aggregated content using AI agents
7. Produces a newspaper-style HTML digest
8. Archives processed videos to save disk space

## Repository Structure

```
agentic-news-generator/
├── pyproject.toml          # Project dependencies and metadata
├── justfile                # Build and run commands
├── AGENTS.md               # AI agent development rules
├── CLAUDE.md               # Redirect to AGENTS.md
├── README.md               # This file
├── config/                 # Configuration files
│   ├── config.yaml        # YouTube channel configuration
│   ├── config.yaml.template  # Template for config.yaml
│   └── semgrep/           # Semgrep rule configurations
├── src/                    # Source code
│   ├── main.py           # Main entry point
│   ├── config.py         # Configuration loading
│   ├── processing/       # Processing modules
│   │   └── repetition_detector.py  # Hallucination detection algorithm
│   └── util/             # Utility modules
│       └── fs_util.py    # File system utilities
├── scripts/                # Utility scripts
│   ├── yt-downloader.sh   # YouTube video downloader (filters out <120s)
│   ├── convert_to_audio.sh  # Video to audio converter
│   ├── transcribe_audio.sh  # Audio transcription (MLX Whisper)
│   ├── transcript-hallucination-detection.py  # Hallucination detection
│   ├── create-hallucination-digest.py  # Digest report generator
│   ├── archive-videos.sh  # Archive processed videos
│   ├── find-files.sh     # Locate every file for a video ID
│   ├── check-audio-track.sh  # Probe a video for audio stream + volume
│   ├── filter-short-videos.py  # Add short / no-audio files to filefilter.json
│   ├── remove-filtered-files.py  # Delete filtered files and upstream copies
│   ├── fetch-video-metadata.py  # Backfill missing info.json for an ID
│   └── find-empty-transcripts.py  # List transcripts that failed silently
├── tests/                  # Test suite
├── prompts/                # LLM prompt templates
├── frontend/               # Frontend applications
│   └── newspaper/         # Nuxt-based newspaper generator
│       ├── nuxt.config.ts # Nuxt configuration
│       ├── package.json   # Node.js dependencies
│       ├── components/    # Vue components (Masthead, ArticleCard, etc.)
│       ├── data/          # Content data (articles.js)
│       ├── pages/         # Nuxt pages
│       └── assets/        # Styles and static assets
└── data/                   # Data files
    ├── downloads/         # Downloaded and processed content
    │   ├── videos/       # Downloaded videos (by channel)
    │   ├── audio/        # Extracted WAV files (by channel)
    │   ├── transcripts/  # Transcripts in multiple formats (by channel)
    │   │   └── {channel}/transcript-analysis/  # Hallucination analysis JSON
    │   └── metadata/     # Video metadata and silence maps (by channel)
    ├── archive/           # Archived content
    │   └── videos/       # Processed videos moved here
    ├── input/             # Input data files
    │   └── newspaper/     # Newspaper input data
    │       └── articles.js # Generated articles data for Nuxt
    ├── temp/              # Temporary processing files
    └── output/            # Generated output files
        ├── hallucination_digest.md  # Hallucination analysis report
        ├── topics/        # Per-topic aggregated JSON files
        └── newspaper/     # Generated HTML newspaper (static site output)
```

## Prerequisites

- **macOS with Apple Silicon** (M1, M2, M3, or later) - Required for MLX Whisper
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager installed
- [just](https://github.com/casey/just) command runner installed
- [FFmpeg](https://ffmpeg.org/) for audio extraction (`brew install ffmpeg`)
- [jq](https://jqlang.github.io/jq/) for JSON processing (`brew install jq`)
- Chrome browser (for YouTube cookie authentication with yt-dlp)
- Node.js 18+ and npm (for the Nuxt newspaper frontend)

## Setup

### 1. Initialize the Project

Initialize the development environment:

```bash
just init
```

This will:
- Create all required directories
- Set up a virtual environment using `uv`
- Install all dependencies from `pyproject.toml`

### 2. Configure YouTube Channels

Copy the configuration template and edit it:

```bash
cp config/config.yaml.template config/config.yaml
```

Edit `config/config.yaml` to add your YouTube channels. Each channel can be configured with:
- `url`: YouTube channel URL
- `name`: Display name
- `language`: Source language code (e.g., "en", "de", "ja") - **Required**
  - English channels (`"en"`): Transcribed in English
  - Non-English channels: Automatically translated to English
- `download-limiter`: Maximum videos to download (or -1 for unlimited)
- `category`: Channel category (optional, for structured format)
- `description` or `what_you_get`: Channel description (for structured format)
- `vibe`: Free-form channel description (alternative flexible format)

Channels can use either:
- **Structured format**: `category` + `description` (or `what_you_get`)
- **Flexible format**: `vibe` only

See `config/config.yaml` for examples of pre-configured AI-focused channels in multiple languages.

### 3. Configuration

All configuration parameters are managed through `config/config.yaml`. **Never use environment variables** for configuration—all settings must be loaded through `config.py` from `config.yaml`.

See `config/config.yaml.template` for all available configuration options including:
- API keys and endpoints for LLM services
- LM Studio configuration for local LLM
- Processing parameters and thresholds

## Usage

### Run the Main Application

```bash
just run
```

### View Available Commands

```bash
just
```

Or see detailed help:

```bash
just help
```

### Common Commands

#### Video Processing Pipeline
- `just download-videos` - Download videos from configured YouTube channels
- `just extract-audio` - Convert downloaded videos to WAV audio files
- `just transcribe` - Transcribe audio files using MLX Whisper (medium.en/medium with auto-translation)
- `just analyze-transcripts` - Analyze transcripts for hallucinations and generate digest
- `just archive-videos` - Archive processed videos and clean up audio files

#### Tools
- `just find-files <video-id>` - Find all files for a video ID across all data directories
- `just check-audio-track <channel> <video-id>` - Probe a downloaded video for audio stream presence + mean/max volume (flags `LOW_VOLUME` below `LOW_VOLUME_THRESHOLD_DB`, default `-40 dB`)
- `uv run scripts/filter-short-videos.py [--write]` - Scan videos + audio dirs and add any file `<120s` (or any video with no audio stream) to `config/filefilter.json`
- `uv run scripts/remove-filtered-files.py [--execute]` - Delete every file referenced by `filefilter.json` together with its upstream copies (video → audio → transcript pipeline)

#### Development
- `just init` - Initialize development environment
- `just run` - Run the main application
- `just newspaper-generate` - Generate static newspaper website from articles data
- `just newspaper-serve` - Run newspaper development server at http://localhost:3000
- `just test` - Run unit tests
- `just test-coverage` - Run tests with coverage report
- `just code-format` - Auto-fix code style and formatting
- `just code-style` - Check code style (read-only)
- `just code-typecheck` - Run type checking with mypy
- `just ci` - Run all validation checks
- `just destroy` - Remove virtual environment

### Video Processing Workflow

The complete video processing pipeline:

```bash
# Step 1: Download videos from YouTube channels
just download-videos

# Step 2: Extract audio from videos (converts to 16kHz mono WAV)
just extract-audio

# Step 3: Transcribe audio using MLX Whisper (medium.en/medium)
# English channels: transcribed in English (medium.en model)
# Non-English channels: translated to English (medium model)
# Generates: .txt, .srt, .vtt, .tsv, .json files
just transcribe

# Step 4: Analyze transcripts for hallucinations
# Generates: JSON analysis + markdown digest
just analyze-transcripts

# Step 5: Archive processed videos and clean up audio files
# Moves videos to data/archive/videos/
# Deletes audio files from data/downloads/audio/
just archive-videos
```

**Notes:**
- All operations are idempotent (safe to re-run)
- Files are organized by channel name
- Transcription uses Whisper medium.en (English) and medium (multilingual) models
- Channels are grouped by language to minimize model switching
- Models are cached in `~/.cache/huggingface/hub/`
- Archive step frees up disk space by moving videos and deleting intermediate audio

### Anti-Hallucination Transcription Features

The transcription pipeline includes advanced anti-hallucination features to improve transcription quality:

**YouTube Metadata-Based Prompting:**
- Automatically extracts video title and description from YouTube metadata
- Builds context-aware prompts: `"This is a YouTube video with the title: [Title]. Description: [Description]"`
- Helps Whisper understand the content domain and improve accuracy for technical terms
- Falls back to generic AI/ML prompt if metadata is unavailable

**Anti-Hallucination Parameters:**
- **`hallucination_silence_threshold: 2.0s`** - When hallucination is detected after 2+ seconds of silence, seeks past the silence and retries transcription
- **`compression_ratio_threshold: 2.0`** - Stricter threshold (vs default 2.4) to catch gibberish/repetitive outputs
- **`initial_prompt`** - Dynamic prompt built from YouTube metadata to guide transcription

**Configure Settings** (in `scripts/config.sh`):
```bash
# Anti-hallucination settings
HALLUCINATION_SILENCE_THRESHOLD=2.0    # Seconds (default: 2.0)
COMPRESSION_RATIO_THRESHOLD=2.0         # Lower = stricter (default: 2.0)
USE_YOUTUBE_METADATA=true               # Use video metadata in prompts (default: true)
```

**Disable Metadata Prompting:**
```bash
USE_YOUTUBE_METADATA=false just transcribe
```

### Skipping Specific Files During Transcription

Individual videos can be excluded from transcription by adding them to `config/filefilter.json`. The filter is matched on the 11-character YouTube video ID, so a single entry skips every on-disk file for that video (mp4, wav, info.json, silence_map.json, AppleDouble sidecars, etc.).

**Format:**
```json
{
    "data_downloads_audio_dir": [
        "Anthropic/56kq0VTkU4k",
        "Lex_Fridman/XW0QZmtbjvs"
    ]
}
```

- The top-level key must be a pipeline dir field from `PathsConfig`: `data_downloads_videos_dir`, `data_downloads_audio_dir`, or `data_downloads_transcripts_dir`. An entry under a downstream key also blocks the upstream stages.
- Each value is `<Channel>/<video_id>` — **no file extension, no filename**. The channel is the subdirectory under the pipeline dir, the video ID is the 11-char YouTube ID from `[...]` in the filename.

**Add entries manually:**
1. `just find-files <video-id>` — confirm the channel and ID
2. Append `"<Channel>/<video_id>"` to the relevant key in `config/filefilter.json`
3. Run `just transcribe` — matching files are skipped

**Add entries automatically (short + no-audio sweep):**
```bash
# Scan every channel for files <120s OR videos with no audio stream
uv run scripts/filter-short-videos.py              # dry run
uv run scripts/filter-short-videos.py --write      # update filefilter.json

# Optional: run for one channel only
uv run scripts/filter-short-videos.py --channel Anthropic --write

# Optional: change the threshold
uv run scripts/filter-short-videos.py --max-duration 60 --write
```
The scan walks `data_downloads_videos_dir/<channel>/` **and** `data_downloads_audio_dir/<channel>/` so orphan wav files without a matching video are also caught. For each video it runs `ffprobe` to verify an audio stream is present; videos without audio are added regardless of duration.

**Delete already-filtered files from disk (with upstream cleanup):**
```bash
uv run scripts/remove-filtered-files.py            # dry run — shows what would be removed
uv run scripts/remove-filtered-files.py --execute  # actually unlink
```
This resolves every entry in `filefilter.json` to concrete paths and removes them together with their upstream copies. The pipeline order is `videos → audio → transcripts`, so an entry under `data_downloads_audio_dir` also removes the video file; an entry under `data_downloads_transcripts_dir` removes audio and video too. Matching is lexical on the `[<video_id>]` substring, so every sibling file (`.info.json`, `.silence_map.json`, `._*` sidecars) is swept up automatically.

**Probe a single video for audio health:**
```bash
just check-audio-track <Channel> <video-id>
# Statuses: HAS_AUDIO | LOW_VOLUME | NO_AUDIO_STREAM | EMPTY_AUDIO_STREAM | NOT_FOUND | AMBIGUOUS
# Tune the low-volume threshold (default -40 dB):
LOW_VOLUME_THRESHOLD_DB=-45 just check-audio-track Anthropic 56kq0VTkU4k
```

**Preventing future short-video downloads:**
`scripts/yt-downloader.sh` passes `--match-filter "duration >= 120"` to `yt-dlp`, so YouTube Shorts and sub-120s clips are rejected at download time and never enter the pipeline.

### Multi-Language Transcription Support

The transcription pipeline supports videos in any language supported by Whisper (100+ languages). Videos from non-English channels are automatically translated to English during transcription.

**How It Works:**
- Each channel in `config/config.yaml` specifies its source language using the `language` field
- **English channels** (`language: "en"`): Transcribed directly in English using the optimized `medium.en` model
- **Non-English channels** (e.g., `language: "de"`, `language: "ja"`): Translated to English using the multilingual `medium` model during transcription
- Channels are **grouped by language** during processing to minimize model switching overhead
- All transcript outputs are in English for consistent downstream processing

**Configuration Example:**
```yaml
channels:
  # English channel - uses medium.en model
  - url: https://www.youtube.com/@AIExplained
    name: AI Explained
    language: "en"  # Source language: English
    download-limiter: 20

  # German channel - uses medium model with translation
  - url: https://www.youtube.com/@TheMorpheusVlogs/videos
    name: The Morpheus
    language: "de"  # Source language: German → translates to English
    download-limiter: 1

  # Japanese channel - uses medium model with translation
  - url: https://www.youtube.com/@SomeJapaneseChannel
    name: Japanese Tech Channel
    language: "ja"  # Source language: Japanese → translates to English
    download-limiter: 10
```

**Models Used:**
- **English-only model**: `mlx-community/whisper-medium.en-mlx` (769M parameters)
  - Optimized for English transcription
  - Task: `transcribe` (same language output)
  - 99.3% accuracy on technical AI/ML terminology

- **Multilingual model**: `mlx-community/whisper-medium-mlx` (769M parameters)
  - Supports 100+ languages
  - Task: `translate` (translates to English)
  - Used for all non-English source languages

**Processing Flow:**
```
1. Script groups channels by language: {en: [...], de: [...], ja: [...]}
2. For each language group:
   - Loads appropriate model once (medium.en for 'en', medium for others)
   - Processes all channels in that language group
   - Minimizes model switching for better performance
3. For English channels:
   - Task: transcribe
   - Language: en
   - Model: medium.en
4. For non-English channels:
   - Task: translate
   - Language: de, ja, etc.
   - Model: medium
   - Output: English transcripts
```

**Supported Languages:**
All languages supported by Whisper are available, including:
- European: de (German), fr (French), es (Spanish), it (Italian), pt (Portuguese), etc.
- Asian: ja (Japanese), zh (Chinese), ko (Korean), hi (Hindi), etc.
- Middle Eastern: ar (Arabic), fa (Persian), he (Hebrew), tr (Turkish), etc.
- And 90+ more languages

For a complete list, see the Whisper documentation or `src/util/whisper_languages.py`.

**Benefits:**
- **Single output language**: All transcripts in English for consistent processing
- **Optimized performance**: Language grouping reduces model loading overhead
- **Automatic translation**: No separate translation step needed
- **High quality**: Whisper's built-in translation produces natural English output
- **YouTube metadata preserved**: Title and description used as context (in original language)

**First Run Note:**
- On first use, the multilingual model (~1.5GB) will be downloaded and cached
- Subsequent runs will use the cached model from `~/.cache/huggingface/hub/`

### Language Detection

The system includes a FastText-based language detector that supports 176 languages.

**Model Download:**

The language detection model is automatically downloaded during initialization:

```bash
just init
```

The model (lid.176.ftz, 917KB) is downloaded to `data/models/fasttext/` as configured in `config.yaml`.

**Usage:**

```python
from pathlib import Path
from src.config import Config
from src.nlp import LanguageDetector

# Load configuration
config = Config(Path("config/config.yaml"))

# Initialize detector with model path from config
model_path = config.getDataModelsDir() / "fasttext" / "lid.176.ftz"
detector = LanguageDetector(model_path=model_path)

# Detect language
language_code = detector.detect_language("Hello world")  # Returns: "en"

# Get detailed results with confidence
result = detector.detect("Bonjour le monde", k=1)
print(f"Language: {result.language}, Confidence: {result.confidence}")
# Output: Language: fr, Confidence: 0.958

# Get top-k predictions
results = detector.detect("Hello", k=3)
for r in results:
    print(f"{r.language}: {r.confidence:.3f}")
```

**Supported Languages:**

The detector supports 176 languages including:
- Common: English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Arabic
- And 166 more languages

Use `detector.get_supported_languages()` to get the full list of language codes and names.

**Model Information:**

- **Model:** FastText lid.176.ftz (compressed)
- **Size:** 917KB
- **Languages:** 176
- **Source:** [FastText language identification](https://fasttext.cc/docs/en/language-identification.html)
- **Location:** `{data_models_dir}/fasttext/` (configured in config.yaml)

### Transcript Hallucination Detection

After transcription, the system analyzes all transcripts to detect and report any hallucinations that may have occurred during the transcription process. This post-processing analysis uses repetition pattern detection to identify anomalous loops and repetitive sequences.

**How It Works:**
1. **Sliding Window Analysis**: Processes transcripts using a 500-word sliding window with 25% overlap
2. **Pattern Detection**: Uses suffix array algorithm to detect consecutive repetitions
3. **SVM Classification**: Machine learning classifier distinguishes real hallucinations from natural speech patterns
4. **Digest Generation**: Creates a markdown report of all detected hallucinations with timestamps

**Run Hallucination Detection:**
```bash
just analyze-transcripts
```

This command:
- Analyzes all `.srt` transcript files in `data/downloads/transcripts/`
- Generates JSON analysis files in `{channel}/transcript-analysis/`
- Creates a digest report at `data/output/hallucination_digest.md`

**Configuration** (in `config/config.yaml`):
```yaml
hallucination_detection:
  min_window_size: 500      # Sliding window size in words
  overlap_percent: 25.0     # Overlap between windows (0-100)
```

**Output Example:**
```
Total files processed: 1072
Files with hallucinations: 31
Total hallucinations detected: 44

Score Distribution:
  Low Score (11-19): 3 patterns
  Medium Score (21-49): 7 patterns
  High Score (51-99): 9 patterns
  Very High Score (101-∞): 22 patterns
```

**Detection Algorithm:**
- **Repetition Score**: `k × n` where `k` = sequence length, `n` = repetition count
- **Natural Speech Filtering**: Filters out common patterns like "you know", "I think", etc.
- **Threshold**: Patterns with score ≥ 11 are flagged for review
- **Window Tracking**: Records exact word count of analysis window for each detection

**Benefits:**
- **Quality Assurance**: Identify transcription issues across your entire corpus
- **Automated Detection**: No manual review needed for 1000+ files
- **Actionable Reports**: Timestamped hallucinations with context for easy verification
- **Configurable**: Adjust window size and overlap based on your transcript characteristics

### Silence Detection and Removal

The audio extraction script automatically removes silence from videos to improve transcription efficiency and reduce processing time.

**How It Works:**
1. **Pass 1**: Converts video to audio while detecting silence intervals using FFmpeg's `silencedetect` filter
2. **Pass 2**: Extracts only speech segments using FFmpeg's `aselect` filter for precise timestamp alignment
3. **Generates JSON mapping**: Creates `{video}.silence_map.json` with timestamp reconstruction data

**Default Parameters** (configurable in `scripts/convert_to_audio.sh`):
- **Threshold**: -40dB (moderate - removes clear silence while preserving quiet speech)
- **Minimum Duration**: 2 seconds (only removes obvious long pauses)

**Output Files:**
- `data/downloads/audio/{channel}/{video}.wav` - Silence-removed audio
- `data/downloads/metadata/{channel}/{video}.silence_map.json` - Timestamp mapping

**Disable Silence Removal:**
```bash
ENABLE_SILENCE_REMOVAL=false ./scripts/convert_to_audio.sh
# or
ENABLE_SILENCE_REMOVAL=false just extract-audio
```

**Timestamp Reconstruction:**

Use the silence map to convert timestamps from trimmed audio back to original video positions:

```python
import json
from pathlib import Path

# Load silence map
silence_map_path = Path('data/downloads/metadata/AI_Explained/video.silence_map.json')
with open(silence_map_path) as f:
    silence_map = json.load(f)

def trimmed_to_original(trimmed_time: float) -> float:
    """Convert trimmed audio timestamp to original video timestamp."""
    for seg in silence_map['kept_segments']:
        if seg['trimmed_start'] <= trimmed_time <= seg['trimmed_end']:
            offset = trimmed_time - seg['trimmed_start']
            return seg['original_start'] + offset
    raise ValueError(f"Timestamp {trimmed_time} not in any kept segment")

# Example: Convert 125.0s in trimmed audio to original video time
original_time = trimmed_to_original(125.0)
print(f"Trimmed 125.0s → Original {original_time:.2f}s")
```

**JSON Schema:**

```json
{
  "version": "1.0",
  "source_video": "video.mp4",
  "audio_duration_original_seconds": 3600.5,
  "audio_duration_trimmed_seconds": 3200.3,
  "silence_threshold_db": -40,
  "silence_min_duration_seconds": 2.0,
  "silence_intervals": [
    {"start_seconds": 120.5, "end_seconds": 125.3, "duration_seconds": 4.8}
  ],
  "kept_segments": [
    {
      "trimmed_start": 0.0,
      "trimmed_end": 120.5,
      "original_start": 0.0,
      "original_end": 120.5
    },
    {
      "trimmed_start": 120.5,
      "trimmed_end": 570.1,
      "original_start": 125.3,
      "original_end": 574.9
    }
  ],
  "total_silence_removed_seconds": 400.2
}
```

**Benefits:**
- **Faster transcription**: 10-30% time savings from shorter audio
- **Reduced costs**: Less audio to process with cloud transcription services
- **Precise mapping**: `aselect` filter ensures exact timestamp alignment with detected intervals
- **Video referencing**: Accurate reconstruction of original video timestamps for topic segmentation

### Topic Segmentation: Token Usage Monitoring

The topic segmentation pipeline includes proactive token usage monitoring to prevent context window overflow and silent failures during LLM API calls.

**How It Works:**
- **Pre-flight validation**: Before each LLM API call (agent and critic), the system counts tokens using tiktoken
- **Threshold enforcement**: Raises `ContextWindowExceededError` when token usage exceeds the configured threshold
- **Observability**: Logs token count and percentage for every API call for monitoring

**Configuration** (in `config/config.yaml`):
```yaml
topic_segmentation:
  agent_llm:
    context_window: 262144              # Model's maximum context window
    context_window_threshold: 90        # Raise error at 90% usage (0-100)

  critic_llm:
    context_window: 262144
    context_window_threshold: 90
```

**Example Output:**
```
[Agent] Token count: 185,432 tokens (70.7% of context window)
[Agent] Calling LLM API...
```

**When Threshold Exceeded:**
```
[Agent] ✗ Token validation failed: Token usage (250,000 tokens) exceeds 90% threshold
(235,929 tokens) of context window (262,144 tokens). Current usage: 95.4%
```

**Benefits:**
- **Early failure detection**: Catch issues before wasting time on API calls that will fail
- **Clear diagnostics**: Know exactly why processing failed and by how much
- **Cost savings**: Avoid wasted API calls on paid services (Claude API, GPT-4, etc.)
- **Visibility**: Track token usage trends across transcripts to identify problematic videos
- **Configurable**: Tune threshold per deployment (0-100%) based on your needs

**Technical Details:**
- Uses tiktoken library with `o200k_base` encoding (GPT-4o compatible)
- Runs locally with no external API costs
- Compatible with existing error handling (ValueError → SegmentationResult with success=False)
- Comprehensive test coverage with 44 tests for validation logic

## Troubleshooting

### Browser Cookie Authentication

The video downloader uses `yt-dlp` with browser cookies for authentication with YouTube. The browser selection is controlled by the `BROWSER` environment variable.

**Default Configuration:**
- **Location**: `scripts/config.sh:96`
- **Default value**: `chrome`

**Changing the Browser:**

If you need to use a different browser (Firefox, Safari, Edge, etc.), you can override the default in two ways:

1. **Per-command override:**
   ```bash
   BROWSER=firefox just download-videos
   ```

2. **Session-wide override:**
   ```bash
   export BROWSER=firefox
   just download-videos
   ```

**Supported Browsers:**
- `chrome` (default)
- `firefox`
- `safari`
- `edge`
- `brave`
- `opera`
- Any browser supported by yt-dlp's `--cookies-from-browser` option

**Common Issues:**
- If video downloads fail with authentication errors, ensure the specified browser is installed and you're logged into YouTube in that browser
- If using a browser profile, cookies must be accessible from the default profile
- On macOS, you may need to grant Terminal access to the browser's data in System Preferences → Security & Privacy

### Missing Metadata Files (info.json)

The transcription pipeline requires a `.info.json` metadata file for every audio file it processes. These files are normally downloaded by `yt-dlp` alongside the video and then moved into `data/downloads/metadata/<channel>/video/` by `scripts/move-metadata.sh`. If metadata is missing for a file, transcription fails with:

```
🚨 ERROR: Metadata file not found: <path>/<title> [<video_id>].info.json
```

**Diagnose which files are affected:**

```bash
# Scan a channel's audio directory and report any WAV files that have no matching info.json
audio_dir="data/downloads/audio/<channel>"
meta_dir="data/downloads/metadata/<channel>/video"
for wav in "$audio_dir"/*.wav; do
    base=$(basename "$wav" .wav)
    [ -f "$meta_dir/$base.info.json" ] || echo "MISSING: $base"
done
```

**Re-fetch missing metadata:**

Use the `fetch-video-metadata` helper to download `.info.json` for one or more video IDs without re-downloading the video itself. The helper preserves the existing audio file stem so the transcription pipeline can find the metadata.

```bash
just fetch-video-metadata <channel_name> <video_id> [<video_id> ...]
```

Example:
```bash
just fetch-video-metadata Y_Combinator DOez-RwJ7mg lJausFj_Dto
```

The helper:
- Reads all paths (audio dir, metadata dir, video subdir) from `config.yaml`
- Looks up the existing WAV file stem in `data/downloads/audio/<channel>/` for each video ID
- Writes `<stem>.info.json` to `data/downloads/metadata/<channel>/video/`
- Fails fast on any video ID it cannot resolve to an existing WAV file

Underlying scripts:
- `scripts/fetch-video-metadata.py` — CLI wrapper that resolves paths and stems from config
- `scripts/fetch-video-metadata.sh` — runs `yt-dlp --skip-download --write-info-json` with browser cookies (sources `scripts/config.sh`)

## Development

### Development Guidelines

For development guidelines and rules, see [AGENTS.md](AGENTS.md).

### Testing

After every change to the code, tests must be executed:

```bash
just test
```

Always verify the program runs correctly:

```bash
just run
```

### Code Quality

The project includes comprehensive code quality checks:

- **Linting & Formatting**: `ruff` for code style and formatting
- **Type Checking**: `mypy` and `pyright` for static type analysis
- **Security**: `bandit` for security vulnerability scanning
- **Dependencies**: `deptry` for dependency hygiene
- **Spelling**: `codespell` for spell checking
- **Static Analysis**: `semgrep` for pattern-based security checks

Run all checks:

```bash
just ci
```

### AI-Powered Code Quality Checks

The project includes AI-powered code quality tools that analyze code using local LLMs:

#### Unit Test Quality Detection
Detects fake/trivial unit tests that don't provide real test coverage:

```bash
just ai-review-unit-tests          # With caching
just ai-review-unit-tests-nocache  # Force re-scan all files
```

- **Cache**: `.cache/unit_test_hashes.json`
- **Report**: `reports/fake_test_report.md`
- **Model**: Local LLM via LM Studio

#### Shell Script Environment Variable Detection
Detects shell scripts that rely on environment variables not passed as CLI arguments or read from files:

```bash
just ai-review-shell-scripts          # With caching
just ai-review-shell-scripts-nocache  # Force re-scan all files
```

- **Cache**: `.cache/shell_script_hashes.json`
- **Report**: `reports/shell_env_var_violations.md`
- **Model**: Local LLM via LM Studio
- **Target**: All `.sh` files in `scripts/` directory

**Violation Rules:**
- Scripts **FAIL** if they use environment variables that are NOT:
  1. Passed as command line arguments to the script, OR
  2. Read from a configuration file by the script (e.g., `source config.sh`)
- Standard system env vars (`HOME`, `PATH`, `USER`) are acceptable
- Detects violating variables and provides actionable recommendations

#### Run All AI Checks

Run all AI-based CI checks together:

```bash
just ci-ai          # Verbose output
just ci-ai-quiet    # Quiet mode (only errors)
```

### Analysis Reports

The project includes comprehensive analysis reports in `reports/analysis/` that document findings from AI-powered code quality checks and model performance evaluations.

#### Shell Script Analysis Report

**Purpose:** Identify shell scripts that don't follow project rules, specifically scripts that use environment variables without properly passing them as command-line arguments or reading them from configuration files.

**Background:** As part of maintaining code quality and preventing hidden dependencies, the project enforces a rule that all shell scripts must explicitly handle their configuration. Scripts that rely on undeclared environment variables create maintenance issues and make it difficult to understand script dependencies.

**Report Location:** `reports/analysis/shell_script_analysis/`

**Key Findings:**
- Identifies violating scripts that use environment variables incorrectly
- Provides actionable recommendations for fixing each violation
- Helps maintain explicit configuration management across the codebase

**Related Command:** `just ai-review-shell-scripts`

#### Model Performance Analysis Report

**Purpose:** Determine the most efficient LLM model for local development and local LLM serving on a laptop, balancing speed, accuracy, and memory usage.

**Background:** The project uses local LLMs (via LM Studio) for AI-powered code quality checks. Selecting the right model is critical for developer productivity - a model that's too slow will bottleneck the development workflow, while a model that's too inaccurate will produce unreliable results. This analysis evaluated 10 different models on a shell script classification task to identify the optimal model for local development.

**Report Location:** `reports/analysis/model_performance/`

**Key Findings:**
- Comparative analysis of 10 models on shell script classification
- Performance metrics: F1 score (accuracy), execution time (speed), model size (memory)
- Pareto frontier analysis identifying optimal speed/accuracy trade-offs
- Model recommendations for different use cases:
  - **Speed-first with acceptable accuracy**: Fastest model with F1 ≥ 0.90
  - **Accuracy-first**: Highest F1 score regardless of speed
  - **Size efficiency**: Best accuracy per GB of memory
  - **Balanced**: Top 3 models on the Pareto frontier

**Interactive Notebook:** `notebooks/shellscript_analyzer/shellscript_analyzer.ipynb`

**Visualizations:** `notebooks/shellscript_analyzer/gfx/`

These reports help developers make informed decisions about tooling configuration and understand the trade-offs involved in local LLM deployment for development workflows.

## Configuration

The system is configured via `config/config.yaml`. The configuration defines:

- **Channels**: List of YouTube channels to monitor
  - Each channel has a URL, name, source language (`language`), and optional metadata
  - The `language` field specifies the source language (e.g., "en", "de", "ja")
  - Channels can use structured format (`category` + `description`/`what_you_get`) or flexible format (`vibe`)

- **Hallucination Detection**: Sliding window analysis parameters
  - `min_window_size`: Window size in words (default: 500)
  - `overlap_percent`: Overlap between windows (default: 25.0)

- **Topic Segmentation**: LLM configuration for topic analysis
  - Agent and critic LLM settings
  - Context window thresholds
  - Retry limits

- **Defaults**: Centralized default parameter values
  - `encoding_name`: Tiktoken encoding for token counting (default: `o200k_base`)
  - `repetition_min_k`: Minimum phrase length for repetition detection (default: 1)
  - `repetition_min_repetitions`: Minimum consecutive repetitions (default: 5)
  - `detect_min_k`: Default min_k for detect() method (default: 3)

### Configuration Philosophy

The project follows strict configuration principles:

1. **No environment variables**: Never use environment variables for configuration. All config parameters must be loaded through `config.py` from `config.yaml`.

2. **No default parameters**: All configuration values must be explicitly provided at call sites. The `defaults` section in `config.yaml` serves as the single source of truth for default values, which are accessed via the `Config` class getter methods:

```python
from src.config import Config

config = Config(config_path)
encoding = config.getEncodingName()          # Returns "o200k_base"
min_k = config.getRepetitionMinK()           # Returns 1
min_reps = config.getRepetitionMinRepetitions()  # Returns 5
```

This approach ensures:
- **Explicit configuration**: No hidden defaults in function signatures
- **Single source of truth**: All defaults defined in one place (config.yaml)
- **Easy modification**: Change defaults by editing config.yaml, not code
- **Clear dependencies**: Call sites explicitly show what configuration they use

See `config/config.yaml` for the current configuration and `config/config.yaml.template` for all available options.

## HTML Newspaper Frontend

The project includes a Nuxt-based newspaper generator that creates a static HTML site styled like a 1950s New York Times newspaper.

### Frontend Structure

```
frontend/newspaper/
├── nuxt.config.ts          # Nuxt config with static generation
├── package.json            # Node.js dependencies
├── app.vue                 # App entry point
├── assets/css/
│   └── newspaper.css       # All newspaper styles
├── components/
│   ├── Masthead.vue        # Newspaper header
│   ├── HeroSection.vue     # Featured article with image
│   ├── ArticleCard.vue     # Reusable article card (normal/large)
│   ├── SidebarArticle.vue  # Compact sidebar article
│   ├── BriefItem.vue       # Single news brief
│   └── BriefsSection.vue   # Briefs grid section
├── data/
│   └── articles.js         # All content data in one place
├── layouts/
│   └── default.vue         # Default page layout
└── pages/
    └── index.vue           # Homepage
```

### Frontend Setup

Navigate to the frontend directory and install dependencies:

```bash
cd frontend/newspaper
npm install
```

### Frontend Development

Start the development server on `http://localhost:3000`:

```bash
npm run dev
```

### Generate Static Newspaper

Generate a static version of the newspaper site:

```bash
npm run generate
```

The static files will be output to `.output/public/`. This folder can be deployed to any static hosting service (Netlify, Vercel, GitHub Pages, etc.).

### Preview Production Build

Preview the production build locally:

```bash
npm run preview
```

### Updating Newspaper Content

Edit `frontend/newspaper/data/articles.js` to update newspaper content. The data structure includes:

- `heroArticle` - Main featured article with image, headline, byline, and paragraphs
- `featuredArticles[]` - Grid of top stories (with optional images)
- `secondaryMain` - Secondary featured article
- `sidebarArticles[][]` - Sidebar article lists
- `briefsColumns[]` - News briefs organized by section (National, International, Business, Arts & Culture)

### Key Features

- **Authentic newspaper design** with gothic typography and classic layout
- **Fully responsive** - works on desktop, tablet, and mobile
- **Modular components** - all components are reusable with props
- **Data-driven content** - all articles defined in one central file
- **Static site generation** - deploy the generated site anywhere
- **Zero runtime dependencies** - fully static HTML/CSS/JS output

### Automated Newspaper Generation

Use the `just newspaper-generate` command to automatically build the static newspaper website:

```bash
just newspaper-generate
```

This command will:
1. Check if `data/input/newspaper/articles.js` exists (generated by your Python pipeline)
2. Copy the articles data to the Nuxt frontend
3. Install npm dependencies (if not already installed)
4. Run `npm run generate` to build the static site
5. Copy the generated files from `frontend/newspaper/.output/public/` to `data/output/newspaper/`

The final static website will be available at `data/output/newspaper/` and can be deployed to any static hosting service.

### Development Server

For development and testing, use the `just newspaper-serve` command to run the Nuxt development server:

```bash
just newspaper-serve
```

This will:
1. Copy the articles data to the frontend
2. Install npm dependencies (if needed)
3. Start the development server at `http://localhost:3000`

The development server includes hot-reloading, so changes to the frontend code will be reflected immediately in the browser.

## Project Status

This project is in active development. Current implementation status:

- ✅ Configuration loading (`src/config.py`)
- ✅ Basic project structure
- ✅ Video downloading pipeline (`scripts/yt-downloader.sh`)
- ✅ Audio extraction pipeline (`scripts/convert_to_audio.sh`)
- ✅ Multi-language transcription pipeline with MLX Whisper (`scripts/transcribe_audio.sh`)
  - ✅ Language grouping for optimized processing
  - ✅ Automatic translation of non-English content to English
  - ✅ medium.en model for English, medium model for other languages
- ✅ Transcript hallucination detection (`scripts/transcript-hallucination-detection.py`)
- ✅ Video archiving and cleanup (`scripts/archive-videos.sh`)
- ✅ HTML newspaper frontend (Nuxt-based, `frontend/newspaper/`)
- 🚧 Topic segmentation
- 🚧 Article generation
- 🚧 Python-to-frontend data pipeline (transform topic data to articles.js format)
- 🚧 Topic ordering based on relevancy ("interests")

__Known Issues / TODO__

- 🐛 Fix hallucination detection during hallucination removal
  - The removal script extracts text using timestamps from the detection JSON, but the timestamps only point to where the hallucination is located (specific subtitle entries), not the full sliding window that was analyzed during detection
  - This causes the suffix array algorithm to find different pattern lengths (e.g., 13-word vs 14-word patterns) when run on smaller text windows
  - Solution: Use the `window_text` field from the JSON directly for validation instead of reconstructing from timestamps

__Potential Additions__

- 🤔 Source linking (video timestamps)
- 🤔 Video image extraction
- 🤔 Article image embedding
- 🤔 Inline Audio or Video player with auto skip function

### Known Security Issues

The following vulnerabilities are intentionally ignored in dependency audits:

#### nbconvert 7.16.6 - GHSA-xm59-rqc7-hhvf

- **Status**: ⚠️ IGNORED (no fix available)
- **Type**: Windows-specific local privilege escalation via PDF conversion
- **Risk Level**: LOW
- **Rationale**:
  - Dev-only dependency (Jupyter notebooks, not production code)
  - Windows-specific vulnerability (development is on macOS)
  - No patch available from nbconvert maintainers
  - Risk is negligible in current usage context
- **Review Task**: Check quarterly for upstream fix from [nbconvert releases](https://github.com/jupyter/nbconvert/releases)
- **Next Review**: 2026-04-07
- **Configured in**: `justfile` line 440 (`--ignore-vuln GHSA-xm59-rqc7-hhvf`)

#### protobuf 5.29.5 - GHSA-7gcm-g887-7qv7

- **Status**: IGNORED (no fix available)
- **Type**: DoS via uncontrolled recursion in `json_format.ParseDict()` with nested `Any` messages
- **Risk Level**: LOW
- **Rationale**:
  - Requires attacker-controlled protobuf JSON input with deeply nested `Any` messages
  - This project does not parse untrusted protobuf JSON from external sources
  - Transitive dependency (required by autogen-core, googleapis-common-protos, opentelemetry-proto)
  - No patch available - all versions up to 6.33.4 are affected
- **Review Task**: Check for upstream fix from [protobuf releases](https://github.com/protocolbuffers/protobuf/releases)
- **Next Review**: 2026-04-24
- **Configured in**: `justfile` line 440 (`--ignore-vuln GHSA-7gcm-g887-7qv7`)

### Video Processing Features

- **Multi-language Support**: Automatic translation from 100+ languages to English
- **Multiple Transcript Formats**: Generates .txt, .srt, .vtt, .tsv, and .json files
- **Idempotent Operations**: All scripts skip already-processed files
- **Channel-based Organization**: Files organized by YouTube channel
- **Language Grouping**: Processes channels by language to minimize model switching
- **Efficient Processing**: Uses all CPU cores for audio conversion
- **Apple Silicon Optimized**: MLX Whisper leverages Metal acceleration

## License

[Add license information here]

## Developer Notes

### Transcription Optimization Journey

This section documents what we tried and what we chose to reduce hallucinations and improve processing efficiency.

#### Problem Discovery

Initial transcription with `large-v3` had two issues:

1. **Hallucination**: At ~02:58 in test videos, repetitive loops appeared:
   ```
   "So we need to understand the different perspectives of the different models.
   And we need to understand the different ways that the models can interact..."
   [repeated endlessly]
   ```

2. **Slow Processing**: Transcribing silence added time without value.

#### Solution 1: Model Switch (large-v3 → medium.en)

**What We Tried:**
- large-v3 (for maximum accuracy)
- Observed hallucinations during silence

**What We Chose:** medium.en

**Why:**
- large-v3 hallucinates 4x more (WER: 53.4 vs 12.7)
- large-v3 generates phantom text during silence
- medium.en: 99.3% accuracy on technical terms
- 769M params vs 2.3B = 2-3x faster
- Specialized for English-only

**Results:**
- ✅ Eliminated hallucination loops
- ✅ 2-3x faster
- ✅ Better AI/ML terminology

#### Solution 2: YouTube Metadata Prompting

**What We Tried:**
- Generic prompt: "This is a technical AI discussion"
- No prompt

**What We Chose:** Extract title + description from metadata

**Why:**
- Videos already have topic summaries
- Context helps with technical terms
- Whisper benefits from domain hints

**Example:**
```
This is a YouTube video with the title: Meta just did the thing.
Description: The latest AI News. Learn about LLMs, Gen AI...
```

**Results:**
- ✅ Better channel-specific jargon recognition
- ✅ Improved domain terminology

#### Solution 3: Anti-Hallucination Parameters

**What We Tried:**
- Default parameters
- Disabling `condition_on_previous_text` (lost context)
- Aggressive `logprob_threshold` (segments failed)

**What We Chose:**
```bash
--language en                           # Skip detection, faster
--hallucination-silence-threshold 2.0   # Retry after silence
--compression-ratio-threshold 2.0       # Catch gibberish (vs default 2.4)
--condition-on-previous-text True       # Keep context
```

**Why:**
- Balance between quality and reliability
- Automatic recovery without failing segments

**Results:**
- ✅ Auto-recovery from hallucinations
- ✅ Maintained context continuity

#### Solution 4: Silence Removal

**What We Tried:**
- No silence removal
- -30dB, -40dB, -50dB thresholds
- 0.5s, 1s, 2s, 3s minimum durations

**What We Chose:** -40dB threshold, >1s minimum

**Why:**
- Hallucinations increase with silence >30s
- Silence adds processing time, no transcription value
- -40dB: removes clear silence, keeps quiet speech
- 1s: removes pauses, keeps natural speech

**Results:**
- ✅ 10-30% faster transcription
- ✅ Reduced hallucination risk
- ✅ Timestamp mapping preserved

**Trade-offs:**
- Need silence maps for timestamp reconstruction

### Results Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Hallucination Loops | Frequent | None | 100% |
| WER (Median) | ~53.4 | ~12.7 | 76% ↓ |
| Processing Speed | Baseline | 2-3x faster | 200-300% ↑ |
| Transcription Time | Baseline | 10-30% faster | Via silence removal |
| Technical Terms | Good | 99.3% | Excellent |

### Key Learnings

1. Bigger ≠ Better: large-v3 multilingual training degraded English vs medium.en
2. Context Matters: YouTube metadata improved domain accuracy
3. Silence is Dangerous: Removing it improved speed AND quality
4. Balance is Critical: Aggressive settings caused more problems

### References

- [Deepgram Whisper-v3 Study](https://deepgram.com/learn/whisper-v3-results)
- [Whisper GitHub](https://github.com/openai/whisper)
- [Hallucination Discussion](https://github.com/openai/whisper/discussions/678)
- [Model Guide](https://whisper-api.com/blog/models/)
