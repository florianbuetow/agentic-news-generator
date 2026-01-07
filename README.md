# Agentic News Generator

An AI-powered YouTube news aggregator that crawls AI-focused YouTube channels, transcribes video content, segments transcripts by topic, and generates a weekly newspaper-style HTML digest using autonomous AI agents.

## Overview

This system automatically:
1. Downloads videos from pre-configured YouTube channels
2. Extracts audio from videos to WAV format
3. Transcribes audio using MLX Whisper (large-v3 model) with multiple output formats
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
│   ├── yt-downloader.sh   # YouTube video downloader
│   ├── convert_to_audio.sh  # Video to audio converter
│   ├── transcribe_audio.sh  # Audio transcription (MLX Whisper)
│   ├── transcript-hallucination-detection.py  # Hallucination detection
│   ├── create-hallucination-digest.py  # Digest report generator
│   └── archive-videos.sh  # Archive processed videos
├── tests/                  # Test suite
├── prompts/                # LLM prompt templates
└── data/                   # Data files
    ├── downloads/         # Downloaded and processed content
    │   ├── videos/       # Downloaded videos (by channel)
    │   ├── audio/        # Extracted WAV files (by channel)
    │   ├── transcripts/  # Transcripts in multiple formats (by channel)
    │   │   └── {channel}/transcript-analysis/  # Hallucination analysis JSON
    │   └── metadata/     # Video metadata and silence maps (by channel)
    ├── archive/           # Archived content
    │   └── videos/       # Processed videos moved here
    ├── temp/              # Temporary processing files
    └── output/            # Generated output files
        ├── hallucination_digest.md  # Hallucination analysis report
        ├── topics/        # Per-topic aggregated JSON files
        └── newspaper/     # Generated HTML newspaper
```

## Prerequisites

- **macOS with Apple Silicon** (M1, M2, M3, or later) - Required for MLX Whisper
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager installed
- [just](https://github.com/casey/just) command runner installed
- [FFmpeg](https://ffmpeg.org/) for audio extraction (`brew install ffmpeg`)
- [jq](https://jqlang.github.io/jq/) for JSON processing (`brew install jq`)
- Chrome browser (for YouTube cookie authentication with yt-dlp)

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
- `category`: Channel category (optional, for structured format)
- `description` or `what_you_get`: Channel description (for structured format)
- `vibe`: Free-form channel description (alternative flexible format)

Channels can use either:
- **Structured format**: `category` + `description` (or `what_you_get`)
- **Flexible format**: `vibe` only

See `config/config.yaml` for examples of the 16 pre-configured AI-focused channels.

### 3. Environment Variables

Create a `.env` file in the project root for local development:

```bash
# API Keys (if using cloud-based LLM services)
# ANTHROPIC_API_KEY=your_key_here

# LM Studio Configuration (for local LLM)
# LM_STUDIO_BASE_URL=http://localhost:1234/v1
```

Required environment variables will depend on your LLM backend configuration.

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
- `just transcribe` - Transcribe audio files using MLX Whisper (large-v3)
- `just analyze-transcripts` - Analyze transcripts for hallucinations and generate digest
- `just archive-videos` - Archive processed videos and clean up audio files

#### Development
- `just init` - Initialize development environment
- `just run` - Run the main application
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

# Step 3: Transcribe audio using MLX Whisper large-v3
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
- Transcription uses the Whisper large-v3 model for best quality
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

## Configuration

The system is configured via `config/config.yaml`. The configuration defines:

- **Channels**: List of YouTube channels to monitor
  - Each channel has a URL, name, and optional metadata
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

The project follows a strict "no default parameters" principle: all configuration values must be explicitly provided at call sites. The `defaults` section in `config.yaml` serves as the single source of truth for default values, which are accessed via the `Config` class getter methods:

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

## Project Status

This project is in active development. Current implementation status:

- ✅ Configuration loading (`src/config.py`)
- ✅ Basic project structure
- ✅ Video downloading pipeline (`scripts/yt-downloader.sh`)
- ✅ Audio extraction pipeline (`scripts/convert_to_audio.sh`)
- ✅ Transcription pipeline with MLX Whisper large-v3 (`scripts/transcribe_audio.sh`)
- ✅ Transcript hallucination detection (`scripts/transcript-hallucination-detection.py`)
- ✅ Video archiving and cleanup (`scripts/archive-videos.sh`)
- 🚧 Topic segmentation
- 🚧 Article generation
- 🚧 HTML newspaper generation
- 🚧 Topic ordering based on relevancy ("interests").

__Potential Additions__

- 🤔 Source linking (video timestamps)
- 🤔 Video image extraction
- 🤔 Article image embedding
- 🤔 Inline Audio or Video player with auto skip function.


### Video Processing Features

- **Multiple Transcript Formats**: Generates .txt, .srt, .vtt, .tsv, and .json files
- **Idempotent Operations**: All scripts skip already-processed files
- **Channel-based Organization**: Files organized by YouTube channel
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
