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
│   └── config.py         # Configuration loading
├── scripts/                # Utility scripts
│   ├── yt-downloader.sh   # YouTube video downloader
│   ├── convert_to_audio.sh  # Video to audio converter
│   ├── transcribe_audio.sh  # Audio transcription (MLX Whisper)
│   └── archive-videos.sh  # Archive processed videos
├── tests/                  # Test suite
├── prompts/                # LLM prompt templates
└── data/                   # Data files
    ├── downloads/         # Downloaded and processed content
    │   ├── videos/       # Downloaded videos (by channel)
    │   ├── audio/        # Extracted WAV files (by channel)
    │   └── transcripts/  # Transcripts in multiple formats (by channel)
    ├── archive/           # Archived content
    │   └── videos/       # Processed videos moved here
    ├── temp/              # Temporary processing files
    └── output/            # Generated output files
        ├── topics/        # Per-topic aggregated JSON files
        └── newspaper/     # Generated HTML newspaper
```

## Prerequisites

- **macOS with Apple Silicon** (M1, M2, M3, or later) - Required for MLX Whisper
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager installed
- [just](https://github.com/casey/just) command runner installed
- [FFmpeg](https://ffmpeg.org/) for audio extraction (`brew install ffmpeg`)
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

# Step 4: Archive processed videos and clean up audio files
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

See `config/config.yaml` for the current channel configuration.

## Project Status

This project is in active development. Current implementation status:

- ✅ Configuration loading (`src/config.py`)
- ✅ Basic project structure
- ✅ Video downloading pipeline (`scripts/yt-downloader.sh`)
- ✅ Audio extraction pipeline (`scripts/convert_to_audio.sh`)
- ✅ Transcription pipeline with MLX Whisper large-v3 (`scripts/transcribe_audio.sh`)
- ✅ Video archiving and cleanup (`scripts/archive-videos.sh`)
- 🚧 Topic segmentation
- 🚧 Article generation
- 🚧 HTML newspaper generation

### Video Processing Features

- **Multiple Transcript Formats**: Generates .txt, .srt, .vtt, .tsv, and .json files
- **Idempotent Operations**: All scripts skip already-processed files
- **Channel-based Organization**: Files organized by YouTube channel
- **Efficient Processing**: Uses all CPU cores for audio conversion
- **Apple Silicon Optimized**: MLX Whisper leverages Metal acceleration

## License

[Add license information here]
