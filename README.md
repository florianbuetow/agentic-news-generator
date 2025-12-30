# Agentic News Generator

An AI-powered YouTube news aggregator that crawls AI-focused YouTube channels, transcribes video content, segments transcripts by topic, and generates a weekly newspaper-style HTML digest using autonomous AI agents.

## Overview

This system automatically:
1. Downloads videos from pre-configured YouTube channels
2. Transcribes video content using Whisper (with timestamps)
3. Segments transcripts into topic-based sections using AI analysis
4. Aggregates related topic segments across multiple videos
5. Generates news articles from aggregated content using AI agents
6. Produces a newspaper-style HTML digest

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
│   └── yt-downloader.sh   # YouTube video downloader
├── tests/                  # Test suite
├── prompts/                # LLM prompt templates
└── data/                   # Data files
    ├── input/             # Input data files
    └── output/            # Generated output files
        ├── videos/        # Downloaded video files
        ├── audio/         # Extracted audio files
        ├── transcripts/   # Whisper transcripts (SRT format)
        ├── topics/        # Per-topic aggregated JSON files
        └── newspaper/     # Generated HTML newspaper
```

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager installed
- [just](https://github.com/casey/just) command runner installed
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

- `just init` - Initialize development environment
- `just run` - Run the main application
- `just test` - Run unit tests
- `just test-coverage` - Run tests with coverage report
- `just code-format` - Auto-fix code style and formatting
- `just code-style` - Check code style (read-only)
- `just code-typecheck` - Run type checking with mypy
- `just ci` - Run all validation checks
- `just destroy` - Remove virtual environment

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
- 🚧 Video downloading pipeline
- 🚧 Transcription pipeline
- 🚧 Topic segmentation
- 🚧 Article generation
- 🚧 HTML newspaper generation

## License

[Add license information here]
