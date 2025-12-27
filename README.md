# Agentic News Generator

An AI-powered news generator that creates news articles using autonomous agents powered by Claude.

## Repository Structure

```
agentic-news-generator/
├── pyproject.toml          # Project dependencies and metadata
├── justfile                # Build and run commands
├── CLAUDE.md               # AI development rules
├── README.md               # This file
├── src/                    # Source code
│   └── main.py            # Main entry point
├── scripts/                # Utility scripts
├── prompts/                # LLM prompt templates
└── data/                   # Data files
    ├── input/             # Input data files
    └── output/            # Generated output files
```

### Directory Descriptions

- **src/** - All source code for the news generator
- **scripts/** - Utility scripts for data processing and maintenance
- **prompts/** - LLM prompt templates for news generation
- **data/input/** - Input data files (topics, sources, etc.)
- **data/output/** - Generated news articles and outputs

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager installed
- [just](https://github.com/casey/just) command runner installed

## Setup

Initialize the project by creating necessary directories and installing dependencies:

```bash
just init
```

This will:
- Create all required directories
- Set up a virtual environment using `uv`
- Install all dependencies from `pyproject.toml`

## Usage

Run the main program:

```bash
just run
```

To see all available commands:

```bash
just
```

## Development

For development guidelines and rules, see [CLAUDE.md](CLAUDE.md).

## Cleaning Up

Remove generated files and cache:

```bash
just clean
```
