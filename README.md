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
├── frontend/               # Frontend applications
│   └── newspaper/         # Nuxt-based newspaper generator
│       ├── nuxt.config.ts # Nuxt configuration
│       ├── package.json   # Node.js dependencies
│       ├── components/    # Vue components (Masthead, ArticleCard, etc.)
│       ├── data/          # Content data (articles.js)
│       ├── pages/         # Nuxt pages
│       └── assets/        # Styles and static assets
└── data/                   # Data files
    ├── input/             # Input data files
    │   └── newspaper/     # Newspaper input data
    │       └── articles.js # Generated articles data for Nuxt
    └── output/            # Generated output files
        ├── videos/        # Downloaded video files
        ├── audio/         # Extracted audio files
        ├── transcripts/   # Whisper transcripts (SRT format)
        ├── topics/        # Per-topic aggregated JSON files
        └── newspaper/     # Generated HTML newspaper (static site output)
```

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager installed
- [just](https://github.com/casey/just) command runner installed
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
- `just newspaper-generate` - Generate static newspaper website from articles data
- `just newspaper-serve` - Run newspaper development server at http://localhost:3000
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
- ✅ HTML newspaper frontend (Nuxt-based, `frontend/newspaper/`)
- 🚧 Video downloading pipeline
- 🚧 Transcription pipeline
- 🚧 Topic segmentation
- 🚧 Article generation
- 🚧 Python-to-frontend data pipeline (transform topic data to articles.js format)

## License

[Add license information here]
