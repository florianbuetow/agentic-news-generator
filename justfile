# Load environment variables from .env file
set dotenv-load := true

# Default recipe: show available commands
_default:
    @just --list

# Show help information
help:
    @echo ""
    @clear
    @echo ""
    @printf "\033[0;34m=== agentic-news-generator ===\033[0m\n"
    @echo ""
    @echo "Available commands:"
    @just --list
    @echo ""

# Initialize the development environment
init:
    @echo ""
    @printf "\033[0;34m=== Initializing Development Environment ===\033[0m\n"
    @mkdir -p reports/coverage
    @mkdir -p reports/security
    @mkdir -p reports/pyright
    @mkdir -p reports/deptry
    @mkdir -p data/downloads/metadata
    @mkdir -p data/input/newspaper
    @mkdir -p data/output/newspaper
    @mkdir -p .cache
    @echo "Installing Python dependencies..."
    @uv sync --all-extras
    @echo "Installing frontend dependencies..."
    @cd frontend/newspaper && npm install
    @printf "\033[0;32m✓ Development environment ready\033[0m\n"
    @echo ""

# Run the main application
run:
    @echo ""
    @printf "\033[0;34m=== Running Application ===\033[0m\n"
    @uv run src/main.py
    @echo ""

# Run the complete pipeline
all:
    @just ci-quiet
    -@just download-videos
    @just extract-audio
    @just transcribe
    @just archive-videos
    @just analyze-transcripts-hallucinations
    @just transcripts-remove-hallucinations

# Download YouTube videos from channels in config.yaml
download-videos:
    @echo ""
    @printf "\033[0;34m=== Downloading YouTube Videos ===\033[0m\n"
    @uv run scripts/yt-downloader.py
    @echo ""
    @printf "\033[0;34m=== Moving Metadata Files ===\033[0m\n"
    @bash scripts/move-metadata.sh
    @echo ""

# Convert downloaded videos to WAV audio files
extract-audio:
    @echo ""
    @printf "\033[0;34m=== Converting Videos to Audio ===\033[0m\n"
    @bash scripts/convert_to_audio.sh
    @echo ""

# Transcribe audio files to text
transcribe:
    #!/usr/bin/env bash
    set +e  # Don't exit on error
    echo ""
    printf "\033[0;34m=== Transcribing Audio Files ===\033[0m\n"
    bash scripts/transcribe_audio.sh
    transcribe_exit_code=$?
    echo ""
    printf "\033[0;34m=== Moving Transcript Metadata ===\033[0m\n"
    bash scripts/move-transcript-metadata.sh
    echo ""
    # Exit with the original transcription exit code
    exit $transcribe_exit_code

# Archive processed videos
archive-videos:
    @echo ""
    @printf "\033[0;34m=== Archiving Processed Videos ===\033[0m\n"
    @bash scripts/archive-videos.sh
    @echo ""

# Analyze transcripts for hallucinations
analyze-transcripts-hallucinations:
    @echo ""
    @printf "\033[0;34m=== Analyzing Transcripts for Hallucinations ===\033[0m\n"
    @uv run scripts/transcript-hallucination-detection.py
    @echo ""
    @printf "\033[0;34m=== Creating Transcript Hallucination Digest ===\033[0m\n"
    @uv run scripts/create-hallucination-digest.py
    @echo ""
    @printf "\033[0;32m✓ Digest created: data/output/hallucination_digest.md\033[0m\n"
    @echo ""

# Remove hallucinations from transcripts using LLM cleaning
transcripts-remove-hallucinations:
    @echo ""
    @printf "\033[0;34m=== Removing Hallucinations from Transcripts ===\033[0m\n"
    @uv run python scripts/transcript-hallucination-removal.py
    @echo ""

# Show processing status of downloads
status:
    @echo ""
    @uv run scripts/status.py
    @echo ""

# Launch Jupyter notebook server
notebooks:
    #!/usr/bin/env bash
    echo ""
    printf "\033[0;34m=== Launching Jupyter Notebook Server ===\033[0m\n"
    echo ""

    # Determine OS and set open command
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OPEN_CMD="open"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OPEN_CMD="xdg-open"
    else
        OPEN_CMD="echo"
        echo "Warning: Unknown OS, cannot auto-open browser"
    fi

    # Open browser after 5 seconds in background
    (sleep 5 && $OPEN_CMD "http://localhost:8888/") &

    # Launch notebook server in foreground
    uv run jupyter notebook --no-browser --NotebookApp.token='' --NotebookApp.password='' notebooks/
    echo ""

# Compile markdown articles into articles.js
compile-articles:
    @echo ""
    @printf "\033[0;34m=== Compiling Articles ===\033[0m\n"
    @uv run scripts/compile_articles.py
    @echo ""

# Generate static newspaper website
newspaper-generate:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Generating Newspaper Website ===\033[0m\n"

    # Check if markdown articles exist
    if [ ! -d "data/input/newspaper/articles" ] || [ -z "$(ls -A data/input/newspaper/articles/*.md 2>/dev/null)" ]; then
        printf "\033[0;31m✗ Error: No markdown articles found in data/input/newspaper/articles/\033[0m\n"
        echo "  Please generate the articles first"
        exit 1
    fi

    # Preprocess markdown articles (extract YAML frontmatter only)
    echo "Preprocessing markdown articles..."
    uv run scripts/preprocess_articles.py

    # Install npm dependencies if needed
    if [ ! -d "frontend/newspaper/node_modules" ]; then
        echo "Installing npm dependencies..."
        cd frontend/newspaper && npm install && cd ../..
    fi

    # Generate static site
    echo "Generating static site..."
    cd frontend/newspaper && npm run generate && cd ../..

    # Clear output directory and copy generated files
    echo "Copying generated site to data/output/newspaper/..."
    rm -rf data/output/newspaper/*
    cp -r frontend/newspaper/.output/public/* data/output/newspaper/

    printf "\033[0;32m✓ Newspaper website generated successfully\033[0m\n"
    echo "  Output: data/output/newspaper/"
    echo ""

# Run newspaper development server
newspaper-serve:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Starting Newspaper Development Server ===\033[0m\n"

    # Check if markdown articles exist
    if [ ! -d "data/input/newspaper/articles" ] || [ -z "$(ls -A data/input/newspaper/articles/*.md 2>/dev/null)" ]; then
        printf "\033[0;31m✗ Error: No markdown articles found in data/input/newspaper/articles/\033[0m\n"
        echo "  Please generate the articles first"
        exit 1
    fi

    # Check if port 3000 is available
    if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        printf "\033[0;31m✗ Error: Port 3000 is already in use\033[0m\n"
        echo "  Please stop the service using port 3000 and try again"
        echo "  You can find the process with: lsof -i :3000"
        exit 1
    fi

    # Preprocess markdown articles (extract YAML frontmatter only)
    echo "Preprocessing markdown articles..."
    uv run scripts/preprocess_articles.py

    # Install npm dependencies if needed
    if [ ! -d "frontend/newspaper/node_modules" ]; then
        echo "Installing npm dependencies..."
        cd frontend/newspaper && npm install && cd ../..
    fi

    # Start development server in background
    echo ""
    printf "\033[0;32m✓ Starting development server at http://localhost:3000\033[0m\n"
    echo ""
    cd frontend/newspaper && npm run dev &
    DEV_PID=$!

    # Wait for server to start, then open browser
    sleep 3
    open http://localhost:3000

    # Wait for dev server to exit (allows Ctrl+C)
    wait $DEV_PID

# Clean up generated newspaper files
newspaper-destroy:
    @echo ""
    @printf "\033[0;34m=== Cleaning Up Generated Newspaper Files ===\033[0m\n"
    @echo "Removing generated newspaper output..."
    @rm -rf data/output/newspaper/*
    @echo "Removing frontend build artifacts..."
    @rm -rf frontend/newspaper/.output
    @rm -rf frontend/newspaper/.nuxt
    @printf "\033[0;32m✓ Newspaper files cleaned up\033[0m\n"
    @echo ""

# Destroy the virtual environment and frontend artifacts
destroy:
    @echo ""
    @printf "\033[0;34m=== Destroying Development Environment ===\033[0m\n"
    @echo "Removing Python virtual environment..."
    @rm -rf .venv
    @echo "Removing frontend dependencies and cache..."
    @rm -rf frontend/newspaper/node_modules
    @rm -rf frontend/newspaper/package-lock.json
    @rm -rf frontend/newspaper/.nuxt
    @rm -rf frontend/newspaper/.output
    @printf "\033[0;32m✓ Development environment destroyed\033[0m\n"
    @echo ""

# Check code style and formatting (read-only)
code-style:
    @echo ""
    @printf "\033[0;34m=== Checking Code Style ===\033[0m\n"
    @uv run ruff check .
    @echo ""
    @uv run ruff format --check .
    @echo ""
    @printf "\033[0;32m✓ Style checks passed\033[0m\n"
    @echo ""

# Auto-fix code style and formatting
code-format:
    @echo ""
    @printf "\033[0;34m=== Formatting Code ===\033[0m\n"
    @uv run ruff check . --fix
    @echo ""
    @uv run ruff format .
    @echo ""
    @printf "\033[0;32m✓ Code formatted\033[0m\n"
    @echo ""

# Check config.yaml matches template structure
code-config:
    @echo ""
    @printf "\033[0;34m=== Checking Config Structure ===\033[0m\n"
    @uv run scripts/check/config_template.py
    @echo ""

# Run static type checking with mypy
code-typecheck:
    @echo ""
    @printf "\033[0;34m=== Running Type Checks ===\033[0m\n"
    @uv run mypy src/
    @echo ""
    @printf "\033[0;32m✓ Type checks passed\033[0m\n"
    @echo ""

# Run strict type checking with Pyright (LSP-based)
code-lspchecks:
    @echo ""
    @printf "\033[0;34m=== Running Pyright Type Checks ===\033[0m\n"
    @mkdir -p reports/pyright
    @uv run pyright --project pyrightconfig.json > reports/pyright/pyright.txt 2>&1 || true
    @uv run pyright --project pyrightconfig.json
    @echo ""
    @printf "\033[0;32m✓ Pyright checks passed\033[0m\n"
    @echo "  Report: reports/pyright/pyright.txt"
    @echo ""

# Run security checks with bandit
code-security:
    @echo ""
    @printf "\033[0;34m=== Running Security Checks ===\033[0m\n"
    @mkdir -p reports/security
    @uv run bandit -c pyproject.toml -r src -f txt -o reports/security/bandit.txt || true
    @uv run bandit -c pyproject.toml -r src
    @echo ""
    @printf "\033[0;32m✓ Security checks passed\033[0m\n"
    @echo ""

# Check dependency hygiene with deptry
code-deptry:
    @echo ""
    @printf "\033[0;34m=== Checking Dependencies ===\033[0m\n"
    @mkdir -p reports/deptry
    @uv run deptry .
    @echo ""
    @printf "\033[0;32m✓ Dependency checks passed\033[0m\n"
    @echo ""

# Generate code statistics with pygount
code-stats:
    @echo ""
    @printf "\033[0;34m=== Code Statistics ===\033[0m\n"
    @mkdir -p reports
    @uv run pygount src/ tests/ scripts/ prompts/ *.md *.toml --suffix=py,md,txt,toml,yaml,yml --format=summary
    @echo ""
    @uv run pygount src/ tests/ scripts/ prompts/ *.md *.toml --suffix=py,md,txt,toml,yaml,yml --format=summary > reports/code-stats.txt
    @printf "\033[0;32m✓ Report saved to reports/code-stats.txt\033[0m\n"
    @echo ""

# Check spelling in code and documentation
code-spell:
    @echo ""
    @printf "\033[0;34m=== Checking Spelling ===\033[0m\n"
    @uv run codespell src tests scripts prompts *.md *.toml
    @echo ""
    @printf "\033[0;32m✓ Spelling checks passed\033[0m\n"
    @echo ""

# Scan dependencies for known vulnerabilities
code-audit:
    @echo ""
    @printf "\033[0;34m=== Scanning Dependencies for Vulnerabilities ===\033[0m\n"
    @uv run pip-audit --skip-editable --ignore-vuln GHSA-xm59-rqc7-hhvf
    @echo ""
    @printf "\033[0;32m✓ No known vulnerabilities found\033[0m\n"
    @echo ""

# Run Semgrep static analysis
code-semgrep:
    @echo ""
    @printf "\033[0;34m=== Running Semgrep Static Analysis ===\033[0m\n"
    @uv run semgrep --config config/semgrep/ --error src
    @echo ""
    @printf "\033[0;32m✓ Semgrep checks passed\033[0m\n"
    @echo ""

# Run AI-powered fake unit test detector
ai-review-unit-tests:
    @echo ""
    @printf "\033[0;34m=== Reviewing Unit Tests with AI ===\033[0m\n"
    @uv run python tools/fake_test_detector/detect_fake_tests.py
    @echo ""

# Run AI-powered fake unit test detector (clear cache and force re-scan)
ai-review-unit-tests-nocache:
    @echo ""
    @printf "\033[0;34m=== Reviewing Unit Tests with AI (No Cache) ===\033[0m\n"
    @rm -rf .cache/test_file_hashes.json
    @printf "\033[0;33m✓ Cache cleared\033[0m\n"
    @uv run python tools/fake_test_detector/detect_fake_tests.py --no-cache
    @echo ""

# Run unit tests only (fast)
test:
    #!/usr/bin/env bash
    set +e
    echo ""
    printf "\033[0;34m=== Running Unit Tests ===\033[0m\n"
    uv run pytest tests/ -v
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 5 ]; then
        printf "\033[0;33m⚠ No tests found (this is OK)\033[0m\n"
        EXIT_CODE=0
    fi
    echo ""
    exit $EXIT_CODE

# Run unit tests with coverage report and threshold check
test-coverage: init
    @echo ""
    @printf "\033[0;34m=== Running Unit Tests with Coverage ===\033[0m\n"
    @uv run pytest tests/ -v \
        --cov=src \
        --cov-report=html:reports/coverage/html \
        --cov-report=term \
        --cov-report=xml:reports/coverage/coverage.xml \
        --cov-fail-under=80
    @echo ""
    @printf "\033[0;32m✓ Coverage threshold met\033[0m\n"
    @echo "  HTML: reports/coverage/html/index.html"
    @echo ""

# Run ALL validation checks (verbose)
ci:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Running CI Checks ===\033[0m\n"
    echo ""
    just init
    just code-config
    just code-format
    just code-style
    just code-typecheck
    just code-security
    just code-deptry
    just code-spell
    just code-semgrep
    just code-audit
    just test
    just code-lspchecks
    echo ""
    printf "\033[0;32m✓ All CI checks passed\033[0m\n"
    echo ""

# Run ALL validation checks silently (only show output on errors)
ci-quiet:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Running CI Checks (Quiet Mode) ===\033[0m\n"
    TMPFILE=$(mktemp)
    trap "rm -f $TMPFILE" EXIT

    just init > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Init failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Init passed\033[0m\n"

    just code-format > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-format failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-format passed\033[0m\n"

    just code-style > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-style failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-style passed\033[0m\n"

    just code-typecheck > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-typecheck failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-typecheck passed\033[0m\n"

    just code-security > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-security failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-security passed\033[0m\n"

    just code-deptry > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-deptry failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-deptry passed\033[0m\n"

    just code-spell > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-spell failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-spell passed\033[0m\n"

    just code-semgrep > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-semgrep failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-semgrep passed\033[0m\n"

    just code-audit > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-audit failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-audit passed\033[0m\n"

    just test > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Test failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Test passed\033[0m\n"

    just code-lspchecks > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-lspchecks failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-lspchecks passed\033[0m\n"

    echo ""
    printf "\033[0;32m✓ All CI checks passed\033[0m\n"
    echo ""

# Run AI-based CI checks (AI-powered test validation, will grow in the future)
# This pipeline is separate from regular CI and includes AI-assisted code quality checks
ci-ai:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Running AI-Based CI Checks ===\033[0m\n"
    echo ""
    just ai-review-unit-tests-nocache
    echo ""
    printf "\033[0;32m✓ All AI-based CI checks passed\033[0m\n"
    echo ""
