# =============================================================================
# Justfile Rules (follow these when editing justfile):
#
# 1. Use printf (not echo) to print colors — some terminals won't render
#    colors with echo.
#
# 2. Always add an empty `@echo ""` line before and after each target's
#    command block.
#
# 3. Always add new targets to the help section and update it when targets
#    are added, modified or removed.
#
# 4. Target ordering in help (and in this file) matters:
#    - Setup targets first (init, setup, install, etc.)
#    - Start/stop/run targets next
#    - Code generation / data tooling targets next
#    - Checks, linting, and tests next (ordered fastest to slowest)
#    Group related targets together and separate groups with an empty
#    `@echo ""` line in the help output.
#
# 5. Composite targets that call multiple sub-targets must surface every
#    failure — never silently ignore errors:
#    - Validation/CI/checks composites (e.g. ci, ci-verbose, checks-all) must
#      fail fast: exit 1 on the first error (use `set -e` or
#      `|| { ...; exit 1; }`).
#    - Data pipelines (e.g. url-all, video-all, pipelines-all) must catch each
#      step in an if/else block, continue through the remaining steps, and
#      print a red/green per-step summary at the end, exiting non-zero if any
#      step failed.
#    NEVER prefix a recipe line with `-` to ignore a non-zero exit code; this
#    is enforced by config/semgrep/justfile-no-error-suppression.yml.
#
# 6. Every target must end with a clear short status message:
#    - On success: green (\033[32m) message confirming completion.
#      E.g. printf "\033[32m✓ init completed successfully\033[0m\n"
#    - On failure: red (\033[31m) message indicating what failed, then exit 1.
#      E.g. printf "\033[31m✗ ci failed: tests exited with errors\033[0m\n"
# 7. Targets must be shown in groups separated by empty newlines in the help section.
#    - init/destroy/clean/help on top, ci and other tests on the bottom, between other groups
# =============================================================================


# Load environment variables from .env file
set dotenv-load := true

# Default recipe: show available commands
_default:
    @just help

# Show help information
help:
    @echo ""
    @clear
    @echo ""
    @printf "\033[0;34m=== agentic-news-generator ===\033[0m\n"
    @echo ""
    @printf "\033[0;33mProject:\033[0m\n"
    @printf "  %-38s %s\n" "init" "Initialize the development environment"
    @printf "  %-38s %s\n" "destroy" "Destroy the virtual environment and frontend artifacts"
    @printf "  %-38s %s\n" "check" "Check if all required tools and prerequisites are available"
    @printf "  %-38s %s\n" "help" "Show this help information"
    @echo ""
    @printf "\033[0;33mPipelines:\033[0m\n"
    @printf "  %-38s %s\n" "pipelines-all" "Run url-all then video-all"
    @printf "  %-38s %s\n" "video-all" "Run the complete video pipeline"
    @printf "  %-38s %s\n" "url-all" "Run the full URL pipeline (fetch, download, clean)"
    @printf "  %-38s %s\n" "checks-all" "Run all checks and CI"
    @printf "  %-38s %s\n" "maintenance" "Run all read-only data pipeline health checks"
    @printf "  %-38s %s\n" "stats [hour|day]" "Show processing status (throttle: once per hour/day)"
    @printf "  %-38s %s\n" "stats-mini" "Show condensed processing status (channels with video archive or audio only)"
    @printf "  %-38s %s\n" "totals [hour|day]" "Show processing status with transcript time totals"
    @printf "  %-38s %s\n" "audio-hours" "Count total audio hours from transcripts"
    @printf "  %-38s %s\n" "status" "Check if LM Studio is running and models are loaded"
    @echo ""
    @printf "\033[0;33mVideo-Pipeline:\033[0m\n"
    @printf "  %-38s %s\n" "download-videos [<channel>]" "Download YouTube videos from channels in config.yaml"
    @printf "  %-38s %s\n" "check-video-integrity" "Check video files for corruption"
    @printf "  %-38s %s\n" "filter-videos" "Filter and delete videos shorter than transcription.min_duration"
    @printf "  %-38s %s\n" "extract-audio" "Convert downloaded videos to WAV audio files"
    @printf "  %-38s %s\n" "transcribe" "Transcribe audio files to text"
    @printf "  %-38s %s\n" "archive-videos" "Archive processed videos"
    @printf "  %-38s %s\n" "analyze-transcripts-hallucinations" "Analyze transcripts for hallucinations"
    @printf "  %-38s %s\n" "transcripts-remove-hallucinations" "Remove hallucinations from transcripts using LLM"
    @printf "  %-38s %s\n" "analyze-transcript-languages" "Analyze transcript languages"
    @printf "  %-38s %s\n" "summarize-transcripts [<channel>]" "Summarize cleaned transcripts using LLM"
    @echo ""
    @printf "\033[0;33mURL-Pipeline:\033[0m\n"
    @printf "  %-38s %s\n" "urls-fetch-raindrop" "Fetch Raindrop.io bookmarks into the URL inbox (categorized)"
    @printf "  %-38s %s\n" "urls-download" "Download raw content from inbox URLs (normalize, classify, fetch)"
    @printf "  %-38s %s\n" "urls-cleancontent" "Convert downloaded raw URL content into cleaned Markdown"
    @echo ""
    @printf "\033[0;33mNewspaper-Pipeline:\033[0m\n"
    @printf "  %-38s %s\n" "notebooks" "Launch Jupyter notebook server"
    @printf "  %-38s %s\n" "compile-articles" "Compile markdown articles into articles.js"
    @printf "  %-38s %s\n" "newspaper-generate" "Generate static newspaper website"
    @printf "  %-38s %s\n" "newspaper-serve" "Run newspaper development server"
    @printf "  %-38s %s\n" "newspaper-destroy" "Clean up generated newspaper files"
    @echo ""
    @printf "\033[0;33mAnalytics:\033[0m\n"
    @printf "  %-38s %s\n" "analytics" "Full research digest (index + themes + timeline; no LLM)"
    @printf "  %-38s %s\n" "analytics-index" "Build corpus index from cleaned transcripts + summaries"
    @printf "  %-38s %s\n" "analytics-themes" "Theme frequency + TF-IDF term report"
    @printf "  %-38s %s\n" "analytics-timeline" "Timeline report bucketed by upload date"
    @echo ""
    @printf "\033[0;33mCode Quality:\033[0m\n"
    @printf "  %-38s %s\n" "code-format" "Auto-fix code style and formatting"
    @printf "  %-38s %s\n" "code-style" "Check code style and formatting (read-only)"
    @printf "  %-38s %s\n" "code-config" "Check config.yaml matches template structure"
    @printf "  %-38s %s\n" "code-spell" "Check spelling in code and documentation"
    @printf "  %-38s %s\n" "check-config-syntax" "Validate YAML file syntax across the project"
    @printf "  %-38s %s\n" "code-typecheck" "Run static type checking with mypy"
    @printf "  %-38s %s\n" "code-lspchecks" "Run strict type checking with Pyright (LSP-based)"
    @printf "  %-38s %s\n" "code-security" "Run security checks with bandit"
    @printf "  %-38s %s\n" "code-deptry" "Check dependency hygiene with deptry"
    @printf "  %-38s %s\n" "code-semgrep" "Run Semgrep static analysis"
    @printf "  %-38s %s\n" "code-audit" "Scan dependencies for known vulnerabilities"
    @printf "  %-38s %s\n" "code-stats" "Generate code statistics with pygount"
    @echo ""
    @printf "\033[0;33mAgentic Reviews:\033[0m\n"
    @printf "  %-38s %s\n" "ai-review-unit-tests" "Run AI-powered fake unit test detector"
    @printf "  %-38s %s\n" "ai-review-unit-tests-nocache" "Run AI-powered fake unit test detector (no cache)"
    @printf "  %-38s %s\n" "ai-review-shell-scripts" "Run AI-powered shell script reviewer"
    @printf "  %-38s %s\n" "ai-review-shell-scripts-nocache" "Run AI-powered shell script reviewer (no cache)"
    @echo ""
    @printf "\033[0;33mTools:\033[0m\n"
    @printf "  %-46s %s\n" "find [query]" "Interactively search cleaned transcripts; with query, list matching files"
    @printf "  %-46s %s\n" "search [query]" "Interactively search summary files; with query, list matching files"
    @printf "  %-46s %s\n" "research \"<keywords>\"" "List transcripts & summaries matching comma-separated keywords"
    @printf "  %-46s %s\n" "find-files <video-id>" "Find all files for a video ID across data directories"
    @printf "  %-46s %s\n" "fetch-video-metadata [<channel> <id...>]" "Fetch missing .info.json; no args scans all non-archived videos"
    @printf "  %-46s %s\n" "check-missing-metadata" "Check all channels for WAV files missing .info.json and fetch them"
    @printf "  %-46s %s\n" "fetch-video-thumbnails [<channel> [<id...>]]" "Fetch missing thumbnails (scan all / scan channel / specific IDs)"
    @printf "  %-46s %s\n" "find-empty-transcripts" "List transcript files that are 100 bytes or smaller"
    @printf "  %-46s %s\n" "find-files-without-youtube-id" "Report data files missing a YouTube ID (writes to reports/)"
    @printf "  %-46s %s\n" "find-files-with-filtered-video-ids" "Flag data files whose video ID is in the filter file (read-only)"
    @printf "  %-46s %s\n" "cleanup-plain-filename-duplicates" "Move plain-named duplicate files (no YouTube ID) to backup location"
    @printf "  %-46s %s\n" "clean-empty-files" "Scan for and remove empty files in data folder"
    @printf "  %-46s %s\n" "clean-video-files <VIDEO_ID>" "Delete all files for a YouTube video ID (interactive)"
    @printf "  %-46s %s\n" "disk-free" "Show free disk space for each drive used by config.yaml paths"
    @echo ""
    @printf "\033[0;33mCI & Testing:\033[0m\n"
    @printf "  %-38s %s\n" "test" "Run unit tests only (fast)"
    @printf "  %-38s %s\n" "test-url-ingestion" "Run URL ingestion integration/e2e tests"
    @printf "  %-38s %s\n" "test-coverage" "Run unit tests with coverage report"
    @printf "  %-38s %s\n" "ci" "Run ALL validation checks silently"
    @printf "  %-38s %s\n" "ci-verbose" "Run ALL validation checks (verbose)"
    @printf "  %-38s %s\n" "ci-ai" "Run AI-based CI checks"
    @printf "  %-38s %s\n" "ci-ai-quiet" "Run AI-based CI checks silently"
    @echo ""

# Initialize the development environment
init:
    @echo ""
    @printf "\033[0;34m=== Initializing Development Environment ===\033[0m\n"
    @echo "Creating directories from config.yaml..."
    @bash scripts/init-directories.sh
    @echo "Downloading FastText language detection models..."
    @bash scripts/download-fasttext-models.sh
    @echo "Installing Python dependencies..."
    @uv sync --all-extras
    @echo "Installing frontend dependencies..."
    @cd frontend/newspaper && npm install
    @printf "\033[0;32m✓ Development environment ready\033[0m\n"
    @echo ""

# Check if all required tools and prerequisites are available
check:
    @echo ""
    @printf "\033[0;34m=== Checking System Prerequisites ===\033[0m\n"
    @command -v ffmpeg &> /dev/null && echo "✓ ffmpeg" || (echo "✗ ffmpeg missing (install with: brew install ffmpeg)" && exit 1)
    @command -v jq &> /dev/null && echo "✓ jq" || (echo "✗ jq missing (install with: brew install jq)" && exit 1)
    @command -v just &> /dev/null && echo "✓ just" || (echo "✗ just missing (install with: brew install just)" && exit 1)
    @command -v uv &> /dev/null && echo "✓ uv" || (echo "✗ uv missing (install with: brew install uv)" && exit 1)
    @command -v node &> /dev/null && echo "✓ node" || (echo "✗ node missing (install with: brew install node)" && exit 1)
    @command -v npm &> /dev/null && echo "✓ npm" || (echo "✗ npm missing (install with: brew install node)" && exit 1)
    @printf "\033[0;32m✓ All system prerequisites are installed\033[0m\n"
    @echo ""
    @printf "\033[0;34m=== Checking Virtual Environment ===\033[0m\n"
    @if [ ! -d ".venv" ]; then \
        echo "✗ Virtual environment not found"; \
        echo ""; \
        printf "\033[0;33mPlease run 'just init' to initialize the development environment\033[0m\n"; \
        echo ""; \
        exit 1; \
    fi
    @echo "✓ Virtual environment exists"
    @echo ""
    @printf "\033[0;34m=== Checking Python Tools ===\033[0m\n"
    @uv run mypy --version &> /dev/null && echo "✓ mypy" || (echo "✗ mypy missing (run: just init)" && exit 1)
    @uv run pytest --version &> /dev/null && echo "✓ pytest" || (echo "✗ pytest missing (run: just init)" && exit 1)
    @uv run ruff --version &> /dev/null && echo "✓ ruff" || (echo "✗ ruff missing (run: just init)" && exit 1)
    @uv run pyright --version &> /dev/null && echo "✓ pyright" || (echo "✗ pyright missing (run: just init)" && exit 1)
    @uv run bandit --version &> /dev/null && echo "✓ bandit" || (echo "✗ bandit missing (run: just init)" && exit 1)
    @uv run deptry --version &> /dev/null && echo "✓ deptry" || (echo "✗ deptry missing (run: just init)" && exit 1)
    @uv run codespell --version &> /dev/null && echo "✓ codespell" || (echo "✗ codespell missing (run: just init)" && exit 1)
    @uv run pip-audit --version &> /dev/null && echo "✓ pip-audit" || (echo "✗ pip-audit missing (run: just init)" && exit 1)
    @uv run semgrep --version &> /dev/null && echo "✓ semgrep" || (echo "✗ semgrep missing (run: just init)" && exit 1)
    @uv run pygount --version &> /dev/null && echo "✓ pygount" || (echo "✗ pygount missing (run: just init)" && exit 1)
    @printf "\033[0;32m✓ All Python tools are installed\033[0m\n"
    @echo ""
    @printf "\033[0;32m✓ All required tools and prerequisites are available\033[0m\n"
    @echo ""

# Run the complete video pipeline
video-all:
    #!/usr/bin/env bash
    echo ""
    printf "\033[0;34m=== Running Video Pipeline ===\033[0m\n"
    echo ""
    steps=(
        ci
        download-videos
        check-video-integrity
        filter-videos
        extract-audio
        transcribe
        archive-videos
        analyze-transcripts-hallucinations
        transcripts-remove-hallucinations
        analyze-transcript-languages
        summarize-transcripts
    )
    failed=()
    for step in "${steps[@]}"; do
        if just "$step"; then
            printf "\033[0;32m✓ %s passed\033[0m\n" "$step"
        else
            printf "\033[0;31m✗ %s failed\033[0m\n" "$step"
            failed+=("$step")
        fi
    done
    echo ""
    if [ ${#failed[@]} -eq 0 ]; then
        printf "\033[0;32m✓ video-all completed successfully\033[0m\n"
        echo ""
    else
        printf "\033[0;31m✗ video-all failed: %s\033[0m\n" "${failed[*]}"
        echo ""
        exit 1
    fi

# Run all checks and CI
checks-all:
    @just clean-empty-files
    @just check-video-integrity
    @just filter-videos
    @just analyze-transcripts-hallucinations
    @just transcripts-remove-hallucinations
    @just analyze-transcript-languages
    @just ci

# Run url-all then video-all
pipelines-all:
    #!/usr/bin/env bash
    echo ""
    printf "\033[0;34m=== Running All Pipelines ===\033[0m\n"
    echo ""
    steps=(
        url-all
        video-all
    )
    failed=()
    for step in "${steps[@]}"; do
        if just "$step"; then
            printf "\033[0;32m✓ %s passed\033[0m\n" "$step"
        else
            printf "\033[0;31m✗ %s failed\033[0m\n" "$step"
            failed+=("$step")
        fi
    done
    echo ""
    if [ ${#failed[@]} -eq 0 ]; then
        printf "\033[0;32m✓ pipelines-all completed successfully\033[0m\n"
        echo ""
    else
        printf "\033[0;31m✗ pipelines-all failed: %s\033[0m\n" "${failed[*]}"
        echo ""
        exit 1
    fi

# Run the full URL pipeline: fetch bookmarks from Raindrop, download raw content, clean into Markdown
url-all:
    #!/usr/bin/env bash
    echo ""
    printf "\033[0;34m=== Running URL Pipeline ===\033[0m\n"
    echo ""
    steps=(
        urls-fetch-raindrop
        urls-download
        urls-cleancontent
    )
    failed=()
    for step in "${steps[@]}"; do
        if just "$step"; then
            printf "\033[0;32m✓ %s passed\033[0m\n" "$step"
        else
            printf "\033[0;31m✗ %s failed\033[0m\n" "$step"
            failed+=("$step")
        fi
    done
    echo ""
    if [ ${#failed[@]} -eq 0 ]; then
        printf "\033[0;32m✓ url-all completed successfully\033[0m\n"
        echo ""
    else
        printf "\033[0;31m✗ url-all failed: %s\033[0m\n" "${failed[*]}"
        echo ""
        exit 1
    fi

# Run all read-only data pipeline health checks (no data modified or deleted)
maintenance:
    @just check
    @just check-video-integrity
    @just check-missing-metadata
    @just check-config-syntax
    @just find-empty-transcripts
    @just find-files-without-youtube-id
    @just analyze-transcripts-hallucinations
    @just analyze-transcript-languages
    @printf "\033[0;32m✓ maintenance completed successfully\033[0m\n"

# Download YouTube videos from channels in config.yaml (optional: just download-videos <channel>)
download-videos channel="":
    #!/usr/bin/env bash
    set +e
    mkdir -p reports
    echo ""
    printf "\033[0;34m=== Downloading YouTube Videos ===\033[0m\n"
    uv run scripts/yt-downloader.py {{ if channel == "" { "" } else { "--channel " + channel } }} 2>&1 | tee reports/video-download.log
    download_exit_code=${PIPESTATUS[0]}
    echo ""
    if [ $download_exit_code -ne 0 ]; then
        printf "\033[0;31m✗ download-videos failed: see the error above and reports/video-download.log\033[0m\n"
        echo ""
        exit $download_exit_code
    fi
    printf "\033[0;34m=== Moving Metadata Files ===\033[0m\n"
    bash scripts/move-metadata.sh
    echo ""
    printf "\033[0;34m=== Adding Members-Only Videos to Skip List ===\033[0m\n"
    uv run scripts/parse-and-archive-membersonly.py
    echo ""
    printf "\033[0;32m✓ download-videos completed successfully\033[0m\n"
    echo ""

# Convert downloaded videos to WAV audio files
extract-audio:
    @echo ""
    @printf "\033[0;34m=== Converting Videos to Audio ===\033[0m\n"
    @bash scripts/convert_to_audio.sh
    @echo ""

# Check video files for corruption
check-video-integrity:
    @echo ""
    @printf "\033[0;34m=== Checking Video File Integrity ===\033[0m\n"
    @uv run python scripts/check_video_integrity.py
    @echo ""

# Filter and delete videos shorter than transcription.min_duration (or with no audio)
filter-videos:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Filtering Short / No-Audio Videos ===\033[0m\n"
    if ! uv run python scripts/filter-short-videos.py; then
        printf "\033[0;31m✗ filter-videos failed: filter-short-videos.py errored\033[0m\n"
        exit 1
    fi
    echo ""
    printf "\033[0;34m=== Removing Filtered Files ===\033[0m\n"
    if ! uv run python scripts/remove-filtered-files.py; then
        printf "\033[0;31m✗ filter-videos failed: remove-filtered-files.py errored\033[0m\n"
        exit 1
    fi
    printf "\033[0;32m✓ filter-videos completed successfully\033[0m\n"
    echo ""

# Transcribe audio files to text (optional n: stop after transcribing n files; default all)
transcribe n="":
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Transcribing Audio Files ===\033[0m\n"
    uv run python scripts/transcribe_audio.py {{ if n == "" { "" } else { "--limit " + n } }}
    echo ""
    printf "\033[0;34m=== Cleaning Up Empty Transcripts ===\033[0m\n"
    transcripts_dir=$(uv run python -c "from pathlib import Path; import sys; sys.path.insert(0,'src'); from src.config import Config; c=Config(Path('config/config.yaml')); print(Path('.') / c.get_data_downloads_transcripts_dir())")
    bash scripts/cleanup-empty-transcripts.sh "$transcripts_dir"
    echo ""
    printf "\033[0;34m=== Moving Transcript Metadata ===\033[0m\n"
    bash scripts/move-transcript-metadata.sh

# Archive processed videos
archive-videos:
    @echo ""
    @printf "\033[0;34m=== Archiving Processed Videos ===\033[0m\n"
    @bash scripts/archive-videos.sh
    @echo ""

# Scan for and remove empty files in data folder
clean-empty-files:
    @uv run python scripts/find-and-clean-empty-data-files.py

# Delete all files for a YouTube video ID and optionally clean the channel archive. Requires: VIDEO_ID
clean-video-files VIDEO_ID:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Cleaning Video Files ===\033[0m\n"
    if uv run python scripts/clean-video-files.py {{ VIDEO_ID }}; then
        printf "\033[0;32m✓ clean-video-files completed successfully\033[0m\n"
    else
        printf "\033[0;31m✗ clean-video-files failed\033[0m\n"
        exit 1
    fi
    echo ""

# Show free disk space for each drive used by config.yaml paths
disk-free:
    @echo ""
    @printf "\033[0;34m=== Disk Free Space ===\033[0m\n"
    @uv run scripts/disk-free.py
    @printf "\033[0;32m✓ disk-free completed successfully\033[0m\n"
    @echo ""

# Analyze transcripts for hallucinations
analyze-transcripts-hallucinations:
    @echo ""
    @printf "\033[0;34m=== Analyzing Transcripts for Hallucinations ===\033[0m\n"
    @uv run scripts/transcript-hallucination-detection.py --skip-existing
    @echo ""
    @printf "\033[0;34m=== Creating Transcript Hallucination Digest ===\033[0m\n"
    @uv run scripts/create-hallucination-digest.py
    @echo ""

# Remove hallucinations from transcripts using LLM cleaning
transcripts-remove-hallucinations:
    @echo ""
    @printf "\033[0;34m=== Removing Hallucinations from Transcripts ===\033[0m\n"
    @uv run python scripts/transcript-hallucination-removal.py --skip-existing
    @echo ""

# Analyze transcript languages (run after hallucination removal for accurate detection)
analyze-transcript-languages:
    @echo ""
    @printf "\033[0;34m=== Analyzing Transcript Languages ===\033[0m\n"
    @uv run scripts/transcript-language-analysis.py
    @echo ""

# Summarize cleaned transcripts using LLM (optional: just summarize-transcripts <channel>)
summarize-transcripts channel="":
    @echo ""
    @printf "\033[0;34m=== Summarizing Cleaned Transcripts ===\033[0m\n"
    @uv run python scripts/summarize-transcripts.py {{ if channel == "" { "" } else { "--channel " + channel } }}
    @echo ""

# Fetch Raindrop.io bookmarks into the URL inbox as categorized Category->Subcategory:url lines (only new links; pass --force to re-emit all)
urls-fetch-raindrop *ARGS:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Fetching Raindrop.io Bookmarks into URL Inbox ===\033[0m\n"
    if ! uv run python integrations/raindrop_io/fetchurls-raindrop.py {{ ARGS }}; then
        printf "\033[0;31m✗ urls-fetch-raindrop failed\033[0m\n"
        exit 1
    fi
    printf "\033[0;32m✓ urls-fetch-raindrop completed successfully\033[0m\n"
    echo ""

# Download raw content from inbox URLs (normalize, deduplicate, classify, fetch, write metadata, archive inbox)
urls-download:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Downloading Raw Content from Inbox URLs ===\033[0m\n"
    if ! uv run python scripts/urls-download.py; then
        printf "\033[0;31m✗ urls-download failed\033[0m\n"
        exit 1
    fi
    printf "\033[0;32m✓ urls-download completed successfully\033[0m\n"
    echo ""

# Convert downloaded raw URL content into cleaned Markdown documents
urls-cleancontent:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Cleaning Raw URL Content into Markdown ===\033[0m\n"
    if ! uv run python scripts/urls-cleancontent.py; then
        printf "\033[0;31m✗ urls-cleancontent failed\033[0m\n"
        exit 1
    fi
    printf "\033[0;32m✓ urls-cleancontent completed successfully\033[0m\n"
    echo ""


# Check if LM Studio is running and required models are loaded
status:
    @echo ""
    @uv run scripts/lmstudio_status.py
    @echo ""

# Show processing status of downloads
stats period="":
    #!/usr/bin/env bash
    cache_flag=""
    stats_from=""
    if [[ "{{period}}" == "hour" ]]; then
        stamp="/tmp/stats-hour.stamp"
        now=$(date +%s)
        if [[ ! -f "$stamp" ]] || (( now - $(stat -f "%m" "$stamp") >= 3600 )); then
            touch "$stamp"
        else
            cache_flag="--no-update-cache"
        fi
        stats_from=$(date -r "$(stat -f "%m" "$stamp")" "+%Y-%m-%d %H:%M:%S")
    elif [[ "{{period}}" == "day" ]]; then
        stamp="/tmp/stats-day.stamp"
        if [[ ! -f "$stamp" || "$(stat -f "%Sm" -t "%Y-%m-%d" "$stamp")" != "$(date +%Y-%m-%d)" ]]; then
            touch "$stamp"
        else
            cache_flag="--no-update-cache"
        fi
        stats_from=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$stamp")
    elif [[ -n "{{period}}" ]]; then
        echo "Unknown period: {{period}}. Use 'hour' or 'day'."
        exit 1
    fi
    clear
    echo ""
    uv run scripts/status.py $cache_flag
    echo ""
    if [[ -n "$stats_from" ]]; then
        printf "\033[0;90mStats from %s — diff until now (%s)\033[0m\n\n" "$stats_from" "$(date "+%Y-%m-%d %H:%M:%S")"
    fi
    uv run scripts/disk-free.py

# Show condensed processing status: only channels with non-zero video archive or audio
stats-mini:
    #!/usr/bin/env bash
    clear
    just stats | perl -pe 's/\x1b\[[0-9;]*[A-Za-z]//g' | awk 'NF < 5 || /TOTAL/ || ($3 != "-" && $4 != "-")' | grep -v '\-\-\-' | awk '{lines[NR]=$0} /===/{last=NR} END{for(i=last+1;i<=NR;i++) print lines[i]}'

# Show processing status of downloads with transcript time totals enabled
totals period="":
    #!/usr/bin/env bash
    cache_flag=""
    stats_from=""
    if [[ "{{period}}" == "hour" ]]; then
        stamp="/tmp/stats-hour.stamp"
        now=$(date +%s)
        if [[ ! -f "$stamp" ]] || (( now - $(stat -f "%m" "$stamp") >= 3600 )); then
            touch "$stamp"
        else
            cache_flag="--no-update-cache"
        fi
        stats_from=$(date -r "$(stat -f "%m" "$stamp")" "+%Y-%m-%d %H:%M:%S")
    elif [[ "{{period}}" == "day" ]]; then
        stamp="/tmp/stats-day.stamp"
        if [[ ! -f "$stamp" || "$(stat -f "%Sm" -t "%Y-%m-%d" "$stamp")" != "$(date +%Y-%m-%d)" ]]; then
            touch "$stamp"
        else
            cache_flag="--no-update-cache"
        fi
        stats_from=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$stamp")
    elif [[ -n "{{period}}" ]]; then
        echo "Unknown period: {{period}}. Use 'hour' or 'day'."
        exit 1
    fi
    clear
    echo ""
    uv run scripts/status.py --show-time $cache_flag
    echo ""
    if [[ -n "$stats_from" ]]; then
        printf "\033[0;90mStats from %s — diff until now (%s)\033[0m\n\n" "$stats_from" "$(date "+%Y-%m-%d %H:%M:%S")"
    fi

# Count total audio hours from transcript timestamps
audio-hours:
    @echo ""
    @uv run scripts/audio-hours.py
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

    # Validate articles directory using config
    uv run scripts/validate_articles_dir.py

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

    # Validate articles directory using config
    uv run scripts/validate_articles_dir.py

    # Check if port 12000 is available
    if lsof -Pi :12000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        printf "\033[0;31m✗ Error: Port 12000 is already in use\033[0m\n"
        echo "  Please stop the service using port 12000 and try again"
        echo "  You can find the process with: lsof -i :12000"
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
    printf "\033[0;32m✓ Starting development server at http://localhost:12000\033[0m\n"
    echo ""
    cd frontend/newspaper && npm run dev &
    DEV_PID=$!

    # Wait for server to start, then open browser
    sleep 3
    open http://localhost:12000

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
    @uv run bandit -c pyproject.toml -r src --severity-level medium --confidence-level medium
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

# Validate YAML file syntax across the project
check-config-syntax:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Validating YAML Syntax ===\033[0m\n"
    find . \( -path ./frontend/newspaper/node_modules -o -path ./.git -o -path ./.venv -o -path ./.playwright-cli \) -prune \
        -o \( -name "*.yml" -o -name "*.yaml" \) -print \
        | xargs uv run yamllint -d "{extends: relaxed, rules: {line-length: disable, empty-lines: disable}}"
    echo ""
    printf "\033[0;32m✓ YAML validation passed\033[0m\n"
    echo ""

# Scan dependencies for known vulnerabilities
code-audit:
    @echo ""
    @printf "\033[0;34m=== Scanning Dependencies for Vulnerabilities ===\033[0m\n"
    @uv run pip-audit --skip-editable \
        --ignore-vuln PYSEC-2026-139 \
        --ignore-vuln GHSA-rrmf-rvhw-rf47  # torch: no fix released; torch is unused at runtime (mlx-whisper declares it but never imports torch_whisper.py)
    @echo ""
    @printf "\033[0;32m✓ No known vulnerabilities found\033[0m\n"
    @echo ""

# Run Semgrep static analysis
code-semgrep:
    @echo ""
    @printf "\033[0;34m=== Running Semgrep Static Analysis ===\033[0m\n"
    @uv run semgrep --config config/semgrep/ --error src scripts justfile
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
    @rm -rf .cache/unit_test_hashes.json
    @printf "\033[0;33m✓ Cache cleared\033[0m\n"
    @uv run python tools/fake_test_detector/detect_fake_tests.py --no-cache
    @echo ""

# Run AI-powered shell script reviewer (detects env var violations)
ai-review-shell-scripts:
    @echo ""
    @printf "\033[0;34m=== Reviewing Shell Scripts for Env Var Violations ===\033[0m\n"
    @uv run python tools/shellscript_analyzer/shellscript_analyzer.py
    @echo ""

# Run AI-powered shell script reviewer (clear cache and force re-scan)
ai-review-shell-scripts-nocache:
    @echo ""
    @printf "\033[0;34m=== Reviewing Shell Scripts for Env Var Violations (No Cache) ===\033[0m\n"
    @rm -rf .cache/shell_script_hashes.json
    @printf "\033[0;33m✓ Cache cleared\033[0m\n"
    @uv run python tools/shellscript_analyzer/shellscript_analyzer.py --no-cache
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

# Run URL ingestion integration and e2e tests
test-url-ingestion:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Running URL Ingestion Integration Tests ===\033[0m\n"
    uv run pytest -v \
        tests/test_config_data_dir.py \
        tests/test_url_queue_reader.py \
        tests/test_url_normalizer.py \
        tests/test_url_classifier.py \
        tests/test_url_identity.py \
        tests/test_url_metadata.py \
        tests/test_url_requeue_unprocessed.py \
        tests/test_url_reachability.py \
        tests/test_url_scripts.py \
        tests/test_url_downloader.py \
        tests/test_url_download_pipeline.py \
        tests/test_url_formatting.py \
        tests/test_url_raw_processing.py \
        tests/test_url_clean_content_pipeline.py \
        tests/test_url_pipeline_e2e.py
    echo ""
    printf "\033[0;32m✓ URL ingestion integration tests passed\033[0m\n"
    echo ""

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

# Fetch missing .info.json metadata. With no args, scans all non-archived video files. With args: CHANNEL VIDEO_ID [...]
fetch-video-metadata *ARGS:
    @echo ""
    @printf "\033[0;34m=== Fetching Video Metadata ===\033[0m\n"
    @echo ""
    @uv run python scripts/fetch-video-metadata.py {{ ARGS }}
    @echo ""

# Check all channels for WAV files missing .info.json and fetch them automatically
check-missing-metadata:
    @echo ""
    @printf "\033[0;34m=== Checking for Missing Metadata ===\033[0m\n"
    @echo ""
    @uv run python scripts/check-missing-metadata.py
    @echo ""

# Fetch missing video thumbnails. With no args, scans all .info.json files. With args: CHANNEL VIDEO_ID [...]
fetch-video-thumbnails *ARGS:
    @echo ""
    @printf "\033[0;34m=== Fetching Video Thumbnails ===\033[0m\n"
    @echo ""
    @uv run python scripts/fetch-video-thumbnails.py {{ ARGS }}
    @echo ""

# Interactively search cleaned transcripts with fzf and open selected file in Sublime Text
# Optionally provide a QUERY to list all files containing the search term
find QUERY="":
    #!/usr/bin/env bash
    echo ""
    printf "\033[0;34m=== Finding Cleaned Transcripts ===\033[0m\n"
    echo ""
    cleaned_dir=$(uv run python -c "from pathlib import Path; import sys; sys.path.insert(0,'src'); from src.config import Config; c=Config(Path('config/config.yaml')); print(c.get_data_downloads_transcripts_cleaned_dir())")
    if [[ -n "{{ QUERY }}" ]]; then
        rg -c --color=never "{{ QUERY }}" "$cleaned_dir" --glob "*.txt" | sort -t: -k2 -rn | while IFS= read -r line; do
            f="${line%:*}"
            count="${line##*:}"
            printf "%4d  %s/%s\n" "$count" "$(basename "$(dirname "$f")")" "$(basename "$f")"
        done
    else
        selected=$(find "$cleaned_dir" -type f -name "*.txt" | fzf --delimiter / --with-nth -2,-1 --nth -2,-1 --tiebreak=length --preview 'bat --color=always {}' --preview-window right:60%) || true
        if [[ -n "$selected" ]]; then
            subl "$selected"
            printf "\033[0;32m✓ Opened: %s\033[0m\n" "$(basename "$selected")"
        fi
    fi
    echo ""

# Interactively search summary files with fzf and open selected file in Sublime Text
search QUERY="":
    #!/usr/bin/env bash
    echo ""
    printf "\033[0;34m=== Searching Transcript Summaries ===\033[0m\n"
    echo ""
    summaries_dir=$(uv run python -c "from pathlib import Path; import sys; sys.path.insert(0,'src'); from src.config import Config; c=Config(Path('config/config.yaml')); print(c.get_data_downloads_transcripts_summaries_dir())")
    if [[ -n "{{ QUERY }}" ]]; then
        rg -c --color=never "{{ QUERY }}" "$summaries_dir" --glob "*.md" | sort -t: -k2 -rn | while IFS= read -r line; do
            f="${line%:*}"
            count="${line##*:}"
            printf "%4d  %s/%s\n" "$count" "$(basename "$(dirname "$f")")" "$(basename "$f")"
        done
    else
        export _SUMMARIES_DIR="$summaries_dir"
        selected=$(
            : | fzf --disabled --ansi --delimiter : \
                  --header 'Type to search summaries | Enter opens in Sublime Text' \
                  --bind "start:reload(rg --line-number --no-heading --color=always --colors 'path:none' --colors 'line:none' --colors 'match:fg:cyan' '.' \"$summaries_dir\" --glob '*.md' | sed 's|${summaries_dir}/||')" \
                  --bind "change:reload(rg --line-number --no-heading --color=always --colors 'path:none' --colors 'line:none' --colors 'match:fg:cyan' --fixed-strings -- {q} \"$summaries_dir\" --glob '*.md' 2>/dev/null | sed 's|${summaries_dir}/||' || true)" \
                  --preview 'f=$(echo {1} | sed "s/\x1b\[[0-9;]*m//g"); if [[ -n "$FZF_QUERY" ]]; then glow "$_SUMMARIES_DIR/$f" 2>/dev/null | rg --passthru --color=always --colors "match:fg:cyan" --fixed-strings --ignore-case -- "$FZF_QUERY" 2>/dev/null || glow "$_SUMMARIES_DIR/$f" 2>/dev/null; else glow "$_SUMMARIES_DIR/$f" 2>/dev/null; fi' \
                  --preview-window right:60%
        ) || true
        if [[ -n "$selected" ]]; then
            clean=$(echo "$selected" | sed 's/\x1b\[[0-9;]*m//g')
            file="${summaries_dir}/$(echo "$clean" | cut -d: -f1)"
            subl "$file"
            printf "\033[0;32m✓ Opened: %s\033[0m\n" "$(basename "$file")"
        fi
    fi
    echo ""

# List cleaned transcripts and summaries relevant to comma-separated keywords/phrases
# Searches the same corpora as 'find' and 'search', most relevant (most matches) first
research KEYWORDS:
    #!/usr/bin/env bash
    echo ""
    printf "\033[0;34m=== Researching Resources ===\033[0m\n"
    echo ""

    # Build literal rg patterns from the comma-separated keywords (trim whitespace, skip empties)
    rg_args=()
    IFS=',' read -ra _raw_terms <<< "{{ KEYWORDS }}"
    for _t in "${_raw_terms[@]}"; do
        _term=$(printf '%s' "$_t" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -n "$_term" ]] && rg_args+=(-e "$_term")
    done
    if [[ ${#rg_args[@]} -eq 0 ]]; then
        printf "\033[0;31m✗ research failed: no keywords provided\033[0m\n"
        echo ""
        exit 1
    fi

    # Resolve the same corpora that 'find' (cleaned transcripts) and 'search' (summaries) read
    cleaned_dir=$(uv run python -c "from pathlib import Path; import sys; sys.path.insert(0,'src'); from src.config import Config; c=Config(Path('config/config.yaml')); print(c.get_data_downloads_transcripts_cleaned_dir())")
    summaries_dir=$(uv run python -c "from pathlib import Path; import sys; sys.path.insert(0,'src'); from src.config import Config; c=Config(Path('config/config.yaml')); print(c.get_data_downloads_transcripts_summaries_dir())")

    # List files under a directory as "path:count" ranked by how many lines match any keyword (most relevant first)
    matches() {
        rg -c --color=never --ignore-case --fixed-strings "${rg_args[@]}" "$1" --glob "$2" 2>/dev/null | sort -t: -k2 -rn
    }

    # Render "path:count" lines as a readable channel/title heading with the exact file path beneath it
    show() {
        while IFS= read -r line; do
            f="${line%:*}"
            count="${line##*:}"
            printf "%4d  %s/%s\n      %s\n" "$count" "$(basename "$(dirname "$f")")" "$(basename "$f")" "$f"
        done
    }

    transcripts=$(matches "$cleaned_dir" "*.txt")
    summaries=$(matches "$summaries_dir" "*.md")

    printf "\033[0;33mCleaned Transcripts:\033[0m\n"
    if [[ -n "$transcripts" ]]; then printf "%s\n" "$transcripts" | show; else echo "  (none)"; fi
    echo ""
    printf "\033[0;33mSummaries:\033[0m\n"
    if [[ -n "$summaries" ]]; then printf "%s\n" "$summaries" | show; else echo "  (none)"; fi
    echo ""

    n_transcripts=0; [[ -n "$transcripts" ]] && n_transcripts=$(printf "%s\n" "$transcripts" | wc -l | tr -d ' ')
    n_summaries=0; [[ -n "$summaries" ]] && n_summaries=$(printf "%s\n" "$summaries" | wc -l | tr -d ' ')
    printf "\033[0;32m✓ research completed: %s transcript(s), %s summary(ies)\033[0m\n" "$n_transcripts" "$n_summaries"
    echo ""

# Find all files for a video ID across all data directories
find-files VIDEO_ID:
    @echo ""
    @printf "\033[0;34m=== Finding Files for Video ID: {{ VIDEO_ID }} ===\033[0m\n"
    @echo ""
    @bash scripts/find-files.sh {{ VIDEO_ID }}
    @echo ""

# Check if a downloaded video has an audible, non-quiet audio track
check-audio-track CHANNEL VIDEO_ID:
    @echo ""
    @printf "\033[0;34m=== Checking Audio Track: {{ CHANNEL }}/{{ VIDEO_ID }} ===\033[0m\n"
    @echo ""
    @bash scripts/check-audio-track.sh {{ CHANNEL }} {{ VIDEO_ID }}
    @echo ""

# List data files that have no YouTube ID and no ID-bearing sibling (read-only)
find-files-without-youtube-id:
    @echo ""
    @printf "\033[0;34m=== Finding Files Without a YouTube ID ===\033[0m\n"
    @echo ""
    @uv run python scripts/find-files-without-youtube-id.py
    @printf "\033[0;32m✓ find-files-without-youtube-id completed successfully\033[0m\n"
    @echo ""

# Move plain-named duplicate files (no YouTube ID) to backup location
cleanup-plain-filename-duplicates:
    @echo ""
    @printf "\033[0;34m=== Cleanup Plain Filename Duplicates ===\033[0m\n"
    @echo ""
    @uv run python scripts/cleanup-plain-filename-duplicates.py
    @echo ""

# List transcript files that are 100 bytes or smaller
find-empty-transcripts:
    @echo ""
    @printf "\033[0;34m=== Finding Empty Transcripts (≤100 bytes) ===\033[0m\n"
    @echo ""
    @bash scripts/find-empty-transcripts.sh
    @echo ""

# Flag data files whose video ID is in the filter file but were never removed (read-only)
find-files-with-filtered-video-ids:
    @echo ""
    @printf "\033[0;34m=== Finding Files With Filtered Video IDs ===\033[0m\n"
    @echo ""
    @uv run python scripts/find-files-with-filtered-video-ids.py

# Run ALL validation checks (verbose)
ci-verbose:
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
    just check-config-syntax
    just code-semgrep
    just code-audit
    just test-url-ingestion
    just test
    just code-lspchecks
    echo ""
    printf "\033[0;32m✓ All CI checks passed\033[0m\n"
    echo ""

# Run ALL validation checks silently (only show output on errors)
ci:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Running CI Checks (Quiet Mode) ===\033[0m\n"
    TMPFILE=$(mktemp)
    trap "rm -f $TMPFILE" EXIT

    just init > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Init failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Init passed\033[0m\n"

    just code-config > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-config failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-config passed\033[0m\n"

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

    just check-config-syntax > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Check-config-syntax failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Check-config-syntax passed\033[0m\n"

    just code-semgrep > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-semgrep failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-semgrep passed\033[0m\n"

    just code-audit > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-audit failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-audit passed\033[0m\n"

    just test-url-ingestion > $TMPFILE 2>&1 || { printf "\033[0;31m✗ URL ingestion integration tests failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ URL ingestion integration tests passed\033[0m\n"

    just test > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Test failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Test passed\033[0m\n"

    just code-lspchecks > $TMPFILE 2>&1 || { printf "\033[0;31m✗ Code-lspchecks failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ Code-lspchecks passed\033[0m\n"

    echo ""
    printf "\033[0;32m✓ All CI checks passed\033[0m\n"
    echo ""

# Run the complete pipeline quietly (only show errors and warnings)
# Run AI-based CI checks (AI-powered test validation, will grow in the future)
# This pipeline is separate from regular CI and includes AI-assisted code quality checks
ci-ai:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Running AI-Based CI Checks ===\033[0m\n"
    echo ""
    just ai-review-unit-tests-nocache
    just ai-review-shell-scripts-nocache
    echo ""
    printf "\033[0;32m✓ All AI-based CI checks passed\033[0m\n"
    echo ""

# Run AI-based CI checks silently (only show output on errors)
ci-ai-quiet:
    #!/usr/bin/env bash
    set -e
    echo ""
    printf "\033[0;34m=== Running AI-Based CI Checks (Quiet Mode) ===\033[0m\n"
    TMPFILE=$(mktemp)
    trap "rm -f $TMPFILE" EXIT

    just ai-review-unit-tests-nocache > $TMPFILE 2>&1 || { printf "\033[0;31m✗ AI-review-unit-tests failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ AI-review-unit-tests passed\033[0m\n"

    just ai-review-shell-scripts-nocache > $TMPFILE 2>&1 || { printf "\033[0;31m✗ AI-review-shell-scripts failed\033[0m\n"; cat $TMPFILE; exit 1; }
    printf "\033[0;32m✓ AI-review-shell-scripts passed\033[0m\n"

    echo ""
    printf "\033[0;32m✓ All AI-based CI checks passed\033[0m\n"
    echo ""

# =============================================================================
# Analytics (isolated; reads cleaned transcripts + summaries + metadata; no LLM)
# =============================================================================

# Build corpus index from cleaned transcripts + summaries + metadata
analytics-index:
    @uv run python scripts/analytics/index.py

# Theme frequency report (filters from the analytics: config block)
analytics-themes:
    @uv run python scripts/analytics/themes.py

# Timeline report by upload_date
analytics-timeline:
    @uv run python scripts/analytics/timeline.py

# Full research digest: index + themes + timeline + emerging diff
analytics:
    @uv run python scripts/analytics/digest.py
