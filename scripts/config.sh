#!/bin/bash

# ============================================================================
# Central Configuration for Shell Scripts
# ============================================================================
# This file contains all shared configuration for scripts in this directory.
# Source this file at the beginning of each script: source "$(dirname "$0")/config.sh"

# --- Project Paths ---
# Get the absolute path to the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Read data directory path from config.yaml (single source of truth)
CONFIG_FILE="$PROJECT_ROOT/config/config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "🚨 ERROR: config.yaml not found at $CONFIG_FILE" >&2
    exit 1
fi

# Extract data_dir from config.yaml using Python (available via uv)
DATA_DIR_RELATIVE=$(cd "$PROJECT_ROOT" && uv run python -c "
import yaml
import sys
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(config['paths']['data_dir'])
" 2>/dev/null)

if [ -z "$DATA_DIR_RELATIVE" ]; then
    echo "🚨 ERROR: Failed to read paths.data_dir from config.yaml" >&2
    exit 1
fi

# Convert relative path to absolute path
DATA_DIR="$(cd "$PROJECT_ROOT" && cd "$DATA_DIR_RELATIVE" && pwd)"
DOWNLOADS_DIR="$DATA_DIR/downloads"
VIDEOS_DIR="$DOWNLOADS_DIR/videos"
AUDIO_DIR="$DOWNLOADS_DIR/audio"
TRANSCRIPTS_DIR="$DOWNLOADS_DIR/transcripts"
METADATA_DIR="$DOWNLOADS_DIR/metadata"
# Metadata subdirectories (per channel):
# - [channelname]/video/        - Video metadata (.info.json from yt-dlp)
# - [channelname]/audio/        - Audio metadata (.silence_map.json)
# - [channelname]/               - Channel metadata (.info.json for playlist/channel)
METADATA_VIDEO_SUBDIR="video"
METADATA_AUDIO_SUBDIR="audio"
ARCHIVE_DIR="$DATA_DIR/archive"
ARCHIVE_VIDEOS_DIR="$ARCHIVE_DIR/videos"
TEMP_DIR="$DATA_DIR/temp"
OUTPUT_DIR="$DATA_DIR/output"

# --- Video/Audio Processing Settings ---
# Whitelist of allowed video file extensions
ALLOWED_EXTENSIONS=("mp4" "mkv" "webm" "m4a" "mov" "m4v" "avi" "flv")

# VERBOSE: Set to "true" to show individual operations/messages
# Can be overridden via environment: VERBOSE=true ./script.sh
VERBOSE="${VERBOSE:-false}"

# --- Silence Detection Settings (convert_to_audio.sh) ---
# Silence threshold in decibels (dB)
# More negative = only very quiet sounds treated as silence
# Less negative = louder sounds also treated as silence
# Common values: -50 (aggressive), -40 (moderate), -30 (conservative)
SILENCE_THRESHOLD_DB="${SILENCE_THRESHOLD_DB:--40}"

# Minimum silence duration in seconds
# Only silence longer than this duration will be removed
# Common values: 0.5 (aggressive), 1.0-2.0 (moderate), 3.0+ (conservative)
SILENCE_MIN_DURATION="${SILENCE_MIN_DURATION:-1}"

# Enable or disable silence removal entirely
# Set to "false" to only convert audio without silence processing
# Can be overridden via environment: ENABLE_SILENCE_REMOVAL=false ./convert_to_audio.sh
ENABLE_SILENCE_REMOVAL="${ENABLE_SILENCE_REMOVAL:-true}"

# --- Transcription Settings (transcribe_audio.sh) ---
# Whisper models
# English-only model (optimized for English content)
MODEL_EN_NAME="${MODEL_EN_NAME:-medium.en}"
MODEL_EN_REPO="${MODEL_EN_REPO:-mlx-community/whisper-medium.en-mlx}"

# Multilingual model (supports all languages, used for translation)
MODEL_MULTI_NAME="${MODEL_MULTI_NAME:-medium}"
MODEL_MULTI_REPO="${MODEL_MULTI_REPO:-mlx-community/whisper-medium-mlx}"

# Anti-hallucination settings
# If hallucination detected after N seconds of silence, seek past silence and retry
HALLUCINATION_SILENCE_THRESHOLD="${HALLUCINATION_SILENCE_THRESHOLD:-2.0}"

# If gzip compression ratio > this value, treat as hallucination
# Lower = stricter (default is 2.4, we use 2.0 for better quality)
COMPRESSION_RATIO_THRESHOLD="${COMPRESSION_RATIO_THRESHOLD:-2.0}"

# Whether to use YouTube metadata (title, description) in the initial prompt
USE_YOUTUBE_METADATA="${USE_YOUTUBE_METADATA:-true}"

# --- YouTube Download Settings (yt-downloader.sh) ---
# Browser to extract cookies from
BROWSER="${BROWSER:-chrome}"

# ============================================================================
# Helper Functions
# ============================================================================

# Function to ensure required directories exist
ensure_directories() {
    mkdir -p "$VIDEOS_DIR"
    mkdir -p "$AUDIO_DIR"
    mkdir -p "$TRANSCRIPTS_DIR"
    mkdir -p "$METADATA_DIR"
    mkdir -p "$ARCHIVE_VIDEOS_DIR"
    mkdir -p "$TEMP_DIR"
    mkdir -p "$OUTPUT_DIR"
}

# Function to detect number of CPU cores for optimal threading
detect_cpu_cores() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sysctl -n hw.ncpu
    else
        # Linux
        nproc
    fi
}

# ============================================================================
# Validation
# ============================================================================

# Ensure PROJECT_ROOT is set
if [ -z "$PROJECT_ROOT" ]; then
    echo "🚨 ERROR: Failed to determine PROJECT_ROOT" >&2
    exit 1
fi
