#!/bin/bash

# ============================================================================
# Single File Transcription Wrapper
# ============================================================================
# Calls mlx_whisper with all parameters passed from Python orchestrator.
# This script does NOT read config - all parameters must be provided.

set -e  # Exit on error
set -u  # Exit on undefined variable

# Validate parameter count
if [ $# -ne 8 ]; then
    echo "❌ ERROR: Expected 8 parameters, got $#" >&2
    echo "Usage: $0 <wav_file> <model_repo> <output_dir> <task> <language> <hallucination_silence_threshold> <compression_ratio_threshold> <initial_prompt>" >&2
    exit 1
fi

# Assign parameters
WAV_FILE="$1"
MODEL_REPO="$2"
OUTPUT_DIR="$3"
TASK="$4"
LANGUAGE="$5"
HALLUCINATION_SILENCE_THRESHOLD="$6"
COMPRESSION_RATIO_THRESHOLD="$7"
INITIAL_PROMPT="$8"

# Validate required parameters are non-empty
if [ -z "$WAV_FILE" ]; then
    echo "❌ ERROR: wav_file parameter is empty" >&2
    exit 1
fi

if [ -z "$MODEL_REPO" ]; then
    echo "❌ ERROR: model_repo parameter is empty" >&2
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "❌ ERROR: output_dir parameter is empty" >&2
    exit 1
fi

if [ -z "$TASK" ]; then
    echo "❌ ERROR: task parameter is empty" >&2
    exit 1
fi

if [ -z "$LANGUAGE" ]; then
    echo "❌ ERROR: language parameter is empty" >&2
    exit 1
fi

if [ -z "$HALLUCINATION_SILENCE_THRESHOLD" ]; then
    echo "❌ ERROR: hallucination_silence_threshold parameter is empty" >&2
    exit 1
fi

if [ -z "$COMPRESSION_RATIO_THRESHOLD" ]; then
    echo "❌ ERROR: compression_ratio_threshold parameter is empty" >&2
    exit 1
fi

# initial_prompt can be empty (fallback handled by Python)

# Validate file exists
if [ ! -f "$WAV_FILE" ]; then
    echo "❌ ERROR: WAV file not found: $WAV_FILE" >&2
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Call mlx_whisper
# Note: When task=translate, we omit --language to avoid confusing Whisper
# (translate task always outputs English, and --language would be misinterpreted as output language)
if [ "$TASK" = "translate" ]; then
    uv run mlx_whisper "$WAV_FILE" \
        --model "$MODEL_REPO" \
        --output-dir "$OUTPUT_DIR" \
        --output-format all \
        --task "$TASK" \
        --hallucination-silence-threshold "$HALLUCINATION_SILENCE_THRESHOLD" \
        --compression-ratio-threshold "$COMPRESSION_RATIO_THRESHOLD" \
        --initial-prompt "$INITIAL_PROMPT" \
        --word-timestamps True
else
    uv run mlx_whisper "$WAV_FILE" \
        --model "$MODEL_REPO" \
        --output-dir "$OUTPUT_DIR" \
        --output-format all \
        --task "$TASK" \
        --language "$LANGUAGE" \
        --hallucination-silence-threshold "$HALLUCINATION_SILENCE_THRESHOLD" \
        --compression-ratio-threshold "$COMPRESSION_RATIO_THRESHOLD" \
        --initial-prompt "$INITIAL_PROMPT" \
        --word-timestamps True
fi

# Check if transcription succeeded
if [ $? -eq 0 ]; then
    echo "      ⏸️  Waiting for 5 seconds..."
    sleep 5
fi

exit $?
