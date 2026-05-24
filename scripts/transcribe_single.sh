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
wav_file="$1"
model_repo="$2"
output_dir="$3"
task="$4"
language="$5"
hallucination_silence_threshold="$6"
compression_ratio_threshold="$7"
initial_prompt="$8"

# Validate required parameters are non-empty
if [ -z "$wav_file" ]; then
    echo "❌ ERROR: wav_file parameter is empty" >&2
    exit 1
fi

if [ -z "$model_repo" ]; then
    echo "❌ ERROR: model_repo parameter is empty" >&2
    exit 1
fi

if [ -z "$output_dir" ]; then
    echo "❌ ERROR: output_dir parameter is empty" >&2
    exit 1
fi

if [ -z "$task" ]; then
    echo "❌ ERROR: task parameter is empty" >&2
    exit 1
fi

if [ -z "$language" ]; then
    echo "❌ ERROR: language parameter is empty" >&2
    exit 1
fi

if [ -z "$hallucination_silence_threshold" ]; then
    echo "❌ ERROR: hallucination_silence_threshold parameter is empty" >&2
    exit 1
fi

if [ -z "$compression_ratio_threshold" ]; then
    echo "❌ ERROR: compression_ratio_threshold parameter is empty" >&2
    exit 1
fi

# initial_prompt can be empty (fallback handled by Python)

# Validate file exists
if [ ! -f "$wav_file" ]; then
    echo "❌ ERROR: WAV file not found: $wav_file" >&2
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Call mlx_whisper
# Note: When task=translate, we omit --language to avoid confusing Whisper
# (translate task always outputs English, and --language would be misinterpreted as output language)
if [ "$task" = "translate" ]; then
    uv run mlx_whisper "$wav_file" \
        --model "$model_repo" \
        --output-dir "$output_dir" \
        --output-format all \
        --task "$task" \
        --hallucination-silence-threshold "$hallucination_silence_threshold" \
        --compression-ratio-threshold "$compression_ratio_threshold" \
        --initial-prompt "$initial_prompt" \
        --word-timestamps True
else
    uv run mlx_whisper "$wav_file" \
        --model "$model_repo" \
        --output-dir "$output_dir" \
        --output-format all \
        --task "$task" \
        --language "$language" \
        --hallucination-silence-threshold "$hallucination_silence_threshold" \
        --compression-ratio-threshold "$compression_ratio_threshold" \
        --initial-prompt "$initial_prompt" \
        --word-timestamps True
fi

# Check if transcription succeeded
if [ $? -eq 0 ]; then
    echo "      ⏸️  Waiting for 5 seconds..."
    sleep 5
fi

exit $?
