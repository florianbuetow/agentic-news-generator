#!/bin/bash

# --- Configuration ---
PROJECT_DIR="/Volumes/2TB/agentic-news-generator.git/florian-topic-segmentation"
MODEL_NAME="large-v3"
MODEL_REPO="mlx-community/whisper-large-v3-mlx"

# VERBOSE: Set to "true" to show individual skip messages, otherwise show summary
# Default is "false" (summary only)
VERBOSE="${VERBOSE:-false}"
# --- End Configuration ---

echo "Starting batch transcription with model: $MODEL_NAME"
echo "Model repo: $MODEL_REPO"
echo "=========================================="
echo ""

# Create temp directory for transcriptions
mkdir -p "$PROJECT_DIR/data/temp"

# Iterate over all channel folders in audio directory
find "$PROJECT_DIR/data/downloads/audio" -mindepth 1 -maxdepth 1 -type d | while read -r channel_dir; do
    channel_name=$(basename "$channel_dir")

    echo "Processing channel: $channel_name"
    echo "---"

    # Create transcripts directory for this channel
    mkdir -p "$PROJECT_DIR/data/downloads/transcripts/$channel_name"

    # Counter for skipped files in this channel
    skipped_count=0

    # Find all .wav files in this channel's audio directory
    while IFS= read -r -d '' wav_file; do

        # Get the base filename without the .wav extension
        base_name=$(basename "$wav_file" .wav)

        # Check if transcript already exists
        if [ -f "$PROJECT_DIR/data/downloads/transcripts/$channel_name/$base_name.txt" ]; then
            skipped_count=$((skipped_count + 1))
            if [ "$VERBOSE" = "true" ]; then
                echo "  ⏭️  Skipping: $base_name.wav (transcript already exists)"
                echo "  ---"
            fi
            continue
        fi

        echo "  Transcribing: $base_name.wav"

        # Create a temp directory for this transcription
        temp_dir=$(mktemp -d "$PROJECT_DIR/data/temp/transcribe.XXXXXX")

        # Transcribe WAV file to temp directory
        uv run mlx_whisper "$wav_file" --model "$MODEL_REPO" --output-dir "$temp_dir" --output-format all

        if [ $? -eq 0 ]; then
            # Move all generated files to the transcripts directory
            moved_files=0
            for ext in txt srt vtt tsv json; do
                generated_file=$(find "$temp_dir" -type f -name "*.$ext" | head -n 1)
                if [ -f "$generated_file" ]; then
                    mv "$generated_file" "$PROJECT_DIR/data/downloads/transcripts/$channel_name/$base_name.$ext"
                    moved_files=$((moved_files + 1))
                fi
            done

            if [ $moved_files -gt 0 ]; then
                echo "    ✅ Done: $base_name"
            else
                echo "    🚨 FAILED: No transcript generated for $base_name.wav"
                rm -rf "$temp_dir"
                exit 1
            fi
        else
            echo "    🚨 FAILED to transcribe $base_name.wav"
            rm -rf "$temp_dir"
            exit 1
        fi

        # Clean up temp directory
        rm -rf "$temp_dir"

        echo "  ---"
    done < <(find "$channel_dir" -maxdepth 1 -type f -name "*.wav" -print0 2>/dev/null)

    # Print skip summary if any files were skipped
    if [ $skipped_count -gt 0 ]; then
        echo "⏭️  Skipped $skipped_count file(s) (transcript already exists)"
    fi

    echo ""
done

echo "Batch transcription complete."
