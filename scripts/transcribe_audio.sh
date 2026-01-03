#!/bin/bash

# Source central configuration
source "$(dirname "$0")/config.sh"

echo "Starting batch transcription with model: $MODEL_NAME"
echo "Model repo: $MODEL_REPO"
echo "Hallucination silence threshold: ${HALLUCINATION_SILENCE_THRESHOLD}s"
echo "Compression ratio threshold: $COMPRESSION_RATIO_THRESHOLD"
echo "Use YouTube metadata: $USE_YOUTUBE_METADATA"
echo "=========================================="
echo ""

# Create temp directory for transcriptions
mkdir -p "$TEMP_DIR"

# Iterate over all channel folders in audio directory
find "$AUDIO_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r channel_dir; do
    channel_name=$(basename "$channel_dir")

    echo "Processing channel: $channel_name"
    echo "---"

    # Create transcripts directory for this channel
    mkdir -p "$TRANSCRIPTS_DIR/$channel_name"

    # Counter for skipped files in this channel
    skipped_count=0

    # Find all .wav files in this channel's audio directory
    while IFS= read -r -d '' wav_file; do

        # Get the base filename without the .wav extension
        base_name=$(basename "$wav_file" .wav)

        # Skip macOS hidden metadata files (._filename)
        if [[ "$base_name" == ._* ]]; then
            if [ "$VERBOSE" = "true" ]; then
                echo "  ⏭️  Skipping macOS metadata file: $base_name"
            fi
            continue
        fi

        # Check if transcript already exists
        if [ -f "$TRANSCRIPTS_DIR/$channel_name/$base_name.txt" ]; then
            skipped_count=$((skipped_count + 1))
            if [ "$VERBOSE" = "true" ]; then
                echo "  ⏭️  Skipping: $base_name.wav (transcript already exists)"
                echo "  ---"
            fi
            continue
        fi

        echo "  Transcribing: $base_name.wav"

        # Create a temp directory for this transcription
        temp_dir=$(mktemp -d "$TEMP_DIR/transcribe.XXXXXX")

        # Build initial prompt using YouTube metadata if available
        initial_prompt=""
        prompt_source="fallback"
        if [ "$USE_YOUTUBE_METADATA" = "true" ]; then
            metadata_file="$METADATA_DIR/$channel_name/$METADATA_VIDEO_SUBDIR/$base_name.info.json"

            if [ "$VERBOSE" = "true" ]; then
                echo "    🔍 Looking for metadata: $metadata_file"
            fi

            if [ -f "$metadata_file" ]; then
                # Extract title and description from metadata
                title=$(jq -r '.title // empty' "$metadata_file" 2>/dev/null)
                description=$(jq -r '.description // empty' "$metadata_file" 2>/dev/null)

                if [ "$VERBOSE" = "true" ]; then
                    echo "    ✅ Metadata found - Title: ${title:0:50}..."
                fi

                if [ -n "$title" ]; then
                    initial_prompt="This is a YouTube video with the title: $title"

                    if [ -n "$description" ]; then
                        # Truncate description to first 500 characters to avoid extremely long prompts
                        description_truncated=$(echo "$description" | head -c 500)
                        initial_prompt="$initial_prompt. Description: $description_truncated"
                    fi

                    prompt_source="metadata"
                fi
            else
                echo "    🚨 ERROR: Metadata file not found: $metadata_file"
                echo "    💡 Run 'bash scripts/move-metadata.sh' to move .info.json files from videos/ to metadata/"
                exit 1
            fi
        fi

        # If no metadata found or USE_YOUTUBE_METADATA is false, use fallback prompt
        if [ -z "$initial_prompt" ]; then
            initial_prompt="This is a technical discussion about artificial intelligence, machine learning, and AI research."
        fi

        # Display transcription settings info box
        echo ""
        echo "  ╔════════════════════════════════════════════════════════════════════════════╗"
        echo "  ║ TRANSCRIPTION SETTINGS                                                     ║"
        echo "  ╠════════════════════════════════════════════════════════════════════════════╣"
        echo "  ║                                                                            ║"
        echo "  ║ Model:                        $MODEL_NAME"
        echo "  ║ Language:                     en"
        echo "  ║ Hallucination Silence Thresh: ${HALLUCINATION_SILENCE_THRESHOLD}s"
        echo "  ║ Compression Ratio Threshold:  $COMPRESSION_RATIO_THRESHOLD"
        echo "  ║ Prompt Source:                $prompt_source"
        echo "  ║                                                                            ║"
        echo "  ╠════════════════════════════════════════════════════════════════════════════╣"
        echo "  ║ INITIAL PROMPT                                                             ║"
        echo "  ╠════════════════════════════════════════════════════════════════════════════╣"
        echo "  ║                                                                            ║"
        # Word wrap the prompt to fit within the box (76 chars per line)
        echo "$initial_prompt" | fold -s -w 76 | while IFS= read -r line; do
            printf "  ║ %-74s ║\n" "$line"
        done
        echo "  ║                                                                            ║"
        echo "  ╚════════════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo ""

        # Transcribe WAV file to temp directory with anti-hallucination settings
        uv run mlx_whisper "$wav_file" \
            --model "$MODEL_REPO" \
            --output-dir "$temp_dir" \
            --output-format all \
            --language en \
            --hallucination-silence-threshold "$HALLUCINATION_SILENCE_THRESHOLD" \
            --compression-ratio-threshold "$COMPRESSION_RATIO_THRESHOLD" \
            --initial-prompt "$initial_prompt" \
            --word-timestamps True

        if [ $? -eq 0 ]; then
            # Move all generated files to the transcripts directory
            moved_files=0
            for ext in txt srt vtt tsv json; do
                generated_file=$(find "$temp_dir" -type f -name "*.$ext" | head -n 1)
                if [ -f "$generated_file" ]; then
                    mv "$generated_file" "$TRANSCRIPTS_DIR/$channel_name/$base_name.$ext"
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

        # Pause between transcriptions to avoid overwhelming the system
        echo "    ⏸️  Pausing for 5 seconds before next transcription..."
        sleep 5

        echo "  ---"
    done < <(find "$channel_dir" -maxdepth 1 -type f -name "*.wav" -print0 2>/dev/null)

    # Print skip summary if any files were skipped
    if [ $skipped_count -gt 0 ]; then
        echo "⏭️  Skipped $skipped_count file(s) (transcript already exists)"
    fi

    echo ""
done

echo "Batch transcription complete."
