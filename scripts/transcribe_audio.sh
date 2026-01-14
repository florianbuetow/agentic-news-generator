#!/bin/bash

# Source central configuration
source "$(dirname "$0")/config.sh"

echo "Starting batch transcription with language grouping"
echo "English model: $MODEL_EN_NAME ($MODEL_EN_REPO)"
echo "Multilingual model: $MODEL_MULTI_NAME ($MODEL_MULTI_REPO)"
echo "Hallucination silence threshold: ${HALLUCINATION_SILENCE_THRESHOLD}s"
echo "Compression ratio threshold: $COMPRESSION_RATIO_THRESHOLD"
echo "Use YouTube metadata: $USE_YOUTUBE_METADATA"
echo "=========================================="
echo ""

# Create temp directory for transcriptions
mkdir -p "$TEMP_DIR"

# Get channels grouped by language from config.yaml
language_groups=$(cd "$PROJECT_ROOT" && uv run python scripts/group-channels-by-language.py)

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to group channels by language"
    echo "$language_groups"
    exit 1
fi

# Extract unique languages (keys from JSON)
languages=$(echo "$language_groups" | jq -r 'keys[]' | sort)

# Counters for final summary
total_processed=0
total_failed=0

# Process each language group
for lang in $languages; do
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Processing Language Group: $lang"
    echo "════════════════════════════════════════════════════════"

    # Determine model and task for this language
    if [ "$lang" = "en" ]; then
        model_repo="$MODEL_EN_REPO"
        model_name="$MODEL_EN_NAME"
        whisper_task="transcribe"
        whisper_language="en"
    else
        model_repo="$MODEL_MULTI_REPO"
        model_name="$MODEL_MULTI_NAME"
        whisper_task="translate"
        whisper_language="$lang"
    fi

    echo "  Model: $model_name"
    echo "  Task: $whisper_task"
    echo "  Language: $whisper_language"
    echo ""

    # Get all channels for this language
    channels=$(echo "$language_groups" | jq -r ".\"$lang\"[] | .sanitized_name")

    # Process each channel in this language group
    for channel_name in $channels; do
        channel_audio_dir="$AUDIO_DIR/$channel_name"

        if [ ! -d "$channel_audio_dir" ]; then
            echo "⚠️  Channel directory not found: $channel_audio_dir (skipping)"
            continue
        fi

        echo "  📁 Channel: $channel_name"
        echo "  ---"

        # Create transcripts directory for this channel
        mkdir -p "$TRANSCRIPTS_DIR/$channel_name"

        # Counter for skipped files in this channel
        skipped_count=0

        # Find and process WAV files for this channel
        while IFS= read -r -d '' wav_file; do
            # Get the base filename without the .wav extension
            base_name=$(basename "$wav_file" .wav)

            # Skip macOS hidden metadata files (._filename)
            if [[ "$base_name" == ._* ]]; then
                if [ "$VERBOSE" = "true" ]; then
                    echo "    ⏭️  Skipping macOS metadata file: $base_name"
                fi
                continue
            fi

            # Check if transcript already exists (idempotent)
            if [ -f "$TRANSCRIPTS_DIR/$channel_name/$base_name.txt" ]; then
                skipped_count=$((skipped_count + 1))
                if [ "$VERBOSE" = "true" ]; then
                    echo "    ⏭️  Skipping: $base_name.wav (transcript already exists)"
                fi
                continue
            fi

            echo "    🎙️  Transcribing: $base_name.wav"

            # Create a temp directory for this transcription
            temp_dir=$(mktemp -d "$TEMP_DIR/transcribe.XXXXXX")

            # Build initial prompt using YouTube metadata if available
            initial_prompt=""
            prompt_source="fallback"
            if [ "$USE_YOUTUBE_METADATA" = "true" ]; then
                metadata_file="$METADATA_DIR/$channel_name/$METADATA_VIDEO_SUBDIR/$base_name.info.json"

                if [ "$VERBOSE" = "true" ]; then
                    echo "      🔍 Looking for metadata: $metadata_file"
                fi

                if [ -f "$metadata_file" ]; then
                    # Extract title and description from metadata
                    title=$(jq -r '.title // empty' "$metadata_file" 2>/dev/null)
                    description=$(jq -r '.description // empty' "$metadata_file" 2>/dev/null)

                    if [ "$VERBOSE" = "true" ]; then
                        echo "      ✅ Metadata found - Title: ${title:0:50}..."
                    fi

                    if [ -n "$title" ]; then
                        initial_prompt="This is a YouTube video with the title: $title"

                        if [ -n "$description" ]; then
                            # Truncate description to first 500 characters
                            description_truncated=$(echo "$description" | head -c 500)
                            initial_prompt="$initial_prompt. Description: $description_truncated"
                        fi

                        prompt_source="metadata"
                    fi
                else
                    echo "      🚨 ERROR: Metadata file not found: $metadata_file"
                    echo "      💡 Run 'bash scripts/move-metadata.sh' to move .info.json files"
                    total_failed=$((total_failed + 1))
                    continue
                fi
            fi

            # If no metadata found or USE_YOUTUBE_METADATA is false, use fallback prompt
            if [ -z "$initial_prompt" ]; then
                initial_prompt="This is a technical discussion about artificial intelligence, machine learning, and AI research."
            fi

            # Display transcription settings info box
            if [ "$VERBOSE" = "true" ]; then
                echo ""
                echo "      ╔════════════════════════════════════════════════════════════════════════════╗"
                echo "      ║ TRANSCRIPTION SETTINGS                                                     ║"
                echo "      ╠════════════════════════════════════════════════════════════════════════════╣"
                echo "      ║                                                                            ║"
                echo "      ║ Model:                        $model_name"
                echo "      ║ Task:                         $whisper_task"
                echo "      ║ Language:                     $whisper_language"
                echo "      ║ Hallucination Silence Thresh: ${HALLUCINATION_SILENCE_THRESHOLD}s"
                echo "      ║ Compression Ratio Threshold:  $COMPRESSION_RATIO_THRESHOLD"
                echo "      ║ Prompt Source:                $prompt_source"
                echo "      ║                                                                            ║"
                echo "      ╚════════════════════════════════════════════════════════════════════════════╝"
                echo ""
            fi

            # Transcribe WAV file to temp directory with dynamic model/task/language
            uv run mlx_whisper "$wav_file" \
                --model "$model_repo" \
                --output-dir "$temp_dir" \
                --output-format all \
                --task "$whisper_task" \
                --language "$whisper_language" \
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
                    echo "      ✅ Success: $base_name"
                    total_processed=$((total_processed + 1))
                else
                    echo "      ❌ Failed: No transcript generated for $base_name.wav"
                    total_failed=$((total_failed + 1))
                fi
            else
                echo "      ❌ Failed to transcribe $base_name.wav"
                total_failed=$((total_failed + 1))
            fi

            # Clean up temp directory
            rm -rf "$temp_dir"

            # Pause between transcriptions
            sleep 5

        done < <(find "$channel_audio_dir" -maxdepth 1 -type f -name "*.wav" -print0 2>/dev/null)

        # Print skip summary if any files were skipped
        if [ $skipped_count -gt 0 ]; then
            echo "  ⏭️  Skipped $skipped_count file(s) (transcript already exists)"
        fi

        echo ""
    done
done

# Final summary
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Transcription Complete"
echo "════════════════════════════════════════════════════════"
echo "  Processed: $total_processed"
echo "  Failed: $total_failed"
echo ""

[ $total_failed -eq 0 ] && exit 0 || exit 1
