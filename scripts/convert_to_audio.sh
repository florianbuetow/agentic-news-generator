#!/bin/bash

# --- Configuration ---
PROJECT_DIR="/Volumes/2TB/agentic-news-generator.git/florian-topic-segmentation"

# Whitelist of allowed video file extensions
ALLOWED_EXTENSIONS=("mp4" "webm" "m4a" "mov" "m4v" "avi" "mkv" "flv")

# VERBOSE: Set to "true" to show individual skip messages, otherwise show summary
VERBOSE="${VERBOSE:-false}"
# --- End Configuration ---

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "🚨 ERROR: ffmpeg is not installed"
    exit 1
fi

# Detect number of CPU cores for optimal threading
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    NUM_THREADS=$(sysctl -n hw.ncpu)
else
    # Linux
    NUM_THREADS=$(nproc)
fi

echo "Converting videos to audio files"
echo "=========================================="
echo ""

# Iterate over all channel folders in videos directory
find "$PROJECT_DIR/data/downloads/videos" -mindepth 1 -maxdepth 1 -type d | while read -r channel_dir; do
    channel_name=$(basename "$channel_dir")

    echo "Processing channel: $channel_name"
    echo "---"

    # Create output directory for this channel
    mkdir -p "$PROJECT_DIR/data/downloads/audio/$channel_name"

    # Counter for skipped files in this channel
    skipped_count=0

    # Process all video files matching the whitelist
    while IFS= read -r -d '' input_file; do

        # Get the base filename and extension
        filename=$(basename "$input_file")
        extension="${filename##*.}"
        base_name="${filename%.*}"

        # Skip macOS metadata files
        if [[ "$filename" == ._* ]]; then
            continue
        fi

        # Validate file extension against whitelist
        extension_lower=$(echo "$extension" | tr '[:upper:]' '[:lower:]')
        is_valid=false
        for allowed_ext in "${ALLOWED_EXTENSIONS[@]}"; do
            if [[ "$extension_lower" == "$allowed_ext" ]]; then
                is_valid=true
                break
            fi
        done

        if [[ "$is_valid" == false ]]; then
            continue
        fi

        # Define the output .wav file path
        output_wav="$PROJECT_DIR/data/downloads/audio/$channel_name/$base_name.wav"

        # Check if output WAV file already exists
        if [ -f "$output_wav" ]; then
            skipped_count=$((skipped_count + 1))
            if [ "$VERBOSE" = "true" ]; then
                echo "  ⏭️  Skipping: $filename"
                echo "  ---"
            fi
        else
            echo "  Processing: $filename"

            # Convert to WAV with proper format
            # -threads: Number of threads to use
            # -y: Overwrite output file without asking
            # -i: Input file
            # -vn: No video (discard video stream)
            # -ar 16000: Audio rate 16kHz
            # -ac 1: Audio channels 1 (mono)
            # -c:a pcm_s16le: Codec for 16-bit PCM WAV
            # -loglevel error: Only show errors
            # -stats: Show brief progress
            echo "    Converting to audio/$channel_name/$base_name.wav (using $NUM_THREADS threads)..."
            ffmpeg -threads "$NUM_THREADS" -y -i "$input_file" -vn -ar 16000 -ac 1 -c:a pcm_s16le -loglevel error -stats "$output_wav" </dev/null
            ffmpeg_exit=$?

            if [ $ffmpeg_exit -eq 0 ]; then
                echo "    ✅ Done: $base_name.wav"
            else
                echo "    🚨 FAILED to convert $filename"
                exit 1
            fi
            echo "  ---"
        fi

    done < <(find "$channel_dir" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.wav" -o -name "*.webm" -o -name "*.m4a" -o -name "*.mov" -o -name "*.m4v" -o -name "*.mp3" -o -name "*.ogg" \) -print0)

    # Print skip summary if any files were skipped
    if [ $skipped_count -gt 0 ]; then
        echo "⏭️  Skipped $skipped_count file(s) (WAV already exists)"
    fi

    echo ""
done

echo "Audio conversion complete."
