#!/bin/bash

# Source central configuration
source "$(dirname "$0")/config.sh"

# Check if required dependencies are installed
if ! command -v ffprobe &> /dev/null; then
    echo "🚨 ERROR: ffprobe is not installed (should come with ffmpeg)"
    exit 1
fi

# Detect timeout command (GNU coreutils: 'timeout' on Linux, 'gtimeout' on macOS)
TIMEOUT_CMD=""
if command -v timeout &> /dev/null; then
    TIMEOUT_CMD="timeout"
elif command -v gtimeout &> /dev/null; then
    TIMEOUT_CMD="gtimeout"
fi

# Minimum bitrate in bytes per second — files below this are incomplete downloads
MIN_BITRATE_BPS=1000
# Only apply bitrate check to files with duration above this threshold (seconds)
MIN_DURATION_FOR_BITRATE_CHECK=60

echo "Checking video file integrity"
echo "=========================================="
echo ""

total_scanned=0
total_corrupt=0
corrupt_files=()

# Iterate over all channel folders in videos directory
while IFS= read -r channel_dir; do
    channel_name=$(basename "$channel_dir")

    channel_corrupt=0
    channel_scanned=0

    echo "Processing channel: $channel_name"

    # Process all video files matching the whitelist
    while IFS= read -r -d '' input_file; do

        filename=$(basename "$input_file")

        # Skip macOS metadata files
        if [[ "$filename" == ._* ]]; then
            continue
        fi

        # Validate file extension against whitelist
        extension="${filename##*.}"
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

        # Skip files modified within the last 60 seconds (still downloading)
        current_time=$(date +%s)
        file_modified_time=$(stat -f %m "$input_file" 2>/dev/null || stat -c %Y "$input_file" 2>/dev/null)
        time_since_modification=$((current_time - file_modified_time))

        if [ $time_since_modification -lt 60 ]; then
            if [ "$VERBOSE" = "true" ]; then
                echo "  ⏸️  Skipping: $filename (modified ${time_since_modification}s ago)"
            fi
            continue
        fi

        channel_scanned=$((channel_scanned + 1))
        total_scanned=$((total_scanned + 1))

        # Print progress every 100 files
        if [ $((total_scanned % 100)) -eq 0 ]; then
            printf "  ... checked %d files so far (%d corrupt)\n" "$total_scanned" "$total_corrupt"
        fi

        file_size=$(stat -f %z "$input_file" 2>/dev/null || stat -c %s "$input_file" 2>/dev/null)

        # Check 1: Can ffprobe read the file and return a valid duration?
        if [ -n "$TIMEOUT_CMD" ]; then
            duration=$($TIMEOUT_CMD 30 ffprobe -v error -show_entries format=duration \
                -of default=noprint_wrappers=1:nokey=1 "$input_file" 2>/dev/null)
        else
            duration=$(ffprobe -v error -show_entries format=duration \
                -of default=noprint_wrappers=1:nokey=1 "$input_file" 2>/dev/null)
        fi

        if [[ -z "$duration" ]] || [[ "$duration" == "N/A" ]]; then
            echo "  🚨 CORRUPT (unreadable): $channel_name/$filename [$file_size bytes]"
            total_corrupt=$((total_corrupt + 1))
            channel_corrupt=$((channel_corrupt + 1))
            corrupt_files+=("$input_file")
            continue
        fi

        # Validate duration is a number
        if ! [[ "$duration" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "  🚨 CORRUPT (invalid duration '$duration'): $channel_name/$filename [$file_size bytes]"
            total_corrupt=$((total_corrupt + 1))
            channel_corrupt=$((channel_corrupt + 1))
            corrupt_files+=("$input_file")
            continue
        fi

        # Check 2: For files longer than MIN_DURATION_FOR_BITRATE_CHECK seconds,
        # verify the bitrate is above MIN_BITRATE_BPS (catches incomplete downloads
        # where the header is valid but the file is truncated)
        duration_int=${duration%%.*}
        if [ "$duration_int" -gt "$MIN_DURATION_FOR_BITRATE_CHECK" ]; then
            bitrate=$(echo "$file_size $duration" | awk '{printf "%.0f", $1 / $2}')
            if [ "$bitrate" -lt "$MIN_BITRATE_BPS" ]; then
                echo "  🚨 CORRUPT (incomplete download): $channel_name/$filename [${file_size}B / ${duration}s = ${bitrate} B/s]"
                total_corrupt=$((total_corrupt + 1))
                channel_corrupt=$((channel_corrupt + 1))
                corrupt_files+=("$input_file")
                continue
            fi
        fi

        if [ "$VERBOSE" = "true" ]; then
            echo "  ✅ OK: $channel_name/$filename"
        fi

    done < <(find "$channel_dir" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.webm" -o -name "*.m4a" -o -name "*.mov" -o -name "*.m4v" -o -name "*.avi" -o -name "*.flv" \) -print0)

    if [ $channel_scanned -gt 0 ]; then
        if [ $channel_corrupt -gt 0 ]; then
            echo "  $channel_name: $channel_corrupt corrupt out of $channel_scanned files"
        elif [ "$VERBOSE" = "true" ]; then
            echo "  $channel_name: all $channel_scanned files OK"
        fi
    fi

done < <(find "$VIDEOS_DIR" -mindepth 1 -maxdepth 1 -type d)

echo ""
echo "=========================================="
echo "Total scanned: $total_scanned"
echo "Total corrupt: $total_corrupt"
echo "Total clean:   $((total_scanned - total_corrupt))"

if [ $total_corrupt -gt 0 ]; then
    echo ""
    echo "Corrupt files:"
    for f in "${corrupt_files[@]}"; do
        # Extract video ID from filename (yt-dlp format: "Title [VIDEO_ID].ext")
        base=$(basename "$f")
        video_id=""
        if [[ "$base" =~ \[([A-Za-z0-9_-]{11})\]\.[a-zA-Z0-9]+$ ]]; then
            video_id="${BASH_REMATCH[1]}"
        fi
        if [ -n "$video_id" ]; then
            echo "  $f (video_id: $video_id)"
        else
            echo "  $f"
        fi
    done
    echo ""
    printf "\033[31m✗ check-video-integrity failed: $total_corrupt corrupt file(s) found\033[0m\n"
    exit 1
else
    echo ""
    printf "\033[32m✓ check-video-integrity passed: all files OK\033[0m\n"
    exit 0
fi
