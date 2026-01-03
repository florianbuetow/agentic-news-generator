#!/bin/bash

# --- Configuration ---
PROJECT_DIR="/Volumes/2TB/agentic-news-generator.git/florian-topic-segmentation"

# VERBOSE: Set to "true" to show individual operations
VERBOSE="${VERBOSE:-false}"
# --- End Configuration ---

echo "Archiving processed videos"
echo "=========================================="
echo ""

# Create archive directory
mkdir -p "$PROJECT_DIR/data/archive/videos"

# Counters
total_archived=0
total_audio_deleted=0

# Iterate over all channel folders in transcripts directory
find "$PROJECT_DIR/data/downloads/transcripts" -mindepth 1 -maxdepth 1 -type d | while read -r channel_dir; do
    channel_name=$(basename "$channel_dir")

    echo "Processing channel: $channel_name"
    echo "---"

    # Create archive directory for this channel
    mkdir -p "$PROJECT_DIR/data/archive/videos/$channel_name"

    # Counters for this channel
    archived_count=0
    audio_deleted_count=0

    # Find all .txt transcript files
    while IFS= read -r -d '' transcript_file; do
        base_name=$(basename "$transcript_file" .txt)

        # Define paths for audio and video files
        audio_file="$PROJECT_DIR/data/downloads/audio/$channel_name/$base_name.wav"

        # Find the video file with any extension
        video_file=""
        for ext in mp4 webm m4a mov m4v avi mkv flv; do
            candidate="$PROJECT_DIR/data/downloads/videos/$channel_name/$base_name.$ext"
            if [ -f "$candidate" ]; then
                video_file="$candidate"
                break
            fi
        done

        # Delete audio file if it exists
        if [ -f "$audio_file" ]; then
            rm "$audio_file"
            audio_deleted_count=$((audio_deleted_count + 1))
            if [ "$VERBOSE" = "true" ]; then
                echo "  🗑️  Deleted audio: $base_name.wav"
            fi
        fi

        # Move video file to archive if it exists
        if [ -n "$video_file" ] && [ -f "$video_file" ]; then
            video_filename=$(basename "$video_file")
            mv "$video_file" "$PROJECT_DIR/data/archive/videos/$channel_name/$video_filename"
            archived_count=$((archived_count + 1))
            if [ "$VERBOSE" = "true" ]; then
                echo "  📦 Archived video: $video_filename"
            fi
        fi

    done < <(find "$channel_dir" -maxdepth 1 -type f -name "*.txt" -print0 2>/dev/null)

    if [ $archived_count -gt 0 ] || [ $audio_deleted_count -gt 0 ]; then
        echo "📦 Archived $archived_count video(s)"
        echo "🗑️  Deleted $audio_deleted_count audio file(s)"
    else
        echo "No files to archive"
    fi

    echo ""
done

echo "Video archiving complete."
