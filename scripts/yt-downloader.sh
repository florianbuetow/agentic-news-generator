#!/bin/bash

# yt-downloader: Download YouTube videos from the past n days
# Usage: ./scripts/yt-downloader.sh 'URL' 'OUTPUT_DIR'
# Note: Always use single quotes around the URL to prevent shell expansion
# Optimized for large channels with hundreds of videos

BROWSER="chrome"
if [ -z "$1" ]; then
    echo "Usage: ./scripts/yt-downloader.sh 'URL' 'OUTPUT_DIR'"
    echo "Example: ./scripts/yt-downloader.sh 'https://www.youtube.com/@channelname/videos' 'data/downloads/video/channelname'"
    echo "Example: ./scripts/yt-downloader.sh 'https://www.youtube.com/watch?v=VIDEO_ID' 'data/downloads/video/channelname'"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: OUTPUT_DIR is required"
    echo "Usage: ./scripts/yt-downloader.sh 'URL' 'OUTPUT_DIR'"
    exit 1
fi

URL="$1"
OUTPUT_DIR="$2"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create download archive file in the output directory
ARCHIVE_FILE="$OUTPUT_DIR/downloaded.txt"

# TODO add --dateafter now-1days filter and remove the --max-downloads filter


yt-dlp --cookies-from-browser $BROWSER \
       --download-archive "$ARCHIVE_FILE" \
       --lazy-playlist \
       --match-filters "!is_live" \
       --max-downloads 1 \
       --sleep-interval 5 \
       -f "bestvideo+(bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio)+best" \
       -o "$OUTPUT_DIR/%(title)s.%(ext)s" \
       "$URL"
