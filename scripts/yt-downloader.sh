#!/bin/bash

# yt-downloader: Download YouTube videos from the past n days
# Usage: ./scripts/yt-downloader.sh 'URL' 'OUTPUT_DIR'
# Note: Always use single quotes around the URL to prevent shell expansion
# Optimized for large channels with hundreds of videos

# Source central configuration
source "$(dirname "$0")/config.sh"
if [ -z "$1" ]; then
    echo "Usage: ./scripts/yt-downloader.sh 'URL' 'OUTPUT_DIR' 'MAX_DOWNLOADS'"
    echo "Example: ./scripts/yt-downloader.sh 'https://www.youtube.com/@channelname/videos' 'data/downloads/videos/channelname' '20'"
    echo "Example: ./scripts/yt-downloader.sh 'https://www.youtube.com/watch?v=VIDEO_ID' 'data/downloads/videos/channelname' '1'"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: OUTPUT_DIR is required"
    echo "Usage: ./scripts/yt-downloader.sh 'URL' 'OUTPUT_DIR' 'MAX_DOWNLOADS'"
    exit 1
fi

if [ -z "$3" ]; then
    echo "Error: MAX_DOWNLOADS is required"
    echo "Usage: ./scripts/yt-downloader.sh 'URL' 'OUTPUT_DIR' 'MAX_DOWNLOADS'"
    exit 1
fi

URL="$1"
OUTPUT_DIR="$2"
MAX_DOWNLOADS="$3"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create download archive file in the output directory
ARCHIVE_FILE="$OUTPUT_DIR/downloaded.txt"

# Run yt-dlp and capture exit code
yt-dlp --cookies-from-browser $BROWSER \
       --download-archive "$ARCHIVE_FILE" \
       --no-abort-on-error \
       --ignore-errors \
       --lazy-playlist \
       --match-filter "availability=public" \
       --match-filters "!is_live" \
       --playlist-items 1-$MAX_DOWNLOADS \
       --min-sleep-interval 1 \
       --max-sleep-interval 5 \
       --write-info-json \
       -f "bestvideo+(bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio)+best" \
       -o "$OUTPUT_DIR/%(title)s [%(id)s].%(ext)s" \
       "$URL"

exit_code=$?

# Exit code 0: All videos succeeded
# Exit code 1: Some videos failed (private, members-only, deleted, etc.)
# Exit code > 1: Fatal error (network, auth, etc.)
#
# We treat exit code 1 as success since we expect some videos to be unavailable
# and the goal is to download as many videos as possible, not all of them.
if [ $exit_code -eq 0 ] || [ $exit_code -eq 1 ]; then
    exit 0
else
    exit $exit_code
fi
