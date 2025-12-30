#!/bin/bash

# yt-downloader: Download YouTube videos from the past n days
# Usage: ./scripts/yt-downloader.sh 'URL'
# Note: Always use single quotes around the URL to prevent shell expansion
# Optimized for large channels with hundreds of videos

BROWSER="chrome"
if [ -z "$1" ]; then
    echo "Usage: ./scripts/yt-downloader.sh 'URL'"
    echo "Example: ./scripts/yt-downloader.sh 'https://www.youtube.com/@channelname/videos'"
    echo "Example: ./scripts/yt-downloader.sh 'https://www.youtube.com/watch?v=VIDEO_ID'"
    exit 1
fi

# Create download archive directory if it doesn't exist
ARCHIVE_DIR="$HOME/scripts/yt-download"
mkdir -p "$ARCHIVE_DIR"

yt-dlp --cookies-from-browser $BROWSER \
       --download-archive "$ARCHIVE_DIR/downloaded.txt" \
       --dateafter now-1days \
       --lazy-playlist \
       --break-match-filter \
       --format "bestvideo+bestaudio/best" \
       "$1"

