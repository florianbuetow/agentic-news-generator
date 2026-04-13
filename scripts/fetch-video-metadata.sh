#!/bin/bash

# fetch-video-metadata: Fetch .info.json for a single YouTube video ID without downloading video
# Writes exactly to OUTPUT_FILE_BASE.info.json (preserving any existing stem from the pipeline)
# Usage: ./scripts/fetch-video-metadata.sh OUTPUT_FILE_BASE VIDEO_ID

source "$(dirname "$0")/config.sh"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./scripts/fetch-video-metadata.sh OUTPUT_FILE_BASE VIDEO_ID"
    echo "Example: ./scripts/fetch-video-metadata.sh '/path/to/metadata/channel/video/Existing Title [DOez-RwJ7mg]' DOez-RwJ7mg"
    exit 1
fi

OUTPUT_FILE_BASE="$1"
VIDEO_ID="$2"

mkdir -p "$(dirname "$OUTPUT_FILE_BASE")"

echo "Fetching metadata for $VIDEO_ID into: $OUTPUT_FILE_BASE.info.json"

yt-dlp \
    --cookies-from-browser "$BROWSER" \
    --skip-download \
    --write-info-json \
    -o "$OUTPUT_FILE_BASE.%(ext)s" \
    "https://www.youtube.com/watch?v=$VIDEO_ID"
