#!/bin/bash

# fetch-video-metadata: Fetch .info.json for a single YouTube video ID without downloading video
# Writes exactly to output_file_base.info.json (preserving any existing stem from the pipeline)
# Usage: ./scripts/fetch-video-metadata.sh output_file_base video_id

source "$(dirname "$0")/config.sh"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./scripts/fetch-video-metadata.sh output_file_base video_id"
    echo "Example: ./scripts/fetch-video-metadata.sh '/path/to/metadata/channel/video/Existing Title [DOez-RwJ7mg]' DOez-RwJ7mg"
    exit 1
fi

output_file_base="$1"
video_id="$2"

mkdir -p "$(dirname "$output_file_base")"

echo "Fetching metadata for $video_id into: $output_file_base.info.json"

yt-dlp \
    --cookies-from-browser "$browser" \
    --skip-download \
    --write-info-json \
    -o "$output_file_base.%(ext)s" \
    "https://www.youtube.com/watch?v=$video_id"
