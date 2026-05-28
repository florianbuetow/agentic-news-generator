#!/bin/bash

# fetch-video-thumbnail: Fetch best thumbnail for a single YouTube video ID without downloading video
# Writes to output_file_base.<original-ext> (jpg or webp, whatever YouTube serves)
# Usage: ./scripts/fetch-video-thumbnail.sh output_file_base video_id

source "$(dirname "$0")/config.sh"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./scripts/fetch-video-thumbnail.sh output_file_base video_id"
    echo "Example: ./scripts/fetch-video-thumbnail.sh '/path/to/metadata/channel/video/Existing Title [DOez-RwJ7mg]' DOez-RwJ7mg"
    exit 1
fi

output_file_base="$1"
video_id="$2"

mkdir -p "$(dirname "$output_file_base")"

echo "Fetching thumbnail for $video_id into: $output_file_base.<ext>"

yt-dlp \
    --cookies-from-browser "$browser" \
    --skip-download \
    --write-thumbnail \
    --no-write-info-json \
    --no-overwrites \
    -o "$output_file_base.%(ext)s" \
    "https://www.youtube.com/watch?v=$video_id"
