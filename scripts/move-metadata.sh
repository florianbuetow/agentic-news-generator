#!/bin/bash

# move-metadata.sh: Move .info.json files from video folders to metadata folders
# Usage: ./scripts/move-metadata.sh

# Source central configuration
source "$(dirname "$0")/config.sh"

# Create metadata directory if it doesn't exist
mkdir -p "$metadata_dir"

echo "Moving metadata files from videos to metadata folders..."
echo ""

# Thumbnail extensions yt-dlp may produce from YouTube responses (see scripts/fetch-video-thumbnails.py).
thumbnail_extensions=(jpg webp png)

# Move sibling thumbnail files matching the .info.json stem to dest_dir.
# Args: $1=source_channel_dir $2=stem $3=dest_dir
# Updates caller's `thumbs_moved` and `errors` counters.
move_sibling_thumbnails() {
    local src_dir="$1"
    local stem="$2"
    local dest_dir="$3"
    local ext thumb
    for ext in "${thumbnail_extensions[@]}"; do
        thumb="$src_dir/$stem.$ext"
        if [ -f "$thumb" ]; then
            if mv -f "$thumb" "$dest_dir/$stem.$ext"; then
                ((thumbs_moved++))
            else
                echo "   ✗ Failed to move thumbnail: $stem.$ext"
                ((errors++))
            fi
        fi
    done
}

# Track statistics
total_moved=0
total_errors=0
total_thumbs_moved=0

# Iterate through all channel directories
for channel_dir in "$videos_dir"/*; do
    # Skip if not a directory
    if [ ! -d "$channel_dir" ]; then
        continue
    fi

    # Get channel name (basename of the directory)
    channel_name=$(basename "$channel_dir")

    # Create corresponding metadata video subdirectory
    metadata_video_dir="$metadata_dir/$channel_name/$metadata_video_subdir"
    mkdir -p "$metadata_video_dir"

    # Count JSON files in this channel
    json_count=$(find "$channel_dir" -maxdepth 1 -name "*.info.json" -type f | wc -l)

    if [ "$json_count" -eq 0 ]; then
        continue
    fi

    echo "📂 Processing channel: $channel_name"
    echo "   Found $json_count metadata file(s)"

    # Move all .info.json files
    moved=0
    errors=0
    channel_metadata_count=0
    thumbs_moved=0
    for json_file in "$channel_dir"/*.info.json; do
        # Skip if no files match (glob didn't expand)
        if [ ! -f "$json_file" ]; then
            continue
        fi

        filename=$(basename "$json_file")

        # Use jq to check the _type field to distinguish channel vs video metadata
        # Channel/playlist metadata has _type: "playlist"
        # Video metadata has _type: "video"
        json_type=$(jq -r '._type // "unknown"' "$json_file" 2>/dev/null)

        stem="${filename%.info.json}"
        if [ "$json_type" = "video" ]; then
            # Video metadata - move to video/ subdirectory
            if mv -f "$json_file" "$metadata_video_dir/$filename"; then
                ((moved++))
                move_sibling_thumbnails "$channel_dir" "$stem" "$metadata_video_dir"
            else
                echo "   ✗ Failed to move: $filename"
                ((errors++))
            fi
        elif [ "$json_type" = "playlist" ]; then
            # Channel/playlist metadata - move to channel root directory
            if mv -f "$json_file" "$metadata_dir/$channel_name/$filename"; then
                ((channel_metadata_count++))
                move_sibling_thumbnails "$channel_dir" "$stem" "$metadata_dir/$channel_name"
            else
                echo "   ✗ Failed to move channel metadata: $filename"
                ((errors++))
            fi
        else
            # Unknown type - skip with warning
            echo "   ⚠️  Skipping unknown metadata type: $filename (type: $json_type)"
        fi
    done

    echo "   ✓ Moved $moved video metadata file(s)"
    if [ "$channel_metadata_count" -gt 0 ]; then
        echo "   ✓ Moved $channel_metadata_count channel metadata file(s)"
    fi
    if [ "$thumbs_moved" -gt 0 ]; then
        echo "   ✓ Moved $thumbs_moved thumbnail file(s)"
    fi
    if [ "$errors" -gt 0 ]; then
        echo "   ✗ Errors: $errors"
    fi
    echo ""

    ((total_moved += moved))
    ((total_errors += errors))
    ((total_thumbs_moved += thumbs_moved))
done

# Summary
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "Summary: $total_moved metadata file(s) moved, $total_thumbs_moved thumbnail(s) moved, $total_errors error(s)"
echo ""

# Exit with error if any failed
if [ "$total_errors" -gt 0 ]; then
    exit 1
fi

exit 0
