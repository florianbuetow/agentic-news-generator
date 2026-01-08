#!/bin/bash

# move-metadata.sh: Move .info.json files from video folders to metadata folders
# Usage: ./scripts/move-metadata.sh

# Source central configuration
source "$(dirname "$0")/config.sh"

# Create metadata directory if it doesn't exist
mkdir -p "$METADATA_DIR"

echo "Moving metadata files from videos to metadata folders..."
echo ""

# Track statistics
total_moved=0
total_errors=0

# Iterate through all channel directories
for channel_dir in "$VIDEOS_DIR"/*; do
    # Skip if not a directory
    if [ ! -d "$channel_dir" ]; then
        continue
    fi

    # Get channel name (basename of the directory)
    channel_name=$(basename "$channel_dir")

    # Create corresponding metadata video subdirectory
    metadata_video_dir="$METADATA_DIR/$channel_name/$METADATA_VIDEO_SUBDIR"
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

        if [ "$json_type" = "video" ]; then
            # Video metadata - move to video/ subdirectory
            if mv -f "$json_file" "$metadata_video_dir/$filename"; then
                ((moved++))
            else
                echo "   ✗ Failed to move: $filename"
                ((errors++))
            fi
        elif [ "$json_type" = "playlist" ]; then
            # Channel/playlist metadata - move to channel root directory
            if mv -f "$json_file" "$METADATA_DIR/$channel_name/$filename"; then
                ((channel_metadata_count++))
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
    if [ "$errors" -gt 0 ]; then
        echo "   ✗ Errors: $errors"
    fi
    echo ""

    ((total_moved += moved))
    ((total_errors += errors))
done

# Summary
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "Summary: $total_moved file(s) moved, $total_errors error(s)"
echo ""

# Exit with error if any failed
if [ "$total_errors" -gt 0 ]; then
    exit 1
fi

exit 0
