#!/bin/bash

# Source central configuration
source "$(dirname "$0")/config.sh"

echo "Moving transcript metadata JSON files to metadata directory..."
echo "=========================================="
echo ""

# Counter for moved files
total_moved=0
total_skipped=0

# Iterate over all channel folders in transcripts directory
if [ ! -d "$TRANSCRIPTS_DIR" ]; then
    echo "⚠️  Transcripts directory does not exist: $TRANSCRIPTS_DIR"
    exit 0
fi

while IFS= read -r channel_dir; do
    channel_name=$(basename "$channel_dir")

    # Skip macOS metadata directories
    if [[ "$channel_name" == ._* ]]; then
        continue
    fi

    # Create metadata/transcript directory for this channel
    metadata_transcript_dir="$METADATA_DIR/$channel_name/transcript"
    mkdir -p "$metadata_transcript_dir"

    # Count JSON files in this channel
    json_count=$(find "$channel_dir" -maxdepth 1 -type f -name "*.json" ! -name "._*" | wc -l | tr -d ' ')

    if [ "$json_count" -eq 0 ]; then
        if [ "$VERBOSE" = "true" ]; then
            echo "  ⏭️  Skipping channel: $channel_name (no JSON files)"
        fi
        continue
    fi

    echo "Processing channel: $channel_name"
    echo "  Found $json_count JSON file(s)"

    moved_count=0
    skipped_count=0

    # Find and move all .json files (excluding macOS metadata files)
    while IFS= read -r -d '' json_file; do
        base_name=$(basename "$json_file")

        # Skip macOS hidden metadata files
        if [[ "$base_name" == ._* ]]; then
            continue
        fi

        dest_file="$metadata_transcript_dir/$base_name"

        # Check if file already exists in destination
        if [ -f "$dest_file" ]; then
            skipped_count=$((skipped_count + 1))
            if [ "$VERBOSE" = "true" ]; then
                echo "    ⏭️  Skipping: $base_name (already exists)"
            fi
            continue
        fi

        # Move the file
        mv "$json_file" "$dest_file"

        if [ $? -eq 0 ]; then
            moved_count=$((moved_count + 1))
            if [ "$VERBOSE" = "true" ]; then
                echo "    ✅ Moved: $base_name"
            fi
        else
            echo "    🚨 FAILED to move: $base_name"
        fi

    done < <(find "$channel_dir" -maxdepth 1 -type f -name "*.json" -print0 2>/dev/null)

    # Update totals
    total_moved=$((total_moved + moved_count))
    total_skipped=$((total_skipped + skipped_count))

    # Print channel summary
    if [ $moved_count -gt 0 ]; then
        echo "  ✅ Moved $moved_count file(s)"
    fi
    if [ $skipped_count -gt 0 ]; then
        echo "  ⏭️  Skipped $skipped_count file(s) (already exists)"
    fi

    echo ""
done < <(find "$TRANSCRIPTS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)

# Print final summary
echo "=========================================="
echo "Summary:"
echo "  Total moved: $total_moved file(s)"
if [ $total_skipped -gt 0 ]; then
    echo "  Total skipped: $total_skipped file(s)"
fi
echo ""
echo "✅ Transcript metadata organization complete."
