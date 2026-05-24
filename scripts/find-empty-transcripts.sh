#!/usr/bin/env bash
#
# Find all transcript files (*.txt) that are 100 bytes or smaller across the
# transcripts directory defined in config/config.yaml.
#

set -euo pipefail

config="config/config.yaml"
threshold=100

if [ ! -f "$config" ]; then
    echo "Error: $config not found" >&2
    exit 1
fi

# Extract data_downloads_transcripts_dir from config
transcripts_dir=$(awk '/^paths:/{found=1; next} found && /^[^ ]/{exit} found && /data_downloads_transcripts_dir:/{sub(/.*data_downloads_transcripts_dir: */,""); sub(/\/.*/,""); print; exit}' "$config")
transcripts_dir=$(awk '/^paths:/{found=1; next} found && /^[^ ]/{exit} found && /data_downloads_transcripts_dir:/{sub(/.*data_downloads_transcripts_dir: */,""); print; exit}' "$config" | tr -d ' ')
transcripts_dir="${transcripts_dir%/}"

if [ -z "$transcripts_dir" ]; then
    echo "Error: data_downloads_transcripts_dir not found in $config" >&2
    exit 1
fi

if [ ! -d "$transcripts_dir" ]; then
    echo "Error: Transcripts directory not found: $transcripts_dir" >&2
    exit 1
fi

found=0

while IFS= read -r file; do
    size=$(wc -c < "$file" | tr -d ' ')
    if [ "$size" -le "$threshold" ]; then
        found=1
        # Extract channel name and filename relative to transcripts_dir
        rel="${file#"$transcripts_dir"/}"
        channel="${rel%%/*}"
        filename="${rel#*/}"
        # Extract video ID from filename (last [...] before extension)
        video_id=$(echo "$filename" | sed 's/.*\[\([^]]*\)\]\.[^.]*$/\1/')
        printf "  [%s] %s bytes  %s\n" "$channel" "$size" "$filename"
    fi
done < <(find "$transcripts_dir" -maxdepth 2 -type f \( -name "*.txt" -o -name "*.vtt" -o -name "*.tsv" -o -name "*.srt" \) | sort)

if [ "$found" -eq 0 ]; then
    printf "\033[0;32m✓ No empty transcripts found (threshold: %d bytes)\033[0m\n" "$threshold"
fi
