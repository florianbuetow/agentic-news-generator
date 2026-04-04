#!/bin/bash
# Remove transcript file groups where the .srt is empty (0 bytes).
# These are whisper outputs for videos with no speech.

TRANSCRIPTS_DIR="$1"

if [ -z "$TRANSCRIPTS_DIR" ] || [ ! -d "$TRANSCRIPTS_DIR" ]; then
    echo "Usage: $0 <transcripts_dir>" >&2
    exit 1
fi

count=0
while IFS= read -r srt; do
    base="${srt%.srt}"
    echo "Removing empty transcript group: $(basename "$base")"
    for ext in srt txt vtt tsv json; do
        f="${base}.${ext}"
        if [ -f "$f" ]; then
            rm "$f"
            echo "  deleted: $(basename "$f")"
        fi
    done
    count=$((count + 1))
done < <(find "$TRANSCRIPTS_DIR" -name "*.srt" -empty)

echo ""
echo "Removed $count empty transcript group(s)"
