#!/usr/bin/env bash
#
# Copy every file found for a video ID (via find-files.sh) into a destination
# directory. Files are nested under their last two source directory
# components so same-named files from different data categories (e.g.
# transcripts vs. transcripts_cleaned) don't collide.
#

set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <video-id> [destination-dir]" >&2
    exit 1
fi

video_id="$1"
dest_dir="${2:-$HOME/Downloads/export-${video_id}}"

matches=$(scripts/find-files.sh "$video_id" | sed -n 's/^    //p')

if [ -z "$matches" ]; then
    echo "No files found containing '${video_id}'."
    exit 1
fi

mkdir -p "$dest_dir"

copied=0
while IFS= read -r file; do
    parent="$(basename "$(dirname "$file")")"
    grandparent="$(basename "$(dirname "$(dirname "$file")")")"
    target_dir="${dest_dir}/${grandparent}/${parent}"
    mkdir -p "$target_dir"
    echo "Copying: ${file}"
    cp "$file" "${target_dir}/"
    copied=$((copied + 1))
done <<< "$matches"

echo ""
echo "Copied ${copied} file(s) to ${dest_dir}"
