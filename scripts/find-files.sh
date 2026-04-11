#!/usr/bin/env bash
#
# Find all files containing a video ID substring across all data directories
# defined in config/config.yaml.
#

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <video-id>" >&2
    exit 1
fi

VIDEO_ID="$1"
CONFIG="config/config.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "Error: $CONFIG not found" >&2
    exit 1
fi

# Collect all paths, strip trailing slashes
declare -a keys=()
declare -a paths=()

while IFS=': ' read -r key path; do
    [ -z "$key" ] && continue
    [ -z "$path" ] && continue
    keys+=("$key")
    paths+=("${path%/}")
done < <(awk '/^paths:/{found=1; next} found && /^[^ ]/{exit} found && /^ /{sub(/^ +/,""); print}' "$CONFIG")

# Keep only shortest unique prefixes: skip a path if a shorter path already covers it
declare -a use_keys=()
declare -a use_paths=()

for i in "${!keys[@]}"; do
    path="${paths[$i]}"
    is_child=0
    for j in "${!paths[@]}"; do
        [ "$i" -eq "$j" ] && continue
        other="${paths[$j]}"
        # If another path is a proper prefix of this one, skip this one
        if [[ "$path" == "$other"/* ]]; then
            is_child=1
            break
        fi
    done
    [ "$is_child" -eq 1 ] && continue
    use_keys+=("${keys[$i]}")
    use_paths+=("$path")
done

found=0

for i in "${!use_keys[@]}"; do
    key="${use_keys[$i]}"
    path="${use_paths[$i]}"

    [ ! -d "$path" ] && continue

    # Derive category name: data_downloads_videos_dir -> Downloads / Videos
    category=$(echo "$key" | sed 's/_dir$//; s/^data_//' | tr '_' ' ' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)}1' | sed 's/ / \/ /')

    matches=$(find "$path" -type f -name "*${VIDEO_ID}*" 2>/dev/null | sort)
    [ -z "$matches" ] && continue

    found=1
    echo "${category}:"
    echo ""
    while IFS= read -r file; do
        echo "    ${file}"
    done <<< "$matches"
    echo ""
done

if [ "$found" -eq 0 ]; then
    echo "No files found containing '${VIDEO_ID}'."
fi
