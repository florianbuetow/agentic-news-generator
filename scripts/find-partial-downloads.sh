#!/usr/bin/env bash
#
# Find format-code artifacts (*.f<code>.*) left behind by yt-dlp downloads whose
# ffmpeg merge never completed, across all data directories defined in
# config/config.yaml. Optionally restrict the scan to a single channel.
#
# Format-code artifacts are unmerged single streams (video-only .f137.mp4,
# audio-only .f251.webm) plus anything derived from them (.f251-8.wav,
# .f251-11.silence_map.json). See TROUBLESHOOTING-GUIDE.md, "Playbook:
# Interrupted Merge".
#

set -euo pipefail

if [ $# -gt 1 ]; then
    echo "Usage: $0 [channel]" >&2
    exit 1
fi

channel="${1:-}"
config="config/config.yaml"

if [ ! -f "$config" ]; then
    echo "Error: $config not found" >&2
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
done < <(awk '/^paths:/{found=1; next} found && /^[^ ]/{exit} found && /^ /{sub(/^ +/,""); print}' "$config")

# Keep only shortest unique prefixes: skip a path if a shorter path already covers it
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
    use_paths+=("$path")
done

# A channel that matches no directory must never read as a clean scan: this is a
# tool whose empty result authorises deletions, so an unknown channel is an error.
if [ -n "$channel" ]; then
    if ! find "${use_paths[@]}" -type d -name "$channel" -print -quit 2>/dev/null | grep -q .; then
        echo "Error: no directory named '${channel}' under any configured data path" >&2
        exit 1
    fi
fi

# Collect every format-code artifact as "basedir<TAB>fullpath".
# AppleDouble sidecars (._*) are counted separately: they are macOS metadata
# shadows of the real artifacts, not downloads in their own right.
artifacts=$(mktemp)
trap 'rm -f "$artifacts"' EXIT
sidecars=0

# A channel is a directory component somewhere below a data directory
# (downloads/videos/<CHANNEL>, downloads/metadata/<CHANNEL>/video,
# archive/videos/<CHANNEL>, ...), never a direct child of the data root — and
# after the prefix filter above, use_paths holds only that root. So restrict by
# path component instead of appending the channel to the scan root. With no
# channel, "*" matches every path, keeping a single code path for both cases.
channel_glob='*'
[ -n "$channel" ] && channel_glob="*/${channel}/*"

for path in "${use_paths[@]}"; do
    [ ! -d "$path" ] && continue

    while IFS= read -r file; do
        [ -z "$file" ] && continue
        if [[ "$(basename "$file")" == ._* ]]; then
            sidecars=$((sidecars + 1))
            continue
        fi
        printf '%s\t%s\n' "$path" "$file" >>"$artifacts"
    done < <(find "$path" -type f -path "$channel_glob" -name '*.f[0-9]*.*' 2>/dev/null)
done

# yt-dlp writes these very artifacts while downloading and only merges at the
# end, so an in-flight download is indistinguishable from an abandoned one by
# filename alone — and the remedy for the abandoned case (delete, re-download)
# destroys a live one. A running process is the exact signal; mtime windows are
# not, because a playlist run works through a channel for hours. A clean result
# is just as unreliable mid-run: the next artifact can appear a second later.
download_running=""
if pgrep -f 'yt-dlp' >/dev/null 2>&1; then
    download_running=1
fi

warn_if_downloading() {
    [ -z "$download_running" ] && return 0
    echo "⚠  A yt-dlp download is RUNNING right now — this scan is only a snapshot."
    echo "   Artifacts may be in flight rather than abandoned, and more may appear."
    echo "   Delete nothing until it exits, then re-run this scan. Check with:"
    echo "       ps -Ao command | grep '[y]t-dlp'"
    echo ""
}

if [ ! -s "$artifacts" ]; then
    if [ -n "$channel" ]; then
        echo "No partial downloads found for channel '${channel}'."
    else
        echo "No partial downloads found."
    fi
    [ "$sidecars" -gt 0 ] && echo "(${sidecars} AppleDouble sidecar(s) ignored)"
    echo ""
    warn_if_downloading
    exit 0
fi

warn_if_downloading

# Group by video ID: that is the unit the cleanup playbook operates on.
video_ids=$(grep -oE '\[[A-Za-z0-9_-]{11}\]' "$artifacts" | tr -d '[]' | sort -u)

count=0

while IFS= read -r video_id; do
    [ -z "$video_id" ] && continue
    count=$((count + 1))

    # Channel is the first path component below the data directory the file sits in.
    first_line=$(grep -m1 -F "[${video_id}]" "$artifacts")
    base="${first_line%%	*}"
    full="${first_line#*	}"
    relative="${full#"$base"/}"
    file_channel="${relative%%/*}"

    # A merged .mp4 is one carrying the video ID but no format code. Its presence
    # decides the remedy: leftovers to delete, versus a download to redo.
    # The video ID is matched with grep -F, not find -name: in a find glob the
    # brackets around an ID would be read as a character class, matching almost
    # every .mp4 on disk.
    merged=""
    for path in "${use_paths[@]}"; do
        [ ! -d "$path" ] && continue
        merged=$(find "$path" -type f -name '*.mp4' ! -name '._*' ! -name '*.f[0-9]*.mp4' ! -name '*.temp.mp4' 2>/dev/null | grep -F "[${video_id}]" | head -n1 || true)
        [ -n "$merged" ] && break
    done

    echo "[${video_id}]  ${file_channel}"
    if [ -n "$merged" ]; then
        echo "    merged .mp4 present — stale leftovers, safe to delete with their derived files"
    else
        echo "    merged .mp4 MISSING — merge never completed, needs re-download"
    fi
    echo ""

    grep -F "[${video_id}]" "$artifacts" | cut -f2- | sort | while IFS= read -r file; do
        echo "    ${file}"
    done
    echo ""
done <<<"$video_ids"

echo "Found format-code artifacts for ${count} video ID(s)."
[ "$sidecars" -gt 0 ] && echo "(${sidecars} AppleDouble sidecar(s) ignored)"
echo ""
echo "Remedy: TROUBLESHOOTING-GUIDE.md, \"Playbook: Interrupted Merge\"."
echo "Deleting a stream artifact means deleting its derived .wav and .silence_map.json too."

exit 1
