#!/bin/bash

# yt-downloader: Download YouTube videos from the past n days
# Usage: ./scripts/yt-downloader.sh 'url' 'output_dir'
# Note: Always use single quotes around the url to prevent shell expansion
# Optimized for large channels with hundreds of videos

# Source central configuration
source "$(dirname "$0")/config.sh"
if [ -z "$1" ]; then
    echo "Usage: ./scripts/yt-downloader.sh 'url' 'output_dir' 'max_downloads'"
    echo "Example: ./scripts/yt-downloader.sh 'https://www.youtube.com/@channelname/videos' 'data/downloads/videos/channelname' '20'"
    echo "Example: ./scripts/yt-downloader.sh 'https://www.youtube.com/watch?v=VIDEO_ID' 'data/downloads/videos/channelname' '1'"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: output_dir is required"
    echo "Usage: ./scripts/yt-downloader.sh 'url' 'output_dir' 'max_downloads'"
    exit 1
fi

if [ -z "$3" ]; then
    echo "Error: max_downloads is required"
    echo "Usage: ./scripts/yt-downloader.sh 'url' 'output_dir' 'max_downloads'"
    exit 1
fi

url="$1"
output_dir="$2"
max_downloads="$3"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Create download archive file in the output directory
archive_file="$output_dir/downloaded.txt"

# Run yt-dlp with real-time output monitoring.
# A background watcher kills yt-dlp immediately if YouTube reports expired cookies
# or bot detection, instead of waiting for all items to fail.
# We merge stdout+stderr since yt-dlp may write warnings to either stream.
output_fifo=$(mktemp -u)
mkfifo "$output_fifo"
trap 'rm -f "$output_fifo"' EXIT

# Start yt-dlp in background, merging stdout+stderr into the FIFO
yt-dlp --cookies-from-browser $browser \
       --download-archive "$archive_file" \
       --no-abort-on-error \
       --ignore-errors \
       --lazy-playlist \
       --match-filter "availability=public & !is_live & live_status!=is_live & live_status!=is_upcoming & live_status!=post_live & duration >= 120" \
       --playlist-items 1-$max_downloads \
       --min-sleep-interval 1 \
       --max-sleep-interval 5 \
       --write-info-json \
       --write-thumbnail \
       --merge-output-format mp4 \
       -o "$output_dir/%(title)s [%(id)s].%(ext)s" \
       "$url" >"$output_fifo" 2>&1 &
yt_pid=$!

# Monitor output in real-time: display it and kill yt-dlp on fatal cookie/bot errors
cookie_error=0
while IFS= read -r line; do
    echo "$line"
    case "$line" in
        *"cookies are no longer valid"*|*"Sign in to confirm you're not a bot"*)
            cookie_error=1
            kill "$yt_pid" 2>/dev/null
            # Drain remaining output so yt-dlp doesn't block on the FIFO
            cat >/dev/null &
            break
            ;;
    esac
done <"$output_fifo"

wait "$yt_pid" 2>/dev/null
exit_code=$?

if [ "$cookie_error" -eq 1 ]; then
    echo ""
    echo "ERROR: YouTube cookies are expired or invalid. Aborting."
    echo "Re-export cookies from your browser to fix this."
    exit 10
fi

# Exit codes consumed by scripts/yt-downloader.py:
#   0  = success (all videos downloaded, OR some unavailable but no fatal error)
#   10 = cookie / bot-detection failure (handled above)
#   11 = yt-dlp reported a fatal error (network, auth, parser, etc.)
#
# yt-dlp exit codes 0 and 1 are both treated as success: code 1 means some
# items were skipped (private, members-only, deleted) which is expected and
# not actionable. Anything higher is a fatal error we want to surface.
if [ $exit_code -eq 0 ] || [ $exit_code -eq 1 ]; then
    exit 0
else
    exit 11
fi
