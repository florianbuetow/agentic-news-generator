#!/usr/bin/env bash
#
# Check whether a downloaded video file contains an audible audio track.
#
# Usage: scripts/check-audio-track.sh <channel> <video-id>
#
# Resolves the file under data_downloads_videos_dir/<channel>/ matching the
# given video id in its filename, then uses ffprobe to determine whether the
# container has at least one audio stream and whether that stream carries any
# samples. If an audio stream is present and non-empty, runs ffmpeg's
# volumedetect filter over the full audio to report mean / max volume in dB
# and flags files whose mean level sits below low_volume_threshold_db
# (default -40 dB — matches convert_to_audio.sh silence threshold, so such
# files get mostly eaten by silence removal).
#
# Exits 0 when audio is present and above the low-volume threshold, 1 when
# audio is missing / empty / too quiet, 2 on usage / lookup error.
#

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <channel> <video-id>" >&2
    exit 2
fi

channel="$1"
video_id="$2"
config="config/config.yaml"

# Defensive: reject path traversal / slashes in channel / id so we can never
# escape the videos directory even with hostile input.
case "$channel" in
    */*|..|.|"")    echo "Error: invalid channel: $channel" >&2; exit 2 ;;
esac
case "$video_id" in
    */*|..|.|"")    echo "Error: invalid video id: $video_id" >&2; exit 2 ;;
esac

if [ ! -f "$config" ]; then
    echo "Error: $config not found (run from project root)" >&2
    exit 2
fi

videos_dir=$(awk '/data_downloads_videos_dir:/ {sub(/\/$/,"",$2); print $2; exit}' "$config")

if [ -z "$videos_dir" ]; then
    echo "Error: data_downloads_videos_dir not defined in $config" >&2
    exit 2
fi

channel_dir="${videos_dir}/${channel}"
if [ ! -d "$channel_dir" ]; then
    echo "Error: channel directory not found: $channel_dir" >&2
    exit 2
fi

if ! which ffprobe >/dev/null 2>&1; then
    echo "Error: ffprobe not installed" >&2
    exit 2
fi

# Locate the single video file for the given id. yt-dlp writes the id in
# square brackets, e.g. "Title [abcdEFG1234].mp4". `*[[]...[]]*` is the
# portable glob for a literal `[...]` substring.
matches=$(find "$channel_dir" -maxdepth 1 -type f \
    -name "*[[]${video_id}[]]*" \
    ! -name '._*' \
    ! -name '*.part' \
    2>/dev/null)

if [ -z "$matches" ]; then
    echo "NOT_FOUND	${channel}	${video_id}	-"
    exit 2
fi

match_count=$(printf '%s\n' "$matches" | wc -l | tr -d ' ')
if [ "$match_count" -gt 1 ]; then
    echo "AMBIGUOUS	${channel}	${video_id}	${match_count} matches" >&2
    echo "$matches" >&2
    exit 2
fi

file="$matches"

# Count audio streams in the container.
audio_stream_count=$(
    ffprobe -v error \
        -select_streams a \
        -show_entries stream=index \
        -of csv=p=0 \
        "$file" 2>/dev/null | grep -c . || true
)

if [ "$audio_stream_count" -eq 0 ]; then
    echo "NO_AUDIO_STREAM	${channel}	${video_id}	${file}"
    exit 1
fi

# A stream may exist but be empty (0 packets) or silent. Pull codec, channels,
# sample rate and duration individually so the output labels are never mixed
# up by ffprobe's internal field ordering. `-count_packets` scans the index
# only, so this stays cheap even on multi-GB files.
ffprobe_a0() {
    ffprobe -v error -select_streams a:0 \
        -show_entries "stream=$1" \
        -of default=nw=1:nk=1 \
        "$file" 2>/dev/null
}

codec=$(ffprobe_a0 codec_name)
channels=$(ffprobe_a0 channels)
sample_rate=$(ffprobe_a0 sample_rate)
duration=$(ffprobe_a0 duration)
packet_count=$(ffprobe -v error -select_streams a:0 -count_packets \
    -show_entries stream=nb_read_packets -of default=nw=1:nk=1 \
    "$file" 2>/dev/null || echo 0)
: "${packet_count:=0}"

if [ "${packet_count:-0}" = "0" ] || [ "${packet_count:-0}" = "N/A" ]; then
    echo "EMPTY_AUDIO_STREAM	${channel}	${video_id}	codec=${codec:-?} ch=${channels:-?} sr=${sample_rate:-?} dur=${duration:-?} pkts=${packet_count}	${file}"
    exit 1
fi

# Full-decode volumedetect pass. `-vn` skips the video stream, so this stays
# proportional to audio duration only. Captures the whole stderr because
# ffmpeg writes the filter report there even on success.
low_volume_threshold_db="-40"

vol_output=$(ffmpeg -hide_banner -nostats -i "$file" -vn -af volumedetect -f null /dev/null 2>&1 || true)
mean_volume=$(printf '%s\n' "$vol_output" | awk -F': ' '/mean_volume:/ {sub(/ dB$/,"",$2); print $2; exit}')
max_volume=$(printf '%s\n' "$vol_output" | awk -F': ' '/max_volume:/ {sub(/ dB$/,"",$2); print $2; exit}')

vol_details="mean=${mean_volume:-?}dB max=${max_volume:-?}dB"
base_details="codec=${codec:-?} ch=${channels:-?} sr=${sample_rate:-?} dur=${duration:-?} pkts=${packet_count} ${vol_details}"

if [ -n "$mean_volume" ]; then
    is_low=$(awk -v m="$mean_volume" -v t="$low_volume_threshold_db" \
        'BEGIN {print (m+0 < t+0) ? 1 : 0}')
    if [ "$is_low" = "1" ]; then
        echo "LOW_VOLUME	${channel}	${video_id}	${base_details} threshold=${low_volume_threshold_db}dB	${file}"
        exit 1
    fi
fi

echo "HAS_AUDIO	${channel}	${video_id}	${base_details}	${file}"
exit 0
