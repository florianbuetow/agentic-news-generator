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
# and flags files whose mean level sits below LOW_VOLUME_THRESHOLD_DB
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

CHANNEL="$1"
VIDEO_ID="$2"
CONFIG="config/config.yaml"

# Defensive: reject path traversal / slashes in channel / id so we can never
# escape the videos directory even with hostile input.
case "$CHANNEL" in
    */*|..|.|"")    echo "Error: invalid channel: $CHANNEL" >&2; exit 2 ;;
esac
case "$VIDEO_ID" in
    */*|..|.|"")    echo "Error: invalid video id: $VIDEO_ID" >&2; exit 2 ;;
esac

if [ ! -f "$CONFIG" ]; then
    echo "Error: $CONFIG not found (run from project root)" >&2
    exit 2
fi

VIDEOS_DIR=$(awk '/data_downloads_videos_dir:/ {sub(/\/$/,"",$2); print $2; exit}' "$CONFIG")

if [ -z "$VIDEOS_DIR" ]; then
    echo "Error: data_downloads_videos_dir not defined in $CONFIG" >&2
    exit 2
fi

CHANNEL_DIR="${VIDEOS_DIR}/${CHANNEL}"
if [ ! -d "$CHANNEL_DIR" ]; then
    echo "Error: channel directory not found: $CHANNEL_DIR" >&2
    exit 2
fi

if ! which ffprobe >/dev/null 2>&1; then
    echo "Error: ffprobe not installed" >&2
    exit 2
fi

# Locate the single video file for the given id. yt-dlp writes the id in
# square brackets, e.g. "Title [abcdEFG1234].mp4". `*[[]...[]]*` is the
# portable glob for a literal `[...]` substring.
matches=$(find "$CHANNEL_DIR" -maxdepth 1 -type f \
    -name "*[[]${VIDEO_ID}[]]*" \
    ! -name '._*' \
    ! -name '*.part' \
    2>/dev/null)

if [ -z "$matches" ]; then
    echo "NOT_FOUND	${CHANNEL}	${VIDEO_ID}	-"
    exit 2
fi

match_count=$(printf '%s\n' "$matches" | wc -l | tr -d ' ')
if [ "$match_count" -gt 1 ]; then
    echo "AMBIGUOUS	${CHANNEL}	${VIDEO_ID}	${match_count} matches" >&2
    echo "$matches" >&2
    exit 2
fi

FILE="$matches"

# Count audio streams in the container.
AUDIO_STREAM_COUNT=$(
    ffprobe -v error \
        -select_streams a \
        -show_entries stream=index \
        -of csv=p=0 \
        "$FILE" 2>/dev/null | grep -c . || true
)

if [ "$AUDIO_STREAM_COUNT" -eq 0 ]; then
    echo "NO_AUDIO_STREAM	${CHANNEL}	${VIDEO_ID}	${FILE}"
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
        "$FILE" 2>/dev/null
}

CODEC=$(ffprobe_a0 codec_name)
CHANNELS=$(ffprobe_a0 channels)
SAMPLE_RATE=$(ffprobe_a0 sample_rate)
DURATION=$(ffprobe_a0 duration)
PACKET_COUNT=$(ffprobe -v error -select_streams a:0 -count_packets \
    -show_entries stream=nb_read_packets -of default=nw=1:nk=1 \
    "$FILE" 2>/dev/null || echo 0)
: "${PACKET_COUNT:=0}"

if [ "${PACKET_COUNT:-0}" = "0" ] || [ "${PACKET_COUNT:-0}" = "N/A" ]; then
    echo "EMPTY_AUDIO_STREAM	${CHANNEL}	${VIDEO_ID}	codec=${CODEC:-?} ch=${CHANNELS:-?} sr=${SAMPLE_RATE:-?} dur=${DURATION:-?} pkts=${PACKET_COUNT}	${FILE}"
    exit 1
fi

# Full-decode volumedetect pass. `-vn` skips the video stream, so this stays
# proportional to audio duration only. Captures the whole stderr because
# ffmpeg writes the filter report there even on success.
LOW_VOLUME_THRESHOLD_DB="${LOW_VOLUME_THRESHOLD_DB:--40}"

VOL_OUTPUT=$(ffmpeg -hide_banner -nostats -i "$FILE" -vn -af volumedetect -f null /dev/null 2>&1 || true)
MEAN_VOLUME=$(printf '%s\n' "$VOL_OUTPUT" | awk -F': ' '/mean_volume:/ {sub(/ dB$/,"",$2); print $2; exit}')
MAX_VOLUME=$(printf '%s\n' "$VOL_OUTPUT" | awk -F': ' '/max_volume:/ {sub(/ dB$/,"",$2); print $2; exit}')

VOL_DETAILS="mean=${MEAN_VOLUME:-?}dB max=${MAX_VOLUME:-?}dB"
BASE_DETAILS="codec=${CODEC:-?} ch=${CHANNELS:-?} sr=${SAMPLE_RATE:-?} dur=${DURATION:-?} pkts=${PACKET_COUNT} ${VOL_DETAILS}"

if [ -n "$MEAN_VOLUME" ]; then
    IS_LOW=$(awk -v m="$MEAN_VOLUME" -v t="$LOW_VOLUME_THRESHOLD_DB" \
        'BEGIN {print (m+0 < t+0) ? 1 : 0}')
    if [ "$IS_LOW" = "1" ]; then
        echo "LOW_VOLUME	${CHANNEL}	${VIDEO_ID}	${BASE_DETAILS} threshold=${LOW_VOLUME_THRESHOLD_DB}dB	${FILE}"
        exit 1
    fi
fi

echo "HAS_AUDIO	${CHANNEL}	${VIDEO_ID}	${BASE_DETAILS}	${FILE}"
exit 0
