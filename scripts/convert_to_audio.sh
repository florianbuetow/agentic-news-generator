#!/bin/bash

# Source central configuration
source "$(dirname "$0")/config.sh"

# Check if required dependencies are installed
if ! command -v ffmpeg &> /dev/null; then
    echo "🚨 ERROR: ffmpeg is not installed"
    exit 1
fi

if ! command -v ffprobe &> /dev/null; then
    echo "🚨 ERROR: ffprobe is not installed (should come with ffmpeg)"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "🚨 ERROR: jq is not installed (required for silence detection)"
    echo "Install with: brew install jq"
    exit 1
fi

# Detect number of CPU cores for optimal threading
NUM_THREADS=$(detect_cpu_cores)

# ============================================================================
# HELPER FUNCTIONS FOR SILENCE DETECTION
# ============================================================================

# Function to parse FFmpeg silence detection log
# Extracts silence intervals from stderr output
# Arguments: $1 = path to silence log file
# Returns: JSON array of silence intervals
parse_silence_intervals() {
    local log_file="$1"
    local intervals="[]"
    local start=""

    while IFS= read -r line; do
        if [[ "$line" =~ silence_start:\ ([0-9.]+) ]]; then
            start="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ silence_end:\ ([0-9.]+)\ \|\ silence_duration:\ ([0-9.]+) ]]; then
            local end="${BASH_REMATCH[1]}"
            local duration="${BASH_REMATCH[2]}"

            if [[ -n "$start" ]]; then
                intervals=$(echo "$intervals" | jq \
                    --arg s "$start" \
                    --arg e "$end" \
                    --arg d "$duration" \
                    '. += [{
                        "start_seconds": ($s | tonumber),
                        "end_seconds": ($e | tonumber),
                        "duration_seconds": ($d | tonumber)
                    }]')
                start=""
            fi
        fi
    done < "$log_file"

    echo "$intervals"
}

# Function to compute speech intervals (inverse of silence intervals)
# Speech intervals are the non-silent parts we want to keep
# Arguments: $1 = silence intervals JSON, $2 = total duration in seconds
# Returns: JSON array of speech intervals with start/end times
compute_speech_intervals() {
    local silence_intervals="$1"
    local total_duration="$2"

    # Use jq to compute speech intervals from silence intervals
    echo "$silence_intervals" | jq \
        --arg total "$total_duration" \
        'def compute_speech($total_dur):
            . as $silences |
            reduce range(0; ($silences | length) + 1) as $i (
                {intervals: [], cursor: 0.0};
                if $i < ($silences | length) then
                    ($silences[$i].start_seconds) as $silence_start |
                    ($silences[$i].end_seconds) as $silence_end |
                    if $silence_start > .cursor then
                        .intervals += [{
                            start_seconds: .cursor,
                            end_seconds: $silence_start,
                            duration_seconds: ($silence_start - .cursor)
                        }] |
                        .cursor = $silence_end
                    else
                        .cursor = $silence_end
                    end
                else
                    if .cursor < ($total_dur | tonumber) then
                        .intervals += [{
                            start_seconds: .cursor,
                            end_seconds: ($total_dur | tonumber),
                            duration_seconds: (($total_dur | tonumber) - .cursor)
                        }]
                    else . end
                end
            ) | .intervals;
        compute_speech($total)'
}

# Function to build FFmpeg select filter expression from speech intervals
# Creates expression like: between(t,0,10.5)+between(t,15.2,20.1)+...
# Arguments: $1 = speech intervals JSON
# Returns: Select filter expression string
build_select_filter() {
    local speech_intervals="$1"

    # Build select expression using jq
    echo "$speech_intervals" | jq -r '
        map("between(t,\(.start_seconds),\(.end_seconds))") | join("+")
    '
}

# Function to compute kept audio segments from speech intervals
# Calculates bidirectional timestamp mapping between original and trimmed audio
# Arguments: $1 = speech intervals JSON
# Returns: JSON array of kept segments with timestamp mappings
compute_kept_segments_from_speech() {
    local speech_intervals="$1"

    # Use jq to compute kept segments with bidirectional mappings
    echo "$speech_intervals" | jq '
        reduce .[] as $interval (
            {segments: [], trimmed_cursor: 0.0};
            .segments += [{
                trimmed_start: .trimmed_cursor,
                trimmed_end: (.trimmed_cursor + $interval.duration_seconds),
                original_start: $interval.start_seconds,
                original_end: $interval.end_seconds
            }] |
            .trimmed_cursor += $interval.duration_seconds
        ) | .segments
    '
}

# ============================================================================

echo "Converting videos to audio files"
echo "=========================================="
echo ""

# Iterate over all channel folders in videos directory
find "$VIDEOS_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r channel_dir; do
    channel_name=$(basename "$channel_dir")

    echo "Processing channel: $channel_name"
    echo "---"

    # Create output directory for this channel
    mkdir -p "$AUDIO_DIR/$channel_name"

    # Counter for skipped files in this channel
    skipped_count=0

    # Process all video files matching the whitelist
    while IFS= read -r -d '' input_file; do

        # Get the base filename and extension
        filename=$(basename "$input_file")
        extension="${filename##*.}"
        base_name="${filename%.*}"

        # Skip macOS metadata files
        if [[ "$filename" == ._* ]]; then
            continue
        fi

        # Validate file extension against whitelist
        extension_lower=$(echo "$extension" | tr '[:upper:]' '[:lower:]')
        is_valid=false
        for allowed_ext in "${ALLOWED_EXTENSIONS[@]}"; do
            if [[ "$extension_lower" == "$allowed_ext" ]]; then
                is_valid=true
                break
            fi
        done

        if [[ "$is_valid" == false ]]; then
            continue
        fi

        # Skip files modified within the last 60 seconds (likely still downloading)
        current_time=$(date +%s)
        file_modified_time=$(stat -f %m "$input_file" 2>/dev/null || stat -c %Y "$input_file" 2>/dev/null)
        time_since_modification=$((current_time - file_modified_time))

        if [ $time_since_modification -lt 60 ]; then
            if [ "$VERBOSE" = "true" ]; then
                echo "  ⏸️  Skipping: $filename (modified ${time_since_modification}s ago, waiting for download to complete)"
            fi
            continue
        fi

        # Define output file paths
        output_wav="$AUDIO_DIR/$channel_name/$base_name.wav"
        output_json="$METADATA_DIR/$channel_name/$METADATA_AUDIO_SUBDIR/$base_name.silence_map.json"
        temp_wav="$AUDIO_DIR/$channel_name/.$base_name.temp.wav"
        silence_log="$AUDIO_DIR/$channel_name/.$base_name.silence.log"

        # Create metadata audio subdirectory if it doesn't exist
        mkdir -p "$METADATA_DIR/$channel_name/$METADATA_AUDIO_SUBDIR"

        # Check if both output files already exist (idempotent)
        if [ -f "$output_wav" ] && { [ "$ENABLE_SILENCE_REMOVAL" = "false" ] || [ -f "$output_json" ]; }; then
            skipped_count=$((skipped_count + 1))
            if [ "$VERBOSE" = "true" ]; then
                echo "  ⏭️  Skipping: $filename"
                echo "  ---"
            fi
        else
            echo "  Processing: $filename"

            if [ "$ENABLE_SILENCE_REMOVAL" = "true" ]; then
                # === TWO-PASS PROCESSING WITH SILENCE REMOVAL ===

                # === PASS 1: Convert + Detect Silence ===
                echo "    [Pass 1/2] Converting and detecting silence..."

                ffmpeg -threads "$NUM_THREADS" -y -i "$input_file" \
                    -vn -ar 16000 -ac 1 -c:a pcm_s16le \
                    -af "silencedetect=noise=${SILENCE_THRESHOLD_DB}dB:d=${SILENCE_MIN_DURATION}" \
                    -f wav "$temp_wav" \
                    -loglevel info -stats \
                    2> "$silence_log" </dev/null
                ffmpeg_exit=$?

                if [ $ffmpeg_exit -ne 0 ]; then
                    echo "    🚨 FAILED to convert $filename (pass 1)"
                    rm -f "$temp_wav" "$silence_log"
                    exit 1
                fi

                # Get original audio duration
                original_duration=$(ffprobe -v error -show_entries format=duration \
                    -of default=noprint_wrappers=1:nokey=1 "$temp_wav")

                # Parse silence intervals
                silence_intervals=$(parse_silence_intervals "$silence_log")
                num_silence=$(echo "$silence_intervals" | jq 'length')

                echo "    Detected $num_silence silence interval(s)"

                # Compute speech intervals (inverse of silence)
                speech_intervals=$(compute_speech_intervals "$silence_intervals" "$original_duration")
                num_speech=$(echo "$speech_intervals" | jq 'length')

                echo "    Identified $num_speech speech segment(s) to keep"

                # === PASS 2: Extract Speech Segments ===
                echo "    [Pass 2/2] Extracting speech segments..."

                if [ $num_speech -eq 0 ]; then
                    # No speech detected - create empty audio file
                    echo "    ⚠️  Warning: No speech detected (entire audio is silence)"
                    ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 0.1 -ar 16000 -ac 1 -c:a pcm_s16le "$output_wav" -loglevel error </dev/null
                else
                    # Build select filter expression
                    select_expr=$(build_select_filter "$speech_intervals")

                    # Use aselect + asetpts to extract and concatenate speech segments
                    ffmpeg -threads "$NUM_THREADS" -y -i "$temp_wav" \
                        -af "aselect='${select_expr}',asetpts=N/SR/TB" \
                        -ar 16000 -ac 1 -c:a pcm_s16le \
                        -loglevel error -stats \
                        "$output_wav" </dev/null
                    ffmpeg_exit=$?

                    if [ $ffmpeg_exit -ne 0 ]; then
                        echo "    🚨 FAILED to extract speech segments from $filename (pass 2)"
                        rm -f "$temp_wav" "$silence_log" "$output_wav"
                        exit 1
                    fi
                fi

                # Get trimmed audio duration
                trimmed_duration=$(ffprobe -v error -show_entries format=duration \
                    -of default=noprint_wrappers=1:nokey=1 "$output_wav")

                # Calculate total silence removed
                total_silence=$(echo "$silence_intervals" | jq '[.[].duration_seconds] | add // 0')

                # Compute kept segments from speech intervals
                kept_segments=$(compute_kept_segments_from_speech "$speech_intervals")

                # Generate JSON mapping file
                echo "    Creating silence map..."
                jq -n \
                    --arg version "1.0" \
                    --arg source "$filename" \
                    --arg orig_dur "$original_duration" \
                    --arg trim_dur "$trimmed_duration" \
                    --arg threshold "$SILENCE_THRESHOLD_DB" \
                    --arg min_dur "$SILENCE_MIN_DURATION" \
                    --argjson intervals "$silence_intervals" \
                    --argjson segments "$kept_segments" \
                    --arg total_silence "$total_silence" \
                    '{
                        version: $version,
                        source_video: $source,
                        audio_duration_original_seconds: ($orig_dur | tonumber),
                        audio_duration_trimmed_seconds: ($trim_dur | tonumber),
                        silence_threshold_db: ($threshold | tonumber),
                        silence_min_duration_seconds: ($min_dur | tonumber),
                        silence_intervals: $intervals,
                        kept_segments: $segments,
                        total_silence_removed_seconds: ($total_silence | tonumber)
                    }' > "$output_json"

                # Validate JSON
                if ! jq empty "$output_json" 2>/dev/null; then
                    echo "    🚨 Generated invalid JSON"
                    rm -f "$output_json" "$temp_wav" "$silence_log" "$output_wav"
                    exit 1
                fi

                # Clean up temp files
                rm -f "$temp_wav" "$silence_log"

                echo "    ✅ Done: $base_name.wav (removed ${total_silence}s of silence)"
            else
                # === SIMPLE CONVERSION (Silence removal disabled) ===
                echo "    Converting to audio/$channel_name/$base_name.wav (using $NUM_THREADS threads)..."

                ffmpeg -threads "$NUM_THREADS" -y -i "$input_file" \
                    -vn -ar 16000 -ac 1 -c:a pcm_s16le \
                    -loglevel error -stats \
                    "$output_wav" </dev/null
                ffmpeg_exit=$?

                if [ $ffmpeg_exit -eq 0 ]; then
                    echo "    ✅ Done: $base_name.wav (silence removal disabled)"
                else
                    echo "    🚨 FAILED to convert $filename"
                    exit 1
                fi
            fi
            echo "  ---"
        fi

    done < <(find "$channel_dir" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.wav" -o -name "*.webm" -o -name "*.m4a" -o -name "*.mov" -o -name "*.m4v" -o -name "*.mp3" -o -name "*.ogg" \) -print0)

    # Print skip summary if any files were skipped
    if [ $skipped_count -gt 0 ]; then
        echo "⏭️  Skipped $skipped_count file(s) (WAV already exists)"
    fi

    echo ""
done

echo "Audio conversion complete."
