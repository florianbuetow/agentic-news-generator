#!/usr/bin/env bash
# Bad example: Relies only on undocumented environment variables

set -euo pipefail

# No documentation, no CLI args, just env vars
ffmpeg -i "$INPUT_FILE" \
    -af "silenceremove=start_threshold=${SILENCE_THRESHOLD}:start_duration=${MIN_DURATION}" \
    "$OUTPUT_FILE"

echo "Processing complete: ${OUTPUT_FILE}"
