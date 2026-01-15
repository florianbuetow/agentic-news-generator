#!/bin/sh
# Good example: Uses ONLY CLI arguments with hardcoded defaults, NO environment variables
set -eu

# Hardcoded default values (no env vars)
DEFAULT_THRESHOLD="-40"
DEFAULT_DURATION="2.0"

# Accept CLI arguments with hardcoded defaults
THRESHOLD="${1:-$DEFAULT_THRESHOLD}"
DURATION="${2:-$DEFAULT_DURATION}"

# Usage function
usage() {
    echo "Usage: $0 [threshold] [duration]"
    echo ""
    echo "Arguments:"
    echo "  threshold  Silence threshold in dB (default: $DEFAULT_THRESHOLD)"
    echo "  duration   Minimum silence duration in seconds (default: $DEFAULT_DURATION)"
    exit 1
}

# Check for help flag
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
fi

# Process audio with parameters
echo "Processing with threshold: ${THRESHOLD}dB, duration: ${DURATION}s"
