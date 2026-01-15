#!/usr/bin/env bash
# Good example: Uses CLI arguments with env var fallbacks

set -euo pipefail

# Default values
DEFAULT_THRESHOLD="-40"
DEFAULT_DURATION="2.0"

# Accept CLI arguments or use environment variables with defaults
THRESHOLD="${1:-${SILENCE_THRESHOLD_DB:-$DEFAULT_THRESHOLD}}"
DURATION="${2:-${SILENCE_MIN_DURATION:-$DEFAULT_DURATION}}"

# Usage function
usage() {
    echo "Usage: $0 [threshold] [duration]"
    echo ""
    echo "Arguments:"
    echo "  threshold  Silence threshold in dB (default: $DEFAULT_THRESHOLD)"
    echo "  duration   Minimum silence duration in seconds (default: $DEFAULT_DURATION)"
    echo ""
    echo "Environment variables (optional):"
    echo "  SILENCE_THRESHOLD_DB    Override threshold default"
    echo "  SILENCE_MIN_DURATION    Override duration default"
    exit 1
}

# Check for help flag
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
fi

# Process audio with parameters
echo "Processing with threshold: ${THRESHOLD}dB, duration: ${DURATION}s"
