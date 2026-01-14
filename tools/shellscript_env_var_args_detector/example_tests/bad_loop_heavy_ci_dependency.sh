#!/bin/sh
# Loop-heavy log analysis with CI dependency
set -eu

LOG=""
NEEDLE=""

usage() {
  echo "Usage: $0 --log FILE --needle TEXT"
  exit 2
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --log)
        LOG="$2"
        shift 2
        ;;
      --needle)
        NEEDLE="$2"
        shift 2
        ;;
      *)
        usage
        ;;
    esac
  done
}

validate() {
  if [ -z "$LOG" ] || [ -z "$NEEDLE" ]; then
    usage
  fi
  if [ ! -f "$LOG" ]; then
    echo "log not found: $LOG"
    exit 3
  fi
}

analyze() {
  total=0
  hits=0

  while IFS= read -r line; do
    total=$((total + 1))
    if echo "$line" | grep -Fq "$NEEDLE"; then
      hits=$((hits + 1))
      if [ $((hits % 5)) -eq 0 ]; then
        echo "Hit milestone: $hits"
      fi
    fi
  done < "$LOG"

  echo "Total lines: $total"
  echo "Hits: $hits"
  echo "CI mode: $CI"
}

main() {
  parse_args "$@"
  validate
  analyze
}

main "$@"
