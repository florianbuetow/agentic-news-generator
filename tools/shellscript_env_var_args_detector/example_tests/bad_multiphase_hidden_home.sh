#!/bin/sh
# Multi-phase workflow with hidden HOME dependency
set -eu

usage() {
  echo "Usage: $0 --input FILE --pattern REGEX"
  exit 2
}

INPUT=""
PATTERN=""

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --input)
        INPUT="$2"
        shift 2
        ;;
      --pattern)
        PATTERN="$2"
        shift 2
        ;;
      -h|--help)
        usage
        ;;
      *)
        echo "Unknown arg: $1"
        usage
        ;;
    esac
  done
}

validate() {
  if [ -z "$INPUT" ] || [ -z "$PATTERN" ]; then
    usage
  fi
  if [ ! -f "$INPUT" ]; then
    echo "Input not found: $INPUT"
    exit 3
  fi
}

scan_lines() {
  i=0
  while IFS= read -r line; do
    i=$((i + 1))
    if echo "$line" | grep -Eq "$PATTERN"; then
      echo "MATCH:$i:$line"
    fi
  done < "$INPUT"
}

report_context() {
  echo "Input: $INPUT"
  echo "Pattern: $PATTERN"
  echo "Home: $HOME"
}

main() {
  parse_args "$@"
  validate
  report_context
  scan_lines
}

main "$@"
