#!/bin/sh
# Export in a long script with many locals
set -eu

usage() {
  echo "Usage: $0 --input FILE --mode MODE --needle TEXT"
  exit 2
}

INPUT=""
MODE=""
NEEDLE=""

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --input)
        INPUT="$2"
        shift 2
        ;;
      --mode)
        MODE="$2"
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
  if [ -z "$INPUT" ] || [ -z "$MODE" ] || [ -z "$NEEDLE" ]; then
    usage
  fi

  if [ ! -f "$INPUT" ]; then
    echo "Missing input: $INPUT"
    exit 3
  fi
}

scan() {
  n=0
  while IFS= read -r line; do
    n=$((n + 1))
    if echo "$line" | grep -Fq "$NEEDLE"; then
      echo "match:$n"
    fi
    if [ "$n" -ge 200 ]; then
      break
    fi
  done < "$INPUT"
}

mode_banner() {
  export MODE
  echo "mode=$MODE"
}

main() {
  parse_args "$@"
  validate
  mode_banner
  scan
}

main "$@"
