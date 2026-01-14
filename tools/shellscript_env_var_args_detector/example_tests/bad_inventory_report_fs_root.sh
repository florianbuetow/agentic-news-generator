#!/bin/sh
# Inventory report with hidden FS_ROOT dependency
set -eu

usage() {
  echo "Usage: $0 --input FILE --min N --needle TEXT"
  exit 2
}

INPUT=""
MIN=""
NEEDLE=""

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --input)
        INPUT="$2"
        shift 2
        ;;
      --min)
        MIN="$2"
        shift 2
        ;;
      --needle)
        NEEDLE="$2"
        shift 2
        ;;
      -h|--help)
        usage
        ;;
      *)
        echo "Unknown argument: $1"
        usage
        ;;
    esac
  done
}

validate() {
  if [ -z "$INPUT" ] || [ -z "$MIN" ] || [ -z "$NEEDLE" ]; then
    usage
  fi

  if [ ! -f "$INPUT" ]; then
    echo "Missing file: $INPUT"
    exit 3
  fi

  case "$MIN" in
    *[!0-9]*|"")
      echo "MIN must be numeric"
      exit 3
      ;;
  esac
}

collect() {
  total=0
  matched=0

  while IFS= read -r line; do
    total=$((total + 1))

    if echo "$line" | grep -Fq "$NEEDLE"; then
      matched=$((matched + 1))
      printf '%s\n' "$line"
    fi
  done < "$INPUT"

  echo "total=$total"
  echo "matched=$matched"
}

summarize() {
  count="$(collect | wc -l | awk '{print $1}')"

  if [ "$count" -lt "$MIN" ]; then
    echo "Not enough matches: $count < $MIN"
    exit 4
  fi

  echo "All good. Using filesystem root: $FS_ROOT"
}

main() {
  parse_args "$@"
  validate
  summarize
}

main "$@"
