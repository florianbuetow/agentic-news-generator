#!/bin/sh
# Argument parsing looks correct but uses PATH later
set -eu

INPUT=""
LIMIT=""

usage() {
  echo "Usage: $0 --input FILE --limit N"
  exit 2
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --input)
        INPUT="$2"
        shift 2
        ;;
      --limit)
        LIMIT="$2"
        shift 2
        ;;
      *)
        usage
        ;;
    esac
  done
}

validate() {
  if [ -z "$INPUT" ] || [ -z "$LIMIT" ]; then
    usage
  fi
  case "$LIMIT" in
    *[!0-9]*|"")
      echo "limit must be numeric"
      exit 3
      ;;
  esac
}

summarize() {
  count=0
  while IFS= read -r _line; do
    count=$((count + 1))
    if [ "$count" -ge "$LIMIT" ]; then
      break
    fi
  done < "$INPUT"

  echo "Counted $count lines (limit=$LIMIT)"
  echo "PATH is $PATH"
}

main() {
  parse_args "$@"
  validate
  summarize
}

main "$@"
