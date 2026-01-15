#!/bin/sh
# Pipeline-heavy reader with LANG dependency
set -eu

INPUT=""
TOP=""

usage() {
  echo "Usage: $0 --input FILE --top N"
  exit 2
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --input)
        INPUT="$2"
        shift 2
        ;;
      --top)
        TOP="$2"
        shift 2
        ;;
      *)
        usage
        ;;
    esac
  done
}

validate() {
  if [ -z "$INPUT" ] || [ -z "$TOP" ]; then
    usage
  fi
  if [ ! -f "$INPUT" ]; then
    echo "input not found: $INPUT"
    exit 3
  fi
}

top_words() {
  cat "$INPUT" \
    | tr -cs '[:alnum:]' '\n' \
    | tr '[:upper:]' '[:lower:]' \
    | sort \
    | uniq -c \
    | sort -nr \
    | head -n "$TOP"

  echo "Locale is: $LANG"
}

main() {
  parse_args "$@"
  validate
  top_words
}

main "$@"
