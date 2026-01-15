#!/bin/sh
# Deep helper chain with late PATH read
set -eu

usage() {
  echo "Usage: $0 --input FILE --top N"
  exit 2
}

INPUT=""
TOP=""

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
    echo "Input does not exist: $INPUT"
    exit 3
  fi
}

count_words() {
  cat "$INPUT" \
    | tr -cs '[:alnum:]' '\n' \
    | tr '[:upper:]' '[:lower:]' \
    | sort \
    | uniq -c \
    | sort -nr \
    | head -n "$TOP"
}

diagnostics() {
  echo "Diagnostics:"
  echo "  input=$INPUT"
  echo "  top=$TOP"
  echo "  path=$PATH"
}

main() {
  parse_args "$@"
  validate
  count_words
  diagnostics
}

main "$@"
