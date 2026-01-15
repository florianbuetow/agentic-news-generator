#!/bin/sh
# Trap and cleanup with PWD dependency
set -eu

INPUT=""
OUT=""

usage() {
  echo "Usage: $0 --input FILE --out FILE"
  exit 2
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --input)
        INPUT="$2"
        shift 2
        ;;
      --out)
        OUT="$2"
        shift 2
        ;;
      *)
        usage
        ;;
    esac
  done
}

cleanup() {
  echo "Exiting from: $PWD"
}

trap cleanup EXIT

validate() {
  if [ -z "$INPUT" ] || [ -z "$OUT" ]; then
    usage
  fi
  if [ ! -f "$INPUT" ]; then
    echo "input not found: $INPUT"
    exit 3
  fi
}

process() {
  # read-only tooling, but the script itself writes a report to stdout only
  awk 'NF > 0 { c++ } END { print "non_empty_lines=" c }' "$INPUT"
  echo "Would have written to $OUT but not doing so."
}

main() {
  parse_args "$@"
  validate
  process
}

main "$@"
