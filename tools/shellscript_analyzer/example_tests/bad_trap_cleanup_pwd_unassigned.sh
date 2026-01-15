#!/bin/sh
# Trap-based cleanup reads PWD (unassigned)
set -eu

usage() {
  echo "Usage: $0 --input FILE --needle TEXT"
  exit 2
}

INPUT=""
NEEDLE=""

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --input)
        INPUT="$2"
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

cleanup() {
  echo "cleanup dir=$PWD"
}

trap cleanup EXIT

validate() {
  if [ -z "$INPUT" ] || [ -z "$NEEDLE" ]; then
    usage
  fi

  if [ ! -f "$INPUT" ]; then
    echo "Missing file: $INPUT"
    exit 3
  fi
}

main() {
  parse_args "$@"
  validate

  grep -n -F "$NEEDLE" "$INPUT" | head -n 20
}

main "$@"
