#!/bin/sh
# Conditional assignment then later use (unset variable)
set -eu

usage() {
  echo "Usage: $0 --input FILE [--mode prod|dev]"
  exit 2
}

INPUT=""
MODE=""

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
      *)
        usage
        ;;
    esac
  done
}

validate() {
  if [ -z "$INPUT" ]; then
    usage
  fi

  if [ ! -f "$INPUT" ]; then
    echo "Missing file: $INPUT"
    exit 3
  fi
}

derive_level() {
  if [ "$MODE" = "prod" ]; then
    LEVEL="high"
  fi

  echo "level=$LEVEL"
}

main() {
  parse_args "$@"
  validate
  derive_level
  wc -l "$INPUT" | awk '{print "lines=" $1}'
}

main "$@"
