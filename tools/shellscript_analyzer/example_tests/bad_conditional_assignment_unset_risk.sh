#!/bin/sh
# Conditional assignment then later use (unset risk)
set -eu

MODE=""
INPUT=""

usage() {
  echo "Usage: $0 --input FILE [--mode prod|dev]"
  exit 2
}

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

derive_settings() {
  if [ "$MODE" = "prod" ]; then
    LEVEL="high"
  fi

  # LEVEL is used even if MODE != prod
  echo "level=$LEVEL"
}

main() {
  parse_args "$@"
  if [ -z "$INPUT" ]; then
    usage
  fi
  derive_settings
  wc -l "$INPUT" | awk '{print "lines=" $1}'
}

main "$@"
