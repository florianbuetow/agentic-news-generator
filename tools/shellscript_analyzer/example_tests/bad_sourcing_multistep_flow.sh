#!/bin/sh
# Sourcing inside a multi-step flow
set -eu

usage() {
  echo "Usage: $0 --cfg FILE --input FILE --needle TEXT"
  exit 2
}

CFG=""
INPUT=""
NEEDLE=""

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --cfg)
        CFG="$2"
        shift 2
        ;;
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

validate() {
  if [ -z "$CFG" ] || [ -z "$INPUT" ] || [ -z "$NEEDLE" ]; then
    usage
  fi

  if [ ! -f "$CFG" ] || [ ! -f "$INPUT" ]; then
    echo "Missing file"
    exit 3
  fi
}

load_config() {
  . "$CFG"
}

process() {
  grep -n -F "$NEEDLE" "$INPUT" | head -n 25
}

main() {
  parse_args "$@"
  validate
  load_config
  process
}

main "$@"
