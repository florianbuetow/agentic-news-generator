#!/bin/sh
# Sourcing configuration inside a complex flow
set -eu

CFG=""
INPUT=""

usage() {
  echo "Usage: $0 --cfg FILE --input FILE"
  exit 2
}

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
      *)
        usage
        ;;
    esac
  done
}

load_cfg() {
  . "$CFG"
}

main() {
  parse_args "$@"
  if [ -z "$CFG" ] || [ -z "$INPUT" ]; then
    usage
  fi
  if [ ! -f "$CFG" ] || [ ! -f "$INPUT" ]; then
    echo "missing file"
    exit 3
  fi

  load_cfg

  # Do some real work
  grep -n "ERROR" "$INPUT" | head -n 10
}

main "$@"
