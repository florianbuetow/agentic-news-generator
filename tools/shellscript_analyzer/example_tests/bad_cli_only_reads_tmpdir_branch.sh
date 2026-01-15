#!/bin/sh
# Apparently CLI-only, but reads TMPDIR in a branch
set -eu

INPUT=""
FLAG=""

usage() {
  echo "Usage: $0 --input FILE [--debug]"
  exit 2
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --input)
        INPUT="$2"
        shift 2
        ;;
      --debug)
        FLAG="debug"
        shift 1
        ;;
      *)
        usage
        ;;
    esac
  done
}

debug_dump() {
  echo "Debug enabled"
  echo "Temp directory: $TMPDIR"
}

main() {
  parse_args "$@"
  if [ -z "$INPUT" ]; then
    usage
  fi

  if [ "$FLAG" = "debug" ]; then
    debug_dump
  fi

  head -n 5 "$INPUT" | nl -ba
}

main "$@"
