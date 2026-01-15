#!/bin/sh
# Branch-only env read: TMPDIR
set -eu

usage() {
  echo "Usage: $0 --input FILE [--debug]"
  exit 2
}

INPUT=""
DEBUG=""

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --input)
        INPUT="$2"
        shift 2
        ;;
      --debug)
        DEBUG="1"
        shift 1
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
    echo "Missing input: $INPUT"
    exit 3
  fi
}

debug_info() {
  echo "debug=on"
  echo "tmpdir=$TMPDIR"
}

main() {
  parse_args "$@"
  validate

  if [ "$DEBUG" = "1" ]; then
    debug_info
  fi

  head -n 10 "$INPUT" | nl -ba
}

main "$@"
