#!/bin/sh
# Indirect expansion via helper function
set -eu

usage() {
  echo "Usage: $0 --key KEY --input FILE"
  exit 2
}

KEY=""
INPUT=""

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --key)
        KEY="$2"
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

validate() {
  if [ -z "$KEY" ] || [ -z "$INPUT" ]; then
    usage
  fi

  if [ ! -f "$INPUT" ]; then
    echo "Input not found: $INPUT"
    exit 3
  fi
}

resolve_value() {
  var_name="$1"
  echo "${!var_name}"
}

main() {
  parse_args "$@"
  validate

  echo "key=$KEY"
  echo "value=$(resolve_value "$KEY")"
  head -n 3 "$INPUT"
}

main "$@"
