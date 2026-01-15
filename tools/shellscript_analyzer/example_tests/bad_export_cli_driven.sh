#!/bin/sh
# Export in a larger, otherwise CLI-driven script
set -eu

INPUT=""
usage() {
  echo "Usage: $0 --input FILE --mode MODE"
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

main() {
  parse_args "$@"
  if [ -z "$INPUT" ] || [ -z "${MODE:-}" ]; then
    usage
  fi

  export MODE

  awk 'NF > 0 { print }' "$INPUT" | head -n 20
  echo "mode=$MODE"
}

main "$@"
