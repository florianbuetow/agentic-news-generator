#!/bin/sh
# Eval hidden behind structured parsing
set -eu

usage() {
  echo "Usage: $0 --name VAR_NAME --input FILE"
  exit 2
}

NAME=""
INPUT=""

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --name)
        NAME="$2"
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
  if [ -z "$NAME" ] || [ -z "$INPUT" ]; then
    usage
  fi

  if [ ! -f "$INPUT" ]; then
    echo "Missing input: $INPUT"
    exit 3
  fi
}

main() {
  parse_args "$@"
  validate

  preview="$(head -n 3 "$INPUT" | tr -d '\r')"
  echo "preview:"
  printf '%s\n' "$preview"

  expr="echo \$NAME"
  eval "$expr"
}

main "$@"
