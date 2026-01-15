#!/bin/sh
# Eval hidden behind seemingly safe parsing
set -eu

NAME=""
usage() {
  echo "Usage: $0 --name VAR_NAME"
  exit 2
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --name)
        NAME="$2"
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
  if [ -z "$NAME" ]; then
    usage
  fi

  expr="echo \$NAME"
  eval "$expr"
}

main "$@"
