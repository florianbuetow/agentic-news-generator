#!/bin/sh
# Indirect expansion hidden in helper function
set -eu

KEY=""
usage() {
  echo "Usage: $0 --key NAME"
  exit 2
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --key)
        KEY="$2"
        shift 2
        ;;
      *)
        usage
        ;;
    esac
  done
}

resolve() {
  name="$1"
  echo "${!name}"
}

main() {
  parse_args "$@"
  if [ -z "$KEY" ]; then
    usage
  fi

  echo "Requested key: $KEY"
  echo "Resolved value: $(resolve "$KEY")"
}

main "$@"
