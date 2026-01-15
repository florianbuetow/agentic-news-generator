#!/bin/sh
# Here-doc report with HOME dependency
set -eu

usage() {
  echo "Usage: $0 --name NAME --input FILE"
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

emit_report() {
  lines="$(wc -l "$INPUT" | awk '{print $1}')"
  words="$(wc -w "$INPUT" | awk '{print $1}')"

  cat <<EOF
name=$NAME
lines=$lines
words=$words
home=$HOME
EOF
}

main() {
  parse_args "$@"
  validate
  emit_report
}

main "$@"
