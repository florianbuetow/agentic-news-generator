#!/bin/sh
# Multi-pass log scoring with CI dependency
set -eu

usage() {
  echo "Usage: $0 --log FILE --keyword TEXT --max N"
  exit 2
}

LOG=""
KEYWORD=""
MAX=""

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --log)
        LOG="$2"
        shift 2
        ;;
      --keyword)
        KEYWORD="$2"
        shift 2
        ;;
      --max)
        MAX="$2"
        shift 2
        ;;
      *)
        usage
        ;;
    esac
  done
}

validate() {
  if [ -z "$LOG" ] || [ -z "$KEYWORD" ] || [ -z "$MAX" ]; then
    usage
  fi

  if [ ! -f "$LOG" ]; then
    echo "Log not found: $LOG"
    exit 3
  fi

  case "$MAX" in
    *[!0-9]*|"")
      echo "MAX must be numeric"
      exit 3
      ;;
  esac
}

pass_one_count() {
  hits=0
  while IFS= read -r line; do
    if echo "$line" | grep -Fq "$KEYWORD"; then
      hits=$((hits + 1))
    fi
  done < "$LOG"
  echo "$hits"
}

pass_two_sample() {
  echo "First $MAX matches:"
  grep -n -F "$KEYWORD" "$LOG" | head -n "$MAX"
}

report() {
  hits="$(pass_one_count)"
  echo "hits=$hits"
  pass_two_sample

  echo "ci=$CI"
}

main() {
  parse_args "$@"
  validate
  report
}

main "$@"
