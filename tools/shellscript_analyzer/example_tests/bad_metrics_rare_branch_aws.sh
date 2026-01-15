#!/bin/sh
# Complex metrics, env var only in rare branch (AWS_REGION)
set -eu

usage() {
  echo "Usage: $0 --input FILE --threshold N [--region-check]"
  exit 2
}

INPUT=""
THRESHOLD=""
REGION_CHECK=""

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --input)
        INPUT="$2"
        shift 2
        ;;
      --threshold)
        THRESHOLD="$2"
        shift 2
        ;;
      --region-check)
        REGION_CHECK="1"
        shift 1
        ;;
      *)
        usage
        ;;
    esac
  done
}

validate() {
  if [ -z "$INPUT" ] || [ -z "$THRESHOLD" ]; then
    usage
  fi

  if [ ! -f "$INPUT" ]; then
    echo "Missing input: $INPUT"
    exit 3
  fi

  case "$THRESHOLD" in
    *[!0-9]*|"")
      echo "threshold must be numeric"
      exit 3
      ;;
  esac
}

compute_metrics() {
  lines="$(wc -l "$INPUT" | awk '{print $1}')"
  words="$(wc -w "$INPUT" | awk '{print $1}')"
  chars="$(wc -c "$INPUT" | awk '{print $1}')"

  echo "lines=$lines"
  echo "words=$words"
  echo "chars=$chars"

  if [ "$lines" -gt "$THRESHOLD" ]; then
    echo "threshold_exceeded=1"
  else
    echo "threshold_exceeded=0"
  fi
}

region_check() {
  echo "region=$AWS_REGION"
}

main() {
  parse_args "$@"
  validate
  compute_metrics

  if [ "$REGION_CHECK" = "1" ]; then
    region_check
  fi
}

main "$@"
