#!/bin/sh
# Mixed local variables but unbound AWS var deep in function
set -eu

INPUT=""
usage() {
  echo "Usage: $0 --input FILE"
  exit 2
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
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

compute_stats() {
  lines="$(wc -l "$INPUT" | awk '{print $1}')"
  words="$(wc -w "$INPUT" | awk '{print $1}')"
  echo "lines=$lines words=$words"
}

report_env() {
  echo "Running in region: $AWS_REGION"
}

main() {
  parse_args "$@"
  if [ -z "$INPUT" ]; then
    usage
  fi
  compute_stats
  report_env
}

main "$@"
