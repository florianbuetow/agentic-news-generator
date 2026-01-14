#!/usr/bin/env bash
set -euo pipefail

cmd=""
retries=3
delay=1

if [[ $# -ge 1 ]]; then cmd=$1; shift; fi
if [[ $# -ge 1 ]]; then retries=$1; shift; fi
if [[ $# -ge 1 ]]; then delay=$1; shift; fi
if [[ -z "$cmd" ]]; then
  printf 'usage: %s <command> [retries] [delay]\n' "$0" >&2
  exit 2
fi

i=0
while true; do
  if eval "$cmd"; then
    exit 0
  fi
  i=$((i + 1))
  if [[ $i -ge $retries ]]; then
    printf 'failed after %s retries\n' "$retries" >&2
    exit 1
  fi
  sleep "$delay"
done
