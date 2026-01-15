#!/usr/bin/env bash
set -euo pipefail

logdir=""
max=5

if [[ $# -ge 1 ]]; then logdir=$1; fi
if [[ $# -ge 2 ]]; then max=$2; fi
if [[ -z "$logdir" || ! -d "$logdir" ]]; then
  printf 'usage: %s <logdir> [max]\n' "$0" >&2
  exit 2
fi

for file in "$logdir"/*.log; do
  [[ -f "$file" ]] || continue
  for ((i=max; i>=1; i--)); do
    next=$((i + 1))
    [[ -f "$file.$i" ]] && mv "$file.$i" "$file.$next"
  done
  mv "$file" "$file.1"
  : >"$file"
done
