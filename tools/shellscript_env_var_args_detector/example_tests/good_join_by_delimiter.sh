#!/usr/bin/env bash
set -euo pipefail

delim=","
if [[ $# -ge 1 ]]; then
  delim=$1
  shift
fi

if [[ $# -lt 1 ]]; then
  printf 'usage: %s [delimiter] <item1> [item2 ...]\n' "$0" >&2
  exit 2
fi

out=""
first=1
for item in "$@"; do
  if [[ $first -eq 1 ]]; then
    out=$item
    first=0
  else
    out="${out}${delim}${item}"
  fi
done

printf '%s\n' "$out"
