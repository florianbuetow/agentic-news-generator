#!/usr/bin/env bash
set -euo pipefail

file=""
if [[ $# -ge 1 ]]; then
  file=$1
fi

if [[ -z "$file" ]]; then
  printf 'usage: %s <file>\n' "$0" >&2
  exit 2
fi
if [[ ! -f "$file" ]]; then
  printf 'error: file not found: %s\n' "$file" >&2
  exit 2
fi

sort -- "$file" | uniq
