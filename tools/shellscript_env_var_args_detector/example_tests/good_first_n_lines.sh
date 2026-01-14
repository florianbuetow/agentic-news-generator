#!/usr/bin/env bash
set -euo pipefail

n="10"
file=""

if [[ $# -ge 1 ]]; then n=$1; fi
if [[ $# -ge 2 ]]; then file=$2; fi

if [[ -z "$file" ]]; then
  printf 'usage: %s [n] <file>\n' "$0" >&2
  exit 2
fi

head -n "$n" -- "$file"
