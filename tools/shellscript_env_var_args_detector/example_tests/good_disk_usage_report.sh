#!/usr/bin/env bash
set -euo pipefail

dir="."
if [[ $# -ge 1 ]]; then
  dir=$1
fi

if [[ ! -d "$dir" ]]; then
  printf 'error: not a directory: %s\n' "$dir" >&2
  exit 2
fi

du -sh -- "$dir"
