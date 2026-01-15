#!/usr/bin/env bash
set -euo pipefail

dir="/tmp"
if [[ $# -ge 1 ]]; then dir=$1; fi

if [[ ! -d "$dir" ]]; then
  printf 'error: %s not a directory\n' "$dir" >&2
  exit 2
fi

find "$dir" -type f -name '*.tmp' -delete
printf 'Temporary files removed from %s\n' "$dir"
