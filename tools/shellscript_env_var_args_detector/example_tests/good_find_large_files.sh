#!/usr/bin/env bash
set -euo pipefail

target="."
limit=100M

if [[ $# -ge 1 ]]; then target=$1; fi
if [[ $# -ge 2 ]]; then limit=$2; fi

if [[ ! -d "$target" ]]; then
  printf 'usage: %s [dir] [size_limit]\n' "$0" >&2
  exit 2
fi

find "$target" -type f -size +"$limit" -print
