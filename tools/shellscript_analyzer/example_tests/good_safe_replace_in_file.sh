#!/usr/bin/env bash
set -euo pipefail

file=""
search=""
replace=""

if [[ $# -ge 1 ]]; then file=$1; fi
if [[ $# -ge 2 ]]; then search=$2; fi
if [[ $# -ge 3 ]]; then replace=$3; fi

if [[ -z "$file" || -z "$search" ]]; then
  printf 'usage: %s <file> <search> [replace]\n' "$0" >&2
  exit 2
fi
if [[ ! -f "$file" ]]; then
  printf 'error: file not found: %s\n' "$file" >&2
  exit 2
fi

tmp="${file}.tmp.$"
sed "s/${search}/${replace}/g" -- "$file" > "$tmp"
mv -- "$tmp" "$file"
