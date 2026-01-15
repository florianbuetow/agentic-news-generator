#!/usr/bin/env bash
set -euo pipefail

target=""
pattern=""
replacement=""

if [[ $# -ge 3 ]]; then
  target=$1; pattern=$2; replacement=$3
else
  printf 'usage: %s <file> <pattern> <replacement>\n' "$0" >&2
  exit 2
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

sed "s/${pattern}/${replacement}/g" -- "$target" >"$tmp"
mv -- "$tmp" "$target"
