#!/usr/bin/env bash
set -euo pipefail

dir=""
if [[ $# -ge 1 ]]; then dir=$1; fi
if [[ -z "$dir" || ! -d "$dir" ]]; then
  printf 'usage: %s <dir>\n' "$0" >&2
  exit 2
fi

prev=$(mktemp)
trap 'rm -f "$prev"' EXIT
find "$dir" -type f | sort >"$prev"

while sleep 2; do
  curr=$(mktemp)
  find "$dir" -type f | sort >"$curr"
  diff -u "$prev" "$curr" || true
  mv "$curr" "$prev"
done
