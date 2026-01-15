#!/usr/bin/env bash
set -euo pipefail

dir="."
workers=4

if [[ $# -ge 1 ]]; then dir=$1; fi
if [[ $# -ge 2 ]]; then workers=$2; fi

if [[ ! -d "$dir" ]]; then
  printf 'usage: %s [dir] [workers]\n' "$0" >&2
  exit 2
fi

tmp_fifo=$(mktemp -u)
mkfifo "$tmp_fifo"
exec 3<>"$tmp_fifo"
rm -f "$tmp_fifo"

for ((i=0; i<workers; i++)); do
  printf '.' >&3
done

find "$dir" -type f -print0 |
while IFS= read -r -d '' file; do
  read -r -u 3 _
  {
    sha256sum -- "$file"
    printf '.' >&3
  } &
done

wait
exec 3>&-
