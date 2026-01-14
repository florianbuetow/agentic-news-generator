#!/usr/bin/env bash
set -euo pipefail

urls=()
max=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j) shift; max=$1 ;;
    *) urls+=("$1") ;;
  esac
  shift || true
done

if [[ ${#urls[@]} -eq 0 ]]; then
  printf 'usage: %s [-j N] <url1> <url2> ...\n' "$0" >&2
  exit 2
fi

tmpfifo=$(mktemp -u)
mkfifo "$tmpfifo"
exec 3<>"$tmpfifo"
rm -f "$tmpfifo"

for ((i=0; i<max; i++)); do printf '.' >&3; done

for u in "${urls[@]}"; do
  read -r -u 3 _
  {
    curl -fsSLO "$u" || printf 'failed: %s\n' "$u" >&2
    printf '.' >&3
  } &
done

wait
exec 3>&-
