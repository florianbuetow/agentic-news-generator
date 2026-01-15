#!/usr/bin/env bash
set -euo pipefail

dir="."
if [[ $# -ge 1 ]]; then dir=$1; fi

if [[ ! -d "$dir" ]]; then
  printf 'usage: %s [dir]\n' "$0" >&2
  exit 2
fi

find "$dir" -type f | awk -F. '/\./{print $NF}' | sort | uniq -c | sort -nr
