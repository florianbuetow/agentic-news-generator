#!/usr/bin/env bash
set -euo pipefail

a=""
b=""
if [[ $# -ge 1 ]]; then a=$1; fi
if [[ $# -ge 2 ]]; then b=$2; fi

if [[ -z "$a" || -z "$b" ]]; then
  printf 'usage: %s <dir1> <dir2>\n' "$0" >&2
  exit 2
fi

diff -rq "$a" "$b" || true
