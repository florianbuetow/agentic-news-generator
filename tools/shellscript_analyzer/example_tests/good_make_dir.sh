#!/usr/bin/env bash
set -euo pipefail

d=""
if [[ $# -ge 1 ]]; then d=$1; fi
if [[ -z "$d" ]]; then
  printf 'usage: %s <dir>\n' "$0" >&2
  exit 2
fi
mkdir -p -- "$d"
