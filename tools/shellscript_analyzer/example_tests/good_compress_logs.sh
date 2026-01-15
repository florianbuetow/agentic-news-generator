#!/usr/bin/env bash
set -euo pipefail

dir="logs"
days=7

if [[ $# -ge 1 ]]; then dir=$1; fi
if [[ $# -ge 2 ]]; then days=$2; fi

if [[ ! -d "$dir" ]]; then
  printf 'error: no such directory\n' >&2
  exit 2
fi

find "$dir" -type f -mtime +"$days" -name "*.log" -exec gzip -f {} +
printf 'Old logs compressed successfully\n'
