#!/usr/bin/env bash
set -euo pipefail

dir=""
days=30
archive="archive.tar.gz"

if [[ $# -ge 1 ]]; then dir=$1; fi
if [[ $# -ge 2 ]]; then days=$2; fi
if [[ $# -ge 3 ]]; then archive=$3; fi

if [[ -z "$dir" ]]; then
  printf 'usage: %s <dir> [days] [archive_name]\n' "$0" >&2
  exit 2
fi

find "$dir" -type f -mtime +"$days" -print0 | tar -czf "$archive" --null -T -
printf 'Archived files older than %s days into %s\n' "$days" "$archive"
