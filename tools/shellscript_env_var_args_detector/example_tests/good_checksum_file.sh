#!/usr/bin/env bash
set -euo pipefail

file=""
algo="sha256"

if [[ $# -ge 1 ]]; then
  file=$1
fi
if [[ $# -ge 2 ]]; then
  algo=$2
fi

if [[ -z "$file" ]]; then
  printf 'usage: %s <file> [sha256|sha1|md5]\n' "$0" >&2
  exit 2
fi
if [[ ! -f "$file" ]]; then
  printf 'error: file not found: %s\n' "$file" >&2
  exit 2
fi

case "$algo" in
  sha256) sha256sum -- "$file" ;;
  sha1)   sha1sum -- "$file" ;;
  md5)    md5sum -- "$file" ;;
  *)      printf 'error: unsupported algorithm: %s\n' "$algo" >&2; exit 2 ;;
esac
