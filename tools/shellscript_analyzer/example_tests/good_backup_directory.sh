#!/usr/bin/env bash
set -euo pipefail

src=""
dest=""
ts=""

if [[ $# -ge 1 ]]; then src=$1; fi
if [[ $# -ge 2 ]]; then dest=$2; fi

if [[ -z "$src" || -z "$dest" ]]; then
  printf 'usage: %s <source_dir> <backup_dir>\n' "$0" >&2
  exit 2
fi

if [[ ! -d "$src" ]]; then
  printf 'error: source directory not found\n' >&2
  exit 2
fi

mkdir -p -- "$dest"
ts=$(date +%Y%m%d_%H%M%S)
tar -czf "$dest/backup_${ts}.tar.gz" -C "$src" .
printf 'Backup complete: %s/backup_%s.tar.gz\n' "$dest" "$ts"
