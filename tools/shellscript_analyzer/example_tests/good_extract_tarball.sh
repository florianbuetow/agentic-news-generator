#!/usr/bin/env bash
set -euo pipefail

archive=""
dest="."

if [[ $# -ge 1 ]]; then
  archive=$1
fi
if [[ $# -ge 2 ]]; then
  dest=$2
fi

if [[ -z "$archive" ]]; then
  printf 'usage: %s <archive.tar[.gz|.bz2|.xz]> [dest_dir]\n' "$0" >&2
  exit 2
fi
if [[ ! -f "$archive" ]]; then
  printf 'error: archive not found: %s\n' "$archive" >&2
  exit 2
fi

mkdir -p -- "$dest"
tar -xf -- "$archive" -C "$dest"
