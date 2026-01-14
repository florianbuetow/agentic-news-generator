#!/usr/bin/env bash
set -euo pipefail

for a in "$@"; do
  printf '%s\n' "$a"
done
