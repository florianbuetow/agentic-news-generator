#!/usr/bin/env bash
set -euo pipefail

sum=0
for x in "$@"; do
  sum=$((sum + x))
done
printf '%s\n' "$sum"
