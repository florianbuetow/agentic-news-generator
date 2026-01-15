#!/usr/bin/env bash
set -euo pipefail

prefix="tmp"
if [[ $# -ge 1 ]]; then prefix=$1; fi
printf '%s.%s\n' "$prefix" "$"
