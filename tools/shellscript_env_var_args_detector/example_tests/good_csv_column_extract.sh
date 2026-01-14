#!/usr/bin/env bash
set -euo pipefail

col=""
if [[ $# -ge 1 ]]; then col=$1; fi
if [[ -z "$col" ]]; then
  printf 'usage: %s <column_number>\n' "$0" >&2
  exit 2
fi
awk -F, -v c="$col" '{print $c}'
