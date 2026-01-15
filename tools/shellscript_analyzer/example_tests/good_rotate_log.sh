#!/usr/bin/env bash
set -euo pipefail

log=""
keep="5"

if [[ $# -ge 1 ]]; then log=$1; fi
if [[ $# -ge 2 ]]; then keep=$2; fi

if [[ -z "$log" ]]; then
  printf 'usage: %s <logfile> [keep]\n' "$0" >&2
  exit 2
fi

if [[ -f "$log.$keep" ]]; then
  rm -f -- "$log.$keep"
fi

i=$keep
while [[ $i -gt 1 ]]; do
  prev=$((i - 1))
  if [[ -f "$log.$prev" ]]; then
    mv -- "$log.$prev" "$log.$i"
  fi
  i=$prev
done

if [[ -f "$log" ]]; then
  mv -- "$log" "$log.1"
fi

: > "$log"
