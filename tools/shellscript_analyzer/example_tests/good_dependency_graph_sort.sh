#!/usr/bin/env bash
set -euo pipefail

declare -A deps
declare -A done

input=""
if [[ $# -ge 1 ]]; then input=$1; fi
if [[ -z "$input" ]]; then
  printf 'usage: %s <file>\n' "$0" >&2
  exit 2
fi
if [[ ! -f "$input" ]]; then
  printf 'error: no such file %s\n' "$input" >&2
  exit 2
fi

while IFS=: read -r node rest; do
  rest=${rest//,/ }
  deps["$node"]="$rest"
done <"$input"

resolve() {
  local n=$1
  if [[ -n "${done[$n]:-}" ]]; then return; fi
  done["$n"]=1
  for dep in ${deps[$n]:-}; do
    resolve "$dep"
  done
  printf '%s\n' "$n"
}

for n in "${!deps[@]}"; do
  resolve "$n"
done | awk '!x[$0]++'
