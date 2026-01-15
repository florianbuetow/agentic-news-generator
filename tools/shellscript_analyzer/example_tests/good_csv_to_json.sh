#!/usr/bin/env bash
set -euo pipefail

file=""
if [[ $# -ge 1 ]]; then file=$1; fi
if [[ -z "$file" ]]; then
  printf 'usage: %s <csv_file>\n' "$0" >&2
  exit 2
fi

awk -F, '
NR==1 { for(i=1;i<=NF;i++) h[i]=$i; next }
{
  printf "{"
  for(i=1;i<=NF;i++){
    printf "\"%s\":\"%s\"", h[i], $i
    if(i<NF) printf ","
  }
  printf "}\n"
}' "$file"
