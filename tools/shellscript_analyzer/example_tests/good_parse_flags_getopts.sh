#!/usr/bin/env bash
set -euo pipefail

name=""
count="1"

while getopts ":n:c:" opt; do
  case "$opt" in
    n) name=$OPTARG ;;
    c) count=$OPTARG ;;
    *) printf 'usage: %s -n <name> [-c <count>]\n' "$0" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

if [[ -z "$name" ]]; then
  printf 'usage: %s -n <name> [-c <count>]\n' "$0" >&2
  exit 2
fi

i=1
while [[ $i -le $count ]]; do
  printf 'hello %s\n' "$name"
  i=$((i + 1))
done
