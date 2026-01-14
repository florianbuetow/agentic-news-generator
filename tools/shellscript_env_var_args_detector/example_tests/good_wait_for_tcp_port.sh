#!/usr/bin/env bash
set -euo pipefail

host="127.0.0.1"
port=""
timeout_s="10"

if [[ $# -ge 1 ]]; then host=$1; fi
if [[ $# -ge 2 ]]; then port=$2; fi
if [[ $# -ge 3 ]]; then timeout_s=$3; fi

if [[ -z "$port" ]]; then
  printf 'usage: %s [host] <port> [timeout_seconds]\n' "$0" >&2
  exit 2
fi

start=$(date +%s)
while true; do
  if (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
    printf 'ready: %s:%s\n' "$host" "$port"
    exit 0
  fi
  now=$(date +%s)
  elapsed=$((now - start))
  if [[ $elapsed -ge $timeout_s ]]; then
    printf 'timeout waiting for %s:%s\n' "$host" "$port" >&2
    exit 1
  fi
  sleep 1
done
