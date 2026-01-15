#!/usr/bin/env bash
set -euo pipefail

host=""
port=""

if [[ $# -ge 1 ]]; then host=$1; fi
if [[ $# -ge 2 ]]; then port=$2; fi

if [[ -z "$host" || -z "$port" ]]; then
  printf 'usage: %s <host> <port>\n' "$0" >&2
  exit 2
fi

if (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
  printf 'open\n'
else
  printf 'closed\n'
fi
