#!/bin/sh
# Trap handler reads env var
parse_args() { OUT="$1"; }
cleanup() { echo "cleanup in $PWD" >> "$OUT"; echo "shell=$SHELL" >> "$OUT"; }
trap cleanup EXIT
parse_args "$@"
echo "work" > "$OUT"
