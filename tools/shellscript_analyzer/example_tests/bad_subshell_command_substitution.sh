#!/bin/sh
# Subshell and command substitution with env var
parse_args() { NAME="$1"; }
parse_args "$@"
out="$(echo "hello $NAME" | tr a-z A-Z)"
echo "$out"
echo "tmp=$TMPDIR"
