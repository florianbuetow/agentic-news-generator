#!/bin/sh
# Arithmetic and parameter expansion with env var
parse_args() { N="$1"; }
parse_args "$@"
n=$((N + 1))
echo "${n}"
echo "jobs=${JOBS:-4}"
