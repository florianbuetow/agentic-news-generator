#!/usr/bin/env bash
set -euo pipefail

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

step1() { printf 'step1\n' >"$tmp/1"; }
step2() { printf 'step2\n' >"$tmp/2"; }
step3() { cat "$tmp/1" "$tmp/2" >"$tmp/result"; }

( step1 && step2 && step3 ) || {
  printf 'transaction failed\n' >&2
  exit 1
}

cat "$tmp/result"
