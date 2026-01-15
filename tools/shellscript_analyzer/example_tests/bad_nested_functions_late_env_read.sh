#!/bin/sh
# Nested functions with late env read
parse_args() { MODE="$1"; }
inner() { echo "mode=$MODE"; echo "home=$HOME"; }
main() { parse_args "$@"; inner; }
main "$@"
