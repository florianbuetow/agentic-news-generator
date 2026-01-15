#!/bin/sh
# Sourcing inside a function
parse_args() { CFG="$1"; }
load_cfg() { . "$CFG"; }
parse_args "$@"
load_cfg
echo "done"
