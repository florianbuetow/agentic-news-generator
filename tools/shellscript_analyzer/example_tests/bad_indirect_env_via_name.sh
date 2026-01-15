#!/bin/sh
# Indirect env use via name variable
parse_args() { KEY="$1"; }
parse_args "$@"
VAR_NAME="HOME"
echo "key=$KEY"
echo "value=${!VAR_NAME}"
