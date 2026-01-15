#!/bin/sh
# Here-doc plus env expansion
parse_args() { APP="$1"; }
parse_args "$@"
cat <<EOF
app=$APP
home=$HOME
EOF
