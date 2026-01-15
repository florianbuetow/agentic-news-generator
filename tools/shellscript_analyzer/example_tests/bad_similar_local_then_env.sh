#!/bin/sh
# Reads env var after apparently defining similarly named local
parse_args() { PATH_ARG="$1"; }
parse_args "$@"
PATH_LOCAL="$PATH_ARG"
echo "path_local=$PATH_LOCAL"
echo "path_env=$PATH"
