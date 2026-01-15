#!/bin/sh
# Uses eval in a larger flow
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --var) VARNAME="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
}
parse_args "$@"
cmd="echo \$VARNAME"
eval "$cmd"
