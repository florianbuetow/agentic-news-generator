#!/bin/sh
# Case/shift parsing but hidden env dependency
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --url) URL="$2"; shift 2 ;;
      --token) TOKEN="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
}
run() { echo "curl $URL with token=$TOKEN"; echo "user=$USER"; }
parse_args "$@"
run
