#!/bin/sh
# Conditional assignment then use (may be unset)
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --mode) MODE="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
}
parse_args "$@"
if [ "$MODE" = "prod" ]; then
  LEVEL="high"
fi
echo "level=$LEVEL"
