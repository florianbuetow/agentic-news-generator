#!/bin/sh
# Uses variable set only conditionally
if [ "$1" = "on" ]; then
    MODE="on"
fi
echo "$MODE"
