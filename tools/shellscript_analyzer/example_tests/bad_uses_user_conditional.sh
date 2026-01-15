#!/bin/sh
# Uses USER in a conditional
if [ "$USER" = "root" ]; then
    echo "running as root"
fi
