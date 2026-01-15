#!/bin/sh
# Uses indirect expansion
VAR_NAME="SECRET"
echo "${!VAR_NAME}"
