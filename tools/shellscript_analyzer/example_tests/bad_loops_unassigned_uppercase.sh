#!/bin/sh
# Loops and locals but unassigned uppercase use
sum=0
for n in 1 2 3; do
  sum=$((sum+n))
done
echo "sum=$sum"
echo "build=$BUILD_ID"
