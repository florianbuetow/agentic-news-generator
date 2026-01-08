#!/bin/bash
FILE="../../data/output/newspaper/articles/japan-and-us-partner-on-robotics-research-lab/index.html"

echo "=== Verifying Static Site ===" 
echo "Checking: $FILE"

if grep -q '<div class="article-body".*<h1' "$FILE"; then
  echo "❌ FAIL: H1 found in article body"
  exit 1
else
  echo "✓ PASS: No H1 in article body"
fi

if grep -q '<div class="article-body".*<h2' "$FILE"; then
  echo "❌ FAIL: H2 found in article body"
  exit 1
else
  echo "✓ PASS: No H2 in article body"
fi

if grep -q 'TOKYO' "$FILE"; then
  echo "✓ PASS: Article content is present"
else
  echo "❌ FAIL: Article content missing"
  exit 1
fi

echo -e "\n✓ All checks passed! Duplicate titles removed successfully."
