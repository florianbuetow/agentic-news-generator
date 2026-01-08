#!/bin/bash
cd /Volumes/2TB/agentic-news-generator.git/florian-html-newspaper-generation/frontend/newspaper

echo "=== Testing Article Filtering ===" 
npm run dev > /tmp/server.log 2>&1 &
SERVER_PID=$!

echo "Waiting for server..."
for i in {1..20}; do
  curl -s http://localhost:3000 >/dev/null 2>&1 && break
  sleep 1
done

curl -s "http://localhost:3000/articles/japan-and-us-partner-on-robotics-research-lab" > /tmp/test-output.html

echo "Results:"
grep -q '<div class="article-body".*<h1' /tmp/test-output.html && echo "❌ H1 in body" || echo "✓ No H1 in body"
grep -q '<div class="article-body".*<h2' /tmp/test-output.html && echo "❌ H2 in body" || echo "✓ No H2 in body"

kill $SERVER_PID 2>/dev/null
