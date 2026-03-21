#!/usr/bin/env bash
# Run all GenMLX examples and report results.
# Usage: bash examples/run_all.sh

set -e
cd "$(dirname "$0")/.."

passed=0
failed=0
errors=""

for f in examples/memo/*.cljs examples/genmlx/*.cljs; do
  name=$(basename "$f" .cljs)
  if bun run --bun nbb "$f" > /dev/null 2>&1; then
    echo "  PASS: $name"
    ((passed++))
  else
    echo "  FAIL: $name"
    ((failed++))
    errors="$errors $name"
  fi
done

echo ""
echo "Examples: $passed passed, $failed failed"
if [ $failed -gt 0 ]; then
  echo "Failed:$errors"
  exit 1
fi
