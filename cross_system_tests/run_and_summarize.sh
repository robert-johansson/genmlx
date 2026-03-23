#!/bin/sh
# Cross-system verification: run all tests and produce summary.
# Usage:
#   ./cross_system_tests/run_and_summarize.sh              # default suites (no inference_quality)
#   ./cross_system_tests/run_and_summarize.sh all           # all suites including inference_quality
#   ./cross_system_tests/run_and_summarize.sh logprob       # specific suite(s)
#
# Exits non-zero if any test fails.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SUMMARY_FILE="$SCRIPT_DIR/results/SUMMARY.md"

cd "$PROJECT_DIR"

# Determine which suites to run
if [ "$1" = "all" ]; then
    # Run all suites including the slow inference_quality suite
    ARGS="mlx_ops logprob assess score_decomposition update project regenerate combinator stability gradient inference_quality"
elif [ -n "$1" ]; then
    ARGS="$*"
else
    ARGS=""
fi

echo "Running cross-system verification..."
echo ""

# Run the orchestrator. It exits 1 on failure.
EXIT_CODE=0
bb cross_system_tests/run_all.clj $ARGS || EXIT_CODE=$?

echo ""
echo "================================================================"

# Print the summary if it was generated
if [ -f "$SUMMARY_FILE" ]; then
    echo ""
    cat "$SUMMARY_FILE"
else
    echo "No summary generated (check for errors above)."
fi

exit $EXIT_CODE
