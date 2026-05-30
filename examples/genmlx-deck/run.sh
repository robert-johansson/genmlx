#!/usr/bin/env bash
# ============================================================================
# GenMLX self-demonstrating slide deck (ink + nbb)
# ============================================================================
# Runs from the repo root so nbb.edn puts genmlx.* on the classpath, and
# borrows ink/react/ink-spinner from ../genmlx-tui/node_modules via NODE_PATH.
# Uses node `nbb` (ink needs the React reconciler + yoga wasm).
#
#   ./run.sh                     # interactive deck (needs a real terminal)
#   DECK_SELFTEST=1    ./run.sh  # headless: run every pure-GenMLX figure
#   DECK_SELFTEST=full ./run.sh  # headless: also load the LLM and run 03/04/05
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."                     # repo root
ROOT="$(pwd)"

if [ ! -d "$ROOT/examples/genmlx-tui/node_modules/ink" ]; then
  echo "ink not found. Install once with:"
  echo "  (cd examples/genmlx-tui && npm install)"
  exit 1
fi

# OS_ACTIVITY_MODE=disable silences macOS os_log / CoreAnalytics "Context leak"
# noise that would otherwise print over the full-screen UI during GPU work.
OS_ACTIVITY_MODE=disable \
NODE_PATH="$ROOT/examples/genmlx-tui/node_modules:$ROOT/node_modules" \
  exec nbb examples/genmlx-deck/deck.cljs "$@"
