#!/usr/bin/env bash
# ============================================================================
# GenMLX VLM deck — perceiving ARC-AGI-3 scenes (ink + nbb)
# ============================================================================
# Runs from the repo root (so nbb.edn puts genmlx.* on the classpath), borrows
# ink/react from ../genmlx-tui/node_modules, and gets @mlx-node/lm from the repo
# node_modules. Uses node `nbb` (ink needs the React reconciler).
#
#   ./run.sh                     # interactive deck (needs a real terminal)
#   DECK_SELFTEST=1    ./run.sh  # headless: load + preview the ARC grids
#   DECK_SELFTEST=full ./run.sh  # headless: also load the VLM and describe each scene
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."
ROOT="$(pwd)"

if [ ! -d "$ROOT/examples/genmlx-tui/node_modules/ink" ]; then
  echo "ink not found. Install once with: (cd examples/genmlx-tui && npm install)"; exit 1
fi
if [ ! -f "$ROOT/dev/arc_frames/sk48_deck.png" ]; then
  echo "ARC scene PNGs missing. Generate them with the prep step in dev/ (see README)."; exit 1
fi

# OS_ACTIVITY_MODE=disable silences macOS CoreAnalytics "Context leak" noise.
OS_ACTIVITY_MODE=disable \
NODE_PATH="$ROOT/examples/genmlx-tui/node_modules:$ROOT/node_modules" \
  exec nbb examples/genmlx-vlm-deck/deck.cljs "$@"
