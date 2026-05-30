#!/bin/bash
# Run the GenMLX TUI prototype
# Uses nbb from genmlx's node_modules, ink from local node_modules
cd "$(dirname "$0")"
NODE_PATH="$(pwd)/node_modules:$(cd ../.. && pwd)/node_modules" \
  npx nbb genmlx_tui.cljs
