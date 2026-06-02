#!/bin/bash
# Run the agentmodels TUI gallery.
#   ./run.sh          interactive gallery (menu -> demos; needs a real terminal)
#   ./run.sh views    static views_demo smoke (renders sample data, then exits)
#
# Mirrors examples/genmlx-tui/run.sh: ink resolves from the local node_modules,
# reagent / nbb / @mlx-node from the repo-root node_modules (NODE_PATH). The
# explicit --classpath adds the repo source roots plus this sub-app dir so the
# local gallery/views/ch3_demo requires and the agentmodels.* library resolve.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/examples/agentmodels-tui"
TARGET="gallery.cljs"
[ "$1" = "views" ] && TARGET="views_demo.cljs"
cd "$ROOT"
NODE_PATH="$HERE/node_modules:$ROOT/node_modules" \
  npx nbb --classpath "src:examples:test:malli/src:instaparse/src:examples/agentmodels-tui" \
    "examples/agentmodels-tui/$TARGET"
