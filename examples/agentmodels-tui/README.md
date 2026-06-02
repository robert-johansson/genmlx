# agentmodels TUI gallery

A terminal gallery for the [agentmodels.org](https://agentmodels.org) port —
watch GenMLX agents plan and walk through gridworlds, right in your terminal.
This is the **demo gallery** (distinct from the Studio TUI, bean `genmlx-tw52`).

## The architecture in one line

Everything below the **render-agnostic data seam** (`agentmodels.presentation`:
`Frame` / `Trajectory` / `PosteriorBars`) is pure CLJS + MLX and is proven by a
headless test; everything here above the seam is just reagent + Ink turning that
same data into colored cells.

```
gridworld.cljs ─► agent.cljs ─► presentation.cljs ─►║ Frame ║─► views.cljs ─► gallery/ch3_demo
   (env tensors)   (VI + GFI      (pure producers +  ║ SEAM  ║   (Ink views)    (this sub-app)
                    policy)        ASCII renderers)   ╚═══════╝
```

## Run it

```bash
cd examples/agentmodels-tui
npm install            # ink + react into local node_modules (one time)
./run.sh               # interactive gallery (needs a real terminal)
./run.sh views         # static views smoke (renders sample data, then exits)
```

`run.sh` mirrors `examples/genmlx-tui/run.sh` (ink from local `node_modules`,
reagent/nbb/@mlx-node from the repo root via `NODE_PATH`) and adds an explicit
`--classpath` so the multi-file sub-app and the `agentmodels.*` library resolve.

## Keys

| screen | key | action |
|--------|-----|--------|
| menu | `↑`/`↓` | select a demo |
| menu | `enter` | open the selected demo |
| menu | `q` | quit |
| Ch 3 demo | `space` | step the agent one frame |
| Ch 3 demo | `r` | resample the rollout at the current alpha |
| Ch 3 demo | `+` / `-` | raise / lower the rationality alpha (`+` toward optimal) |
| Ch 3 demo | `n` | cycle the transition noise (0 → 0.1 → 0.2 → 0.4) |
| Ch 3 demo | `q` / `esc` | back to the menu |

The agent auto-walks a maze where a wall belt forces a detour. Two separable
sources of randomness: **alpha** is *decision* noise (low alpha → worse actions;
`INF` → optimal), while **n** sets *environment* noise — the agentmodels
orthogonal slip, where the intended move slips to a perpendicular one. Raise
alpha toward `INF` at noise 0 and the path snaps to the sharp optimal route to
the high-utility goal `B`; add noise and the same policy visibly skids off course.

## What proves it works

The whole pipeline below the seam is verified without a terminal:

```bash
bun run --bun nbb test/genmlx/agentmodels_slice_test.cljs   # 39/39
```

Because the views are pure functions of the data that test validates, a passing
test means the live TUI renders correct pictures by construction.

## Adding a demo

Append one entry to `demos` in `gallery.cljs`:
`{:id … :title … :view … :on-key … :enter … :leave …}`. No nav code to touch.
View primitives (`grid-view`, `bars-view`, `frames-view`, `status-bar`) live in
`views.cljs` and consume only `agentmodels.presentation` data shapes.
