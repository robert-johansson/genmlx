# agentmodels TUI gallery

A terminal gallery for the [agentmodels.org](https://agentmodels.org) port —
watch GenMLX agents plan and walk through gridworlds, right in your terminal.
This is the **demo gallery** (distinct from the Studio TUI, bean `genmlx-tw52`).

## The architecture in one line

Everything below the **render-agnostic data seam** (`genmlx.agents.presentation`:
`Frame` / `Trajectory` / `PosteriorBars`) is pure CLJS + MLX and is proven by a
headless test; everything here above the seam is just reagent + Ink turning that
same data into colored cells.

```
gridworld.cljs ─► agent.cljs ─► presentation.cljs ─►║ Frame  ║─► views.cljs ─► gallery / demos
   (env tensors)   (VI + GFI      (pure producers +  ║ Bars   ║   (Ink views)    (this sub-app)
   inverse.cljs    policy)        ASCII renderers)    ║ SEAM   ║
   pomdp{,_env}.cljs (QMDP belief filtering)          ╚════════╝
```

Four demos: **Ch 3** (an MDP agent walking a maze), **Ch 3c** (a POMDP agent
acting under uncertainty about which goal pays, its belief snapping to the truth
when it reaches a signpost), **Ch 3d** (bandits — Beta belief per arm, Thompson
posterior sampling concentrating pulls on the best arm), and **Ch 5** (inverse
goal inference — infer what an agent wants from how it moves, watching the
posterior sharpen live).

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
| Ch 3c demo | `space` | step the agent one frame |
| Ch 3c demo | `r` | resample a fresh belief-filtered rollout |
| Ch 3c demo | `t` | toggle which goal is actually rewarding |
| Ch 3c demo | `q` / `esc` | back to the menu |
| Ch 3d demo | `space` | step one pull |
| Ch 3d demo | `r` | resample a fresh bandit run |
| Ch 3d demo | `t` | toggle Thompson ↔ softmax-greedy |
| Ch 3d demo | `q` / `esc` | back to the menu |
| Ch 5 demo | `space` | reveal the next observed action |
| Ch 5 demo | `r` | resample a fresh walk |
| Ch 5 demo | `t` | toggle the agent's true goal (A / B) |
| Ch 5 demo | `q` / `esc` | back to the menu |

**Ch 5** plays the true agent's walk on the left and the observer's `P(goal)`
bars on the right; the bars stay flat while the agent heads down the symmetry
axis, then snap to the true goal the instant it turns. The likelihood is
`p/assess` on each goal's policy — the forward agent model, inverted.

The Ch 3 agent auto-walks a maze where a wall belt forces a detour. Two separable
sources of randomness: **alpha** is *decision* noise (low alpha → worse actions;
`INF` → optimal), while **n** sets *environment* noise — the agentmodels
orthogonal slip, where the intended move slips to a perpendicular one. Raise
alpha toward `INF` at noise 0 and the path snaps to the sharp optimal route to
the high-utility goal `B`; add noise and the same policy visibly skids off course.

## What proves it works

The whole pipeline below the seam is verified without a terminal:

```bash
bun run --bun nbb test/genmlx/agentmodels_slice_test.cljs   # 51/51  (Ch3/Ch5 + recursive==tensor)
bun run --bun nbb test/genmlx/agentmodels_pomdp_test.cljs   # 22/22  (Ch3c belief filtering + QMDP)
bun run --bun nbb test/genmlx/bandit_test.cljs              # 18/18  (Ch3d Beta filter + Thompson)
```

Because the views are pure functions of the data that test validates, a passing
test means the live TUI renders correct pictures by construction.

## Adding a demo

Append one entry to `demos` in `gallery.cljs`:
`{:id … :title … :view … :on-key … :enter … :leave …}`. No nav code to touch.
View primitives (`grid-view`, `bars-view`, `frames-view`, `status-bar`) live in
`views.cljs` and consume only `genmlx.agents.presentation` data shapes.
