# Sequential Decisions: MDPs and the Maze

> **Ports** agentmodels.org 3a-mdp.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

Integer-line MDP = a 1-D Pac-Man corridor with a pellet at the far end and a per-step time cost. Restaurant-Choice gridworld = a Pac-Man maze with pellet caches of different values as terminal cells. Show act/expectedUtility mutual recursion, terminal states, and the exponential-blowup → dp.cache (memoization) lesson as planning-step count grows.

## What this chapter builds on

- **genmlx code:** exists — src/genmlx/agents/agent.cljs (bellman-step, value-iteration, value-iteration-lazy, recursive-eu memoized via exact/with-cache, make-mdp-agent, simulate-mdp); src/genmlx/agents/worlds.cljs (line-mdp); src/genmlx/agents/gridworld.cljs (parse-grid, build-mdp, transition-tensor). Tests: test/genmlx/agentmodels_worlds_test.cljs (line S=7 rollout 0→6) + agentmodels_slice_test.cljs (3×3 VI Q/V + rollout). The dp.cache vs no-cache runtime-scaling demo is the agent.cljs two-path (tensor VI vs recursive-eu) story — a small timing harness to write for the figure.

## Live figures (planned)

- ch03-corridor.gif — Pac-Man walking the 1-D line MDP to the goal pellet
- ch03-gridworld-policy.png — maze with optimal-policy direction arrows per cell (render.cljs overlay)
- ch03-runtime-scaling.png — bar chart: recursive EU runtime with vs without with-cache memoization

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
