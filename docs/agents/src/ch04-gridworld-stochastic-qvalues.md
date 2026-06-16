# Gridworld in Depth: Stochastic Transitions and Q-Values

> **Ports** agentmodels.org 3b-mdp-gridworld.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

Hiking MDP as a Pac-Man maze: West/East peaks = high-value pellet clusters, the Hill hazard row = a ghost corridor that ends the episode with negative reward, transitionNoiseProbability = a slippery maze floor (orthogonal slip). Deterministic Pac-Man cuts the short risky route; the stochastic agent detours the long safe route. Q-value heatmap shows per-cell action values; alpha sweep tunes player noisiness.

## What this chapter builds on

- **genmlx code:** exists — src/genmlx/agents/worlds.cljs (hike-mdp 5×5 noise 0.0/0.1, big-hike-mdp 6×6); gridworld.cljs transition-tensor implements orthogonal-slip noise; agent.cljs value-iteration yields Q for the heatmap; alpha lives in softmax-action (helpers.cljs). Verified in test/genmlx/agentmodels_worlds_test.cljs (deterministic bottom-row cut vs stochastic top detour, both reach East peak). Trajectory-length histogram + Q-value-overlay rendering to write for figures (presentation.cljs env->trajectory accepts an MLX V for value shading).

## Live figures (planned)

- ch04-hike-deterministic.gif — short risky route past the ghost corridor
- ch04-hike-stochastic.gif — long safe detour under slippery floor
- ch04-qvalue-heatmap.png — per-cell action Q-values overlaid on the maze (presentation V-shading + render.cljs)
- ch04-trajectory-length-hist.png — histogram of rollout lengths under stochastic transitions

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
