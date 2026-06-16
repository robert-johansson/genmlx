# Learning the Maze: Bandits, Thompson Sampling, and PSRL

> **Ports** agentmodels.org 3d-reinforcement-learning.md (also the bandit arm of 3c).

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

Multi-armed bandit = Pac-Man repeatedly choosing among corridors, tracking which yields more fruit (cumulative regret vs random baseline). PSRL on a gridworld = Pac-Man learning an unknown maze: the unknown reward function is the unknown fruit layout; each episode it samples a hypothesized layout, plans optimally, then updates beliefs from what it actually found. The lava-world variant shows uncertainty-driven exploration avoiding danger.

## What this chapter builds on

- **genmlx code:** exists — src/genmlx/agents/pomdp.cljs (make-bandit-agent Thompson/softmax, update-arm Beta-Bernoulli, simulate-bandit + simulate-bandit-batched [N,K] no per-step mx/item); src/genmlx/agents/pomdp_env.cljs (bandit-pomdp); src/genmlx/agents/worlds.cljs (lava-world). Tests: agentmodels_psrl_test.cljs (posterior concentrates P>=0.99, final-episode zero regret, beats baseline), agentmodels_psrl_lava_test.cljs (reduces lava exposure across 6 seeds). Example examples/agentmodels/psrl.cljs. PSRL gridworld loop reuses agent.cljs make-mdp-agent per sampled model.

## Live figures (planned)

- ch06-bandit-regret.png — cumulative-regret line, Thompson/softmax vs random baseline
- ch06-psrl-episodes.gif — concatenated per-episode Pac-Man trajectories getting more direct to fruit
- ch06-lava-avoidance.gif — PSRL Pac-Man learning to skirt the lava gap
- ch06-arm-posteriors.gif — Beta posteriors over corridor fruit-rates sharpening (presentation/bandit-bars)

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
