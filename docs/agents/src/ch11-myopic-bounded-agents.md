# Bounded Agents: Reward-Myopia and Update-Myopia

> **Ports** agentmodels.org 5c-myopic.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

Reward-myopic Pac-Man only looks C_g steps ahead — greedy about nearby pellets, may miss optimal routes to power pellets (epsilon-greedy-like on bandits). Update-myopic Pac-Man assumes belief updates stop after C_m steps: with hidden ghosts observable only by moving adjacent, a C_m=1 Pac-Man won't plan a route that first reveals ghost positions then exploits the info — the Restaurant-Search failure mode. Runtime: update-myopic scales better than full POMDP.

## What this chapter builds on

- **genmlx code:** exists — src/genmlx/agents/biased_planners.cljs (reward-myopia C_g and update-myopia C_m bounds in the delay-indexed recursion, line-mdp reward-myopia corridor, make-biased-pomdp-agent with C_m bound, simulate-biased-pomdp, voi-world value-of-information walk-and-check POMDP). Limit recovery biased(C_g=Inf) matches standard agent — agentmodels_biased_planners_test.cljs. Bandit reward-myopia regret-ratio + runtime-scaling timing harness to write for figures.

## Live figures (planned)

- ch11-rewardmyopic-bandit.png — average-reward-near-optimal-despite-myopia bar/line
- ch11-voi-failure.gif — update-myopic Pac-Man avoiding the info-revealing detour (voi-world)
- ch11-runtime-scaling.png — runtime vs trials: update-myopic vs optimal POMDP line chart

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
