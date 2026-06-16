# Time Inconsistency I: Hyperbolic Discounting in the Maze

> **Ports** agentmodels.org 5a-time-inconsistency.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

Hyperbolic discounting = Pac-Man increasingly tempted by a NEARBY power pellet even while heading to a higher-value distant pellet. Naive Pac-Man gets diverted to the close pellet; Sophisticated Pac-Man pre-commits to a route that avoids passing the tempting pellet, paying extra steps. Immediate vs delayed utility = eat a pellet now (small score) vs a power pellet enabling ghost-eating later (large delayed score). Discount-curve comparison D=1/2^t vs 1/(1+2t).

## What this chapter builds on

- **genmlx code:** exists — src/genmlx/agents/biased_planners.cljs (delta hyperbolic 1/(1+kd), bias->perceived-delay :naive=inc / :sophisticated=const-0, biased-eu/biased-eu-inf, make-biased-mdp-agent, simulate-biased-mdp re-planning each step, planned-rollout = believed trajectory, restaurant-mdp + restaurant-temptation-mdp). Tests: agentmodels_biased_planners_test.cljs (delta limit cases, restaurant-temptation naive vs sophisticated action differences, limit recovery k=0 matches standard agent at alpha=1 and Inf). Discount-curve plot to write for figure.

## Live figures (planned)

- ch09-discount-curves.png — exponential vs hyperbolic discount-curve line chart
- ch09-naive-detour.gif — Naive Pac-Man diverting to the tempting near pellet
- ch09-sophisticated-precommit.gif — Sophisticated Pac-Man taking the avoidance route

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
