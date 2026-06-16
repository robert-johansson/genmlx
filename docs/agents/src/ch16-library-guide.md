# Quick-Start Guide to the genmlx.agents Library

> **Ports** agentmodels.org 8-guide-library.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

A tutorial-style tour building each agent on Pac-Man worlds: line MDP = corridor; gridworld with walls = maze; named features (gold/silver) = pellets/power-pellets with different utilities; line POMDP with signpost = a corridor with a hidden ghost (treasureAt3 → ghostAt[x,y]); custom RANDOM and EPSILON-GREEDY policy GFs. Shows the minimal agent contract (:act + :params) and the four constructors.

## What this chapter builds on

- **genmlx code:** exists — src/genmlx/agents/CONTRACTS.md (frozen v1.0 API: the four constructor shapes), worlds.cljs (line-mdp, hike-mdp), agent.cljs (make-mdp-agent optimal/soft), biased_planners.cljs (Naive hyperbolic), pomdp.cljs + pomdp_env.cljs (line POMDP belief filter). Tests: test/genmlx/agentmodels_ch08_library_test.cljs (make-line-mdp + agent, hiking, naive biased, POMDP belief snaps at signpost + QMDP reaches goal, custom RANDOM + EPSILON-GREEDY GFs) + agents_api_test.cljs (8 namespaces load) + agents_contracts_test.cljs (per-constructor return-map shapes + :act signatures pinned). Example examples/agentmodels/ch08_library_guide.cljs.

## Live figures (planned)

- ch16-corridor-basic.png — bare line MDP / 3×4 gridworld with walls (render.cljs)
- ch16-named-features.gif — agent navigating to the gold (power-pellet) terminal
- ch16-custom-policy.gif — epsilon-greedy Pac-Man wandering then exploiting

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
