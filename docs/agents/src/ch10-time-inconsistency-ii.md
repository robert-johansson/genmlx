# Time Inconsistency II: Procrastination and Changing Plans

> **Ports** agentmodels.org 5b-time-inconsistency.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

Procrastination MDP = Pac-Man must eat a power pellet before a deadline or lose the level (work/wait actions, deadline T, diminishing reward). Naive hyperbolic Pac-Man keeps deferring the grab — always 'next step' — until forced or failing; Sophisticated commits. dynamicActionExpectedUtilities shows EU of each action changing with position and subjective delay along the trajectory. Discount-rate sweep finds the threshold above which Pac-Man never acts.

## What this chapter builds on

- **genmlx code:** exists — src/genmlx/agents/biased_planners.cljs (procrastination-mdp time-augmented wait/work MDP, biased-eu with perceived-delay branching, planned-rollout for the believed-vs-actual divergence). Tests: agentmodels_biased_planners_test.cljs (procrastination preference reversal — naive procrastinates, sophisticated commits). The dynamic-EU-along-trajectory overlay (per-step Q at the agent's current delay) is presentation-layer work to write for the figure.

## Live figures (planned)

- ch10-procrastination-graph.png — work/wait transition graph authored figure
- ch10-dynamic-eu.gif — per-step action-EU overlay changing as Naive Pac-Man defers (presentation V-shading per delay)
- ch10-discount-threshold.png — final-state / time-elapsed vs discount rate k sweep

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
