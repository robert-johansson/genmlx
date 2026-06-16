# Joint Inference of Biases and Preferences I: Procrastination and No-Exploration

> **Ports** agentmodels.org 5d-joint-inference.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

A Pac-Man that never explores an unknown corridor: the Optimal model explains it as strongly preferring the known reward; the Possibly-Reward-myopic model explains it as greedy (low C_g). The procrastination scenario: a Pac-Man that must hit a switch before a countdown — Naive hyperbolic Pac-Man keeps deferring, while the Optimal-only model wrongly infers the switch reward is tiny + noise is huge. Online posteriors revise sharply when the task finally completes.

## What this chapter builds on

- **genmlx code:** exists — joint inference over agent TYPE and parameters (U, alpha, b0, k, C). Tests: test/genmlx/agentmodels_5d_joint_test.cljs (optimal explains waiting via E[reward]~0.530 + E[alpha]~457.5, predictWorkLastMinute~0.006; discounting via E[discount]~2.63, predict~0.216; posterior revision on completion E[reward]>2.0 and E[alpha] collapses >100x; online sequence 9 entries). Example examples/agentmodels/joint_5d_inference.cljs. Built on biased_planners.cljs + inverse.cljs p/assess + standard inference backends.

## Live figures (planned)

- ch12-procrastination-timeseries.gif — posterior E[reward], E[alpha], E[discount], predict-work over t=0..9, optimal vs discounting (presentation bars per step)
- ch12-noexplore-horizon.png — E[chocolate utility] and E[C_g] vs time horizon 2..10, optimal vs reward-myopic
- ch12-completion-revision.png — before/after posterior bars when the switch is finally hit

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
