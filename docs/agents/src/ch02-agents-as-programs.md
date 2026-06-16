# Agents as Probabilistic Programs: One-Shot Choice at a Junction

> **Ports** agentmodels.org 3-agents-as-programs.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

Pac-Man at a 4-way junction: actions = up/down/left/right, transition returns the resulting cell (deterministic) or a ghost-perturbed distribution (stochastic), utility high for pellet cells / negative for ghost cells. maxAgent = one-step-lookahead Pac-Man; inferenceAgent = planning-as-inference ('imagine we observe Pac-Man ate the pellet — which way did it go?'); softMaxAgent = a noisy human player. Monty-Hall exercise → which of three corridors hides the ghost.

## What this chapter builds on

- **genmlx code:** exists — src/genmlx/agents/helpers.cljs implements factor-dist (WebPPL factor as defdist) and softmax-action (Boltzmann policy via dist/categorical, delegates to exact/categorical-argmax at alpha=Inf); verified in test/genmlx/agentmodels_helpers_test.cljs (factor score injection additive, softmax matches hand-computed reference). The agent-as-GF policy (gen [s] (trace :action (softmax-action alpha Q[s]))) is the core contract. No new code for the concepts; one small worked junction example to write.

## Live figures (planned)

- ch02-junction.png — static maze junction with 4 candidate next-cells annotated with utility
- ch02-softmax-bars.gif — action-probability bar chart sweeping alpha from 0 to Inf (presentation bars over a sweep)
- ch02-planning-as-inference.png — diagram: condition on outcome → posterior over action

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
