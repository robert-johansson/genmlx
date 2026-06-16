# GenMLX Agents

**Inferring minds in the maze.**

This book ports the curriculum of *Modeling Agents with Probabilistic Programs*
(agentmodels.org) to GenMLX — and tells the whole story inside one environment: a
Pac-Man maze.

The thesis runs through every chapter: **an agent is a generative function, and
inference is a pluggable, orthogonal axis.** A policy is a `gen` function that
traces an action; planning is running it; *inverse* planning — asking what an agent
wanted or believed — is inference *over* that same function. Exact enumeration,
Monte-Carlo, gradients, or an LLM can each play the role of the inference engine
without the model changing a line.

Why Pac-Man? Because almost the entire arc — Markov decision processes, gridworlds,
partial observability, reinforcement learning, inverse reward inference, cognitive
biases, and multi-agent games — is *sequential decision-making on a grid under a
utility function*, and that is exactly Pac-Man. Pellets are rewards; the power
pellet is a large, delayed payoff; a ghost you cannot see through a wall is hidden
state. One maze gives the book a single visual language, and every figure in it is
a **live capture** of real GenMLX inference, not a hand-drawn diagram.

> **Honest scope.** This is Pac-Man-*primary*, not Pac-Man-*only*. A ghost is
> modeled as POMDP *latent* state — a belief over where it is — never as part of
> the planner's state vector, so planning stays tractable. The probabilistic-
> programming primer and the signaling material stay deliberately plain; the
> adversarial multi-ghost game drops to classical tree search and is labeled as
> such. Where the maze illuminates, we use it; where it would obscure, we don't.

## How to read this book

Every chapter shows runnable GenMLX code, lifted verbatim from a file under
`examples/agentmodels/`, and every figure is regenerated from an actual rollout or
posterior. The substrate is one namespace — `genmlx.agents.pacman` — a thin layer
over the `genmlx.agents` library. Start with the [legend](./legend.md), which fixes
the maze's visual vocabulary once.

## Running the code

```bash
bun run --bun nbb examples/agentmodels/<chapter>.cljs
```
