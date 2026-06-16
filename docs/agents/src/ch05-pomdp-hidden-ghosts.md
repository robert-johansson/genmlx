# Hidden State: POMDPs and Belief Over Ghosts

> **Ports** agentmodels.org 3c-pomdp.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

Ghost positions are the canonical hidden state — Pac-Man can't see ghosts through walls and maintains a belief distribution over their locations, updating it when a ghost enters the observable neighborhood. Restaurant-open/closed latent maps to 'is this corridor safe?'. Signpost reveal and adjacency-reveal both shown. QMDP belief-directed exploration and rational replanning when a corridor is found blocked.

## What this chapter builds on

- **genmlx code:** exists — src/genmlx/agents/belief.cljs (filter-step differentiable Bayes update, tensor-update-belief, obs-id-tensor, belief<->vec); src/genmlx/agents/pomdp.cljs (make-pomdp-agent QMDP [W,S,A] belief-Q, simulate-pomdp, fused-simulate-pomdp); src/genmlx/agents/pomdp_env.cljs (restaurant-gridworld signpost, restaurant-pomdp adjacency-reveal 2^k worlds). Tests: agentmodels_pomdp_test.cljs (belief snaps at signpost, QMDP reaches goal), agentmodels_pomdp_adjacency_test.cljs (detour when preferred closed), agents_fused_pomdp_test.cljs (S×W obs-tensor parity). No new agent code; ghostbusters belief-heatmap browser renderer is the figure gap.

## Live figures (planned)

- ch05-belief-heatmap.gif — belief over ghost cells concentrating as Pac-Man senses (cs188 ghostbusters shade → ghostbusters.html canvas extension, or ANSI fallback)
- ch05-pomdp-graph.png — POMDP dependency graph (s,a,o,u,b) authored figure
- ch05-signpost-reveal.gif — belief snaps to revealed goal at the signpost cell
- ch05-adjacency-detour.gif — Pac-Man detours after observing a corridor closed

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
