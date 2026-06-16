# Multi-Agent Models: Coordination, Adversaries, and Signaling

> **Ports** agentmodels.org 7-multi-agent.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

Schelling coordination = two Pac-Men trying to meet at a junction without communicating, the popular corridor as the focal point; recursive theory-of-mind at increasing depth predicts convergence. Adversarial = Pac-Man vs ghost planning ahead via simulate/act alternation (minimax/expectimax) on the cs188 GameState. RSA signaling = Pac-Man signaling ghost positions to a partner through movement patterns (literal vs pragmatic listener/speaker).

## What this chapter builds on

- **genmlx code:** exists — Schelling + focal-point amplification + RSA scalar implicature shipped. Tests: test/genmlx/agentmodels_multi_agent_test.cljs (bob(0)=0.55 prior, focal amplification strictly increasing, alice(4)>=0.83; exact ~ importance-sampling tol 0.05; L0/S1/L1 RSA, 'some' implies not-all, pragmatic posteriors analytically verified). Example examples/agentmodels/multi_agent.cljs. Adversarial Pac-Man tree search ALREADY exists in cs188.agents (minimax/alpha-beta/expectimax over generate-successor, proven alpha-beta==minimax) — partial: bridging cs188.agents adversarial search into a genmlx multi-agent GF narrative is the one must-write piece.

## Live figures (planned)

- ch15-schelling-convergence.png — meeting-probability vs recursion depth line chart
- ch15-adversarial-pacman.gif — Pac-Man vs ghost expectimax self-play (cs188 play.cljs / web canvas)
- ch15-rsa-implicature.png — literal vs pragmatic listener posterior bars

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
