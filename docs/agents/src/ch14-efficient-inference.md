# Efficient Inference: One Model, Many Backends

> **Ports** agentmodels.org 6-efficient-inference.md, 6a-inference-dp.md, 6b-inference-sampling.md, 6c-inference-rl.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

The same Pac-Man goal-inference problem solved four ways that MUST agree: (6a) DP — value iteration over maze states == faithful recursive EU; (6b) sampling — exact enumeration vs importance sampling vs Metropolis-Hastings over the joint goal-inference GF; (6c) RL/gradient — gradient and amortized utility recovery through the differentiable planner. Demonstrates inference is a pluggable orthogonal axis over the agent-GF.

## What this chapter builds on

- **genmlx code:** exists — src/genmlx/agents/differentiable.cljs (build-diff-mdp, diff-q via value-iteration-lazy, action-loglik-loss, recover-params Adam, loss-at). agent.cljs value-iteration vs recursive-eu certified equal. Tests: test/genmlx/agentmodels_ch06_pluggable_inference_test.cljs (VI==recursive-EU, enumerate/IS/MH agreement cross-checked by independent posterior-sequence oracle, gradient/amortized recovery) + agentmodels_diff_learn_test.cljs (diff-reward reconstructs R, lazy Q matches eager 1e-5, fix-alpha-learn-utilities + fix-utilities-learn-alpha to likelihood-equivalence). Example examples/agentmodels/ch06_pluggable_inference.cljs. The 6a/6b/6c stubs in agentmodels are fleshed out here.

## Live figures (planned)

- ch14-backend-agreement.png — same posterior from enumerate/IS/MH overlaid as bars
- ch14-dp-value-iteration.gif — V(s) sweeps converging across the maze (cs188 rl.cljs value-iteration display → canvas)
- ch14-gradient-recovery.png — loss-history curve recovering planted utilities + alpha

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
