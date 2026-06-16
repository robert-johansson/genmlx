# Reasoning About Agents: Inverse Reinforcement Learning

> **Ports** agentmodels.org 4-reasoning-about-agents.md.

*(Chapter content is authored under the book's content epics. Code is lifted verbatim from a runnable `examples/agentmodels/*.cljs`; every figure is a live capture from the Pac-Man environment — see [the authoring contract](../PLAN.md).)*

## The Pac-Man scenario

An observer watches Pac-Man's path and infers which pellet cache it values most. One leftward step makes the near cache the MAP goal; symmetric caches stay unidentifiable until paths diverge — motivating active learning and multiple trajectories. Joint inference of utility + softmax noise alpha + time cost. POMDP-IRL adds inferring Pac-Man's beliefs about hidden ghosts from its detours.

## What this chapter builds on

- **genmlx code:** exists — src/genmlx/agents/inverse.cljs (goal-agents, action-loglik via p/assess on the goal policy, normalize-logs, posterior-sequence batched [G,S,A], observe-rollout). The likelihood IS p/assess — no bespoke likelihood code. Tests: agentmodels_irl_test.cljs (Eq 1 utility-table inference, donut MAP after one step, Veg/Noodle unidentifiable, soft-alpha washout, POMDP-IRL Eq 2 factorSequence, joint alpha+utility, deterministic via with-redefs of rng/fresh-key) + agentmodels_hike_irl_test.cljs (Big-Hiking S=36, posterior flat over shared prefix then concentrates P>0.99). Examples irl.cljs, pomdp_irl.cljs, hike_irl.cljs.

## Live figures (planned)

- ch07-observed-path.png — the trajectory to be explained, on the maze
- ch07-posterior-marginals.gif — prior→posterior bars over favorite cache as steps accrue (presentation/marginals->bars)
- ch07-single-step.png — one-action [3,1]→[2,1] conditioning illustration
- ch07-multiple-trajectories.png — two side-by-side paths sharpening the posterior

<!-- TODO(content): prose + code listings lifted from the example + embedded live figures -->
