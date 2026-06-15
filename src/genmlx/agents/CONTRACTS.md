# genmlx.agents — API contracts (v1.0 surface)

The stable public surface of the `genmlx.agents` vertical, pinned by **tests as
contracts**. Every claim below is enforced by a test; if the surface drifts, the
test fails. This document is descriptive of what the code guarantees today — it is
not a license to change these signatures silently.

**Enforcing tests**
- `test/genmlx/agents_api_test.cljs` — reachability of the 8 promoted namespaces;
  an agent *is* a generative function (GFI ops); tensor-VI `Q` == recursive-EU;
  the `genmlx-xpbm` regressions; inverse-planning posteriors (host == batched).
- `test/genmlx/agents_contracts_test.cljs` — the per-constructor return-map shapes,
  the family-specific `:act` signatures, and the POMDP/bandit agent contracts.
- `test/genmlx/belief_tensor_test.cljs` (36/36) — the belief filter math: host ==
  tensor equivalence, nil-observation identity, impossible-observation defensiveness.

## The minimal agent contract

The **only** keys guaranteed on *every* agent returned by an `agents` constructor:

| key       | type           | meaning                                   |
|-----------|----------------|-------------------------------------------|
| `:act`    | fn             | the action entry point (signature is family-specific — see below) |
| `:params` | map            | the agent's parameters (for inspection/logging) |

Everything else is **family-specific**. In particular there is **no** uniform
`:policy`, `:Q`, or `:expected-utility` across all agents, and **no** single `:act`
arity. Consumers must dispatch on the agent family, not assume a universal shape.

## Constructors

### `agent/make-mdp-agent {:keys [mdp alpha gamma n-iters]}` — fully-observed MDP
- **Returns** `{:mdp :Q [S,A] :V [S] :policy <GF> :act :expected-utility :params}`.
- `:policy` is a generative function `(gen [s] (trace :action ...))` — full GFI
  (`p/simulate`/`p/assess`/`p/generate`/`p/update`).
- `:act` — **state-based**: `(act s)` draws fresh entropy, `(act s key)` is
  deterministic in `key`; both return an action int.
- `:Q`/`:V` come from tensor value iteration; `:expected-utility` is the faithful
  recursive path. The two agree to float32 (the flagship invariant).

### `biased-planners/make-biased-mdp-agent {:keys [mdp alpha gamma n-iters]} {:keys [discount bias reward-myopic-bound]}` — biased (hyperbolic) MDP
- **Returns** `{:mdp :policy <GF> :act :expected-utility :eu :params}` — **the same
  shape as make-mdp-agent MINUS `:Q`/`:V`.** The biased agent is recursion-only (the
  delay-indexed value has no single tensor table), so `:Q`/`:V` are **absent**.
- `:act` — **state-based**, identical arities to make-mdp-agent. Re-plans from delay 0
  each call (the Naive plan↔do divergence).
- At `discount 0` it recovers the unbiased agent (asserted).

### `pomdp/make-pomdp-agent {:keys [grid goals alpha noise gamma n-iters start prior observe world-utils]}` — partially-observed (QMDP)
- **Returns** `{:worlds :world-agents :prior :observe :belief-Q :update-belief
  :update-belief-tensor :act :expected-utility :params}` (all ten keys are
  returned; `:world-agents` is the per-world MDP-agent map and
  `:update-belief-tensor` the opt-in `[W]`-MLX filter).
- **Belief** is a plain `{world -> prob}` map (always; both host and tensor filters
  return a map).
- `:act` — **belief-based**: `(act belief s)` → action int (softmax over the QMDP
  belief-Q).
- `:belief-Q` — `(belief s)` → `[A]` MLX row. `:expected-utility` — `(belief s a)` → float.
- `:update-belief` — `(belief loc obs)` → belief'. **Observation contract**: `obs = nil`
  is the identity (absence is non-informative); an impossible observation leaves the
  belief unchanged (defensive, no NaN). `:update-belief-tensor` runs the same algebra
  as pure `[W]` MLX ops and returns the same map (opt-in via `simulate-pomdp`
  `:belief-mode :tensor`).

### `pomdp/make-bandit-agent {:keys [strategy alpha]}` — multi-armed bandit
- **Returns** `{:act :update-belief :arm-values :params}` — **no `:policy`** (Thompson
  posterior-sampling / softmax-of-means, not a GF over a fixed action set).
- **Belief** is `{:arms [[alpha beta] ...]}` (per-arm Beta).
- `:act` — **belief-based**: `(act belief key)` → arm int. `:arm-values` — `(belief)` →
  vector of posterior means. `:update-belief` — `(belief arm reward)` → conjugate
  Beta increment (success → α+1, failure → β+1; other arms unchanged).

## Summary of the family-specific differences (the honest contract)

| family  | `:act` signature   | `:policy` (GF) | `:Q`/`:V` | belief surface                  |
|---------|--------------------|----------------|-----------|---------------------------------|
| MDP     | `(s)` / `(s key)`  | yes            | yes       | —                               |
| biased  | `(s)` / `(s key)`  | yes            | **no**    | —                               |
| POMDP   | `(belief s)`       | — (hoisted)    | per-world | `{world->prob}`, `:update-belief` |
| bandit  | `(belief key)`     | **no**         | —         | `{:arms [[a b]...]}`, `:update-belief` |

A future uniform agent protocol could normalize `:act` (e.g. a single
`(act agent state-or-belief & [key])`), but that is an API *change* (owner decision),
not part of this freeze. This document fixes the surface **as built**.

## NOT frozen — `genmlx.agents.remote` is PROVISIONAL

`genmlx.agents.remote` (the glue for the FIRST EXTERNAL ENVIRONMENT, ROADMAP Phase 3
item 5 — Gym transport, env-server handlers, and remote rollouts over the
`genmlx.world.net` membrane) is **explicitly OUTSIDE this frozen surface**. Its
signatures may change; it is **not pinned by a contracts test** (only by the
behavioural/parity self-checks in `examples/external_env.cljs` +
`test/genmlx/external_env_test.cljs`). Do not treat anything in `genmlx.agents.remote`
as a stable v1.0 contract. (The membrane it sits on, `genmlx.world.net`, lives in the
`genmlx.world.*` membrane tree, mirroring `genmlx.mlx` — not the agents surface.)
