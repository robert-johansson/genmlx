# genmlx.control — v1.0 contracts

`control = agents pointed at computation`. The metareasoner is an agent whose
environment is the inference process itself; its actuators are scheduling
decisions. One principle, projected onto the computational environment.

These are the v1.0-frozen shapes (extensions are additive).

## One-way dependency (enforced)

```
core -> { llm, agents } -> control
```

`genmlx.control.*` may `:require` `genmlx.agents.*` (it reuses
`agents.helpers/softmax-action` as the meta-policy). **No `genmlx.agents.*` file
may `:require` `genmlx.control`** — verified by a grep guard. Keeping them
separate is what lets agents stay a pure planning-as-inference vertical and
confines the one side effect (the scheduler) to control.

## The scheduler is the SOLE side effect, and is NEVER a generative function

The control-layer `eval!` is `genmlx.world.proc/with-deadline` (the
process/scheduler face of the Bun world membrane). The metareasoner is pure; only
the scheduler touches the clock and the GC. The controller realizes its policy by
**gating a base steppable's `done?`** with the VOC stop, so `proc` drives it
directly — the wall-clock budget is the hard cap, the VOC is the resource-rational
stop. Do not make the scheduler a GF (the dishonest-shim anti-pattern).

## decision-value is DOWNSTREAM, never a sampler diagnostic

`genmlx.control.decision-value`: the reward at `:stop` is a real downstream
quantity — `neg-bayes-risk` (negative posterior variance under squared-error
loss) or `max-eu` (Bayes-optimal expected utility over an action set). It is
**NEVER ESS or log-ML**. `assert-downstream!` throws if handed a `peek` map.
Optimizing a sampler diagnostic is the classic metareasoning trap (spending
compute to make the sampler *look* healthy instead of making the *decision*
better).

## Meta-MDP surface (v1.0, MYOPIC + SMC/SMCP3-only)

`make-metareasoner` opts: `{:alpha :lambda :latent-addr :decision-value-fn
:cost-key :hysteresis}` — returns `{:params :policy :act :decision-value
:control}`, mirroring the `agents` `make-*-agent` return shape.

- `:policy` is a generative function over the meta-action (`p/simulate` works on
  it): `action ~ Categorical(softmax(alpha * EU))`. `alpha = ##Inf` is the
  deterministic meta-greedy argmax (the Callaway-2018 baseline, free).
- Action set: `[:continue :stop]`. `:add-particle` / `:refine` need a
  particle-growth substrate (the steppable advances per observation); they are
  deferred. `:switch-method` (live SMC<->MCMC translation) is deferred —
  `switch-method-translate` throws `:not-implemented`.
- Stop rule: blinkered one-step VOC + **hysteresis** (require `:hysteresis`
  consecutive stop-leans, default 3, so a single noisy down-estimate cannot stop
  an improving run — Russell & Wefald 1991 meta-greedy, hardened against MC
  noise) + the hard wall-clock cap (proc's deadline).
- Substrate: SMC/SMCP3-only for v1.0. The `peek` payload is not portable across
  substrates (accept-rate vs weights/ESS/log-ML), and VI is a genuine interface
  mismatch.

Anchors: Russell & Wefald 1991 (value of computation); Callaway 2018 (meta-greedy
baseline). Paper-side: topml-heic (uniform realization — scoped to SMC).
