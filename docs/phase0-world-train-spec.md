# Phase 0 Spec — `genmlx.world.train` (the training `eval!` boundary)

> **Bean:** `genmlx-zftr` (Phase 0 of milestone `genmlx-nv1t`). **Blocked-by:** `genmlx-o94r`.
> **Status:** SPEC — presented for review per the milestone protocol. **No implementation until approved.**
> **Date:** 2026-06-20.

## 1. Scope & goal

`genmlx.world.train` is a new **Layer-0 membrane face** — a sibling to `genmlx.mlx` (compute),
`genmlx.world.net` (network), `genmlx.world.proc` (scheduler). It binds mlx-node's native training
engines (`GRPOEngine`, later `SftEngine`) behind a mutable-state quarantine, exposing **one new
effect: a training step that mutates model weights + optimizer state in place.**

This is **RL's `eval!`-equivalent — the sole training side effect.** Everything above it stays pure:
the reward is a pure `(trace → scalar)`, the rollout population is pure steppable inference, the
policy is a pure GF. Phase 0 ships the *boundary + plumbing*; Phase 1 (`genmlx-ugkv`) wires GFI-quantity
rewards through it.

Non-goals for Phase 0: the GFI reward scorers themselves (Phase 1), VOC rollout allocation (Phase 2),
multi-GPU, the server. Phase 0 ships a trivial reward (length) to prove the round-trip end-to-end.

## 2. Where it lands & dependencies

- **Namespace:** `src/genmlx/world/train.cljs` (Layer 0). Loads its native binding lazily, like the
  other faces; `available?` guards presence so non-training code/tests don't require it.
- **Native dep: NONE new.** *Verified 2026-06-20:* the already-built `@mlx-node/core` addon GenMLX
  already loads exposes the full training surface directly — `GrpoTrainingEngine`
  (`fromQwen35`/`fromQwen35Moe`, `trainStep`, `trainStepAuto`), `SftTrainingEngine`,
  `NativeRewardRegistry`, `buildRewardOutputs` (6 training classes among 224 exports). So Phase 0 binds
  `@mlx-node/core` directly. `@mlx-node/trl` is **not built** (`dist/` has no `index.js`) and is **not
  needed** — this resolves §9.1.
- **Gated by `genmlx-o94r` (sharded safetensors):** real training loads >3GB checkpoints, most sharded;
  the loader must support `model.safetensors.index.json`. Phase 0 membrane plumbing can be built +
  tested against a tiny model; *real* fine-tuning is gated by `o94r`.
- **Interaction with `genmlx-nldo`:** once the superset addon lands, the GRPO surface is reached through
  `@genmlx/core` like everything else; until then, through `@mlx-node/trl`/`@mlx-node/core`.

## 3. The mutable boundary (the quarantine)

The native `GRPOEngine` mutates the model's weights and the AdamW optimizer's moment state **in place**
on each step. That in-place mutation is the effect. It is quarantined exactly like the two existing
precedents:

- `world.net`'s `with-server` (the `Bun.serve` listener — a live external resource, torn down by
  `p/finally`).
- `backend.cljs`'s KV-cache atom (`CljsForwardModel.cache`, reset in `init-cache!`/`reset-cache!`,
  swapped inside a `try`).

**Invariant:** the engine handle is the *one* mutable resource of this face; it is created inside a
blessed scope, consumed locally, and torn down on exit (success or throw). It never escapes into pure
code. The trained model's *weights* are observable state owned by the caller (like the KV cache), not
hidden global state.

## 4. API surface (every public fn)

```clojure
(available?)                       ;; => bool. Native GRPO/TRL present? (guards the require)

;; Construction — blessed scope + low-level escape hatch (mirrors with-server/serve!)
(with-trainer model config f)      ;; build engine over a loaded model, call (f trainer),
                                   ;;   p/finally tears the engine down. THE blessed path.
(make-trainer! model config)       ;; => trainer. Low-level; caller owns teardown via (dispose! trainer).

;; The one effect — a GRPO step
(train-step! trainer prompts reward-fn opts)
  ;; prompts    : vector of prompt strings (or chat-message vectors)
  ;; reward-fn  : pure CLJS (fn [prompt completion] -> number)   [per-completion], OR
  ;;              (fn [prompts completions] -> number[])          [batched]
  ;; opts       : {:group-size N :loss {...} :timeout-ms ...}
  ;; => promise of {:loss :reward-mean :reward-std :kl :grad-norm :completions [...]}
  ;; Drives native train_step_auto via a ThreadsafeFunction reward callback (see §5).

;; Lifecycle passthroughs (native GRPOEngine methods)
(step trainer) (epoch trainer)
(start-epoch! trainer) (end-epoch! trainer secs)
(reset! trainer)
(generate-batch trainer prompts)   ;; for inspection; no weight update

;; Checkpointing (native resumable AdamW — moments + weights)
(save-checkpoint! trainer path)
(load-checkpoint! trainer path)
(dispose! trainer)                 ;; explicit teardown for the make-trainer! path
```

`config` (→ native `GRPOEngineConfig`): `{:lr :group-size :beta :max-completion-len :temperature
:loss-type (:grpo|:bnpo|:dr-grpo|:dapo) :seed ...}`.

## 5. The reward bridge (the seam to Phase 1)

The native step takes `reward_fn: ThreadsafeFunction<String, Promise<Vec<f64>>>` (`engine.rs:911`),
surfaced to JS (verified) as **`trainStepAuto(prompts, rewardFn, recordOutputs)`** where
`rewardFn: (err: Error | null, outputsJson: string) => Promise<number[]>` (node-style `(err, value)`).
`train-step!` marshals the **pure CLJS reward-fn** into that callback:

1. native generates a rollout group → invokes the ThreadsafeFunction with each completion (JSON/string),
2. the CLJS reward-fn is called (synchronously, on the callback) → returns a number,
3. numbers flow back as `Vec<f64>` → native computes group-relative advantages → clipped surrogate →
   AdamW update.

**Phase 0 reward:** a trivial pure fn (e.g. completion length, or a `RewardRegistry` built-in via
`register_builtin_reward`) — enough to prove the round-trip mutates weights.

**Phase 1 reward (`genmlx-ugkv`):** the same seam, reward-fn = a GFI quantity:
`msa/score-model` exact-or-IS log-evidence (`msa.cljs:495`) and/or `codegen/verify-transition-fn`
accuracy + `score-structure` idiomaticity (`codegen.cljs:304,455`) — both already return number-shaped
values. **No new Rust; the bridge is reused verbatim.**

## 6. Coverage matrix

Move the training-orchestration exports (`GRPOEngine`/`GRPOTrainer`, `trainStepAuto`,
`RewardRegistry`, native optimizers, `OutputStore`) from the **`:training-orchestration` intentional-
omission** group → **wrapped**, in `docs/membrane-coverage.md` + `test/genmlx/membrane_coverage_test.cljs`.
This is the surface-drift guard recognizing the newly-tapped stack.

## 7. Purity invariants (and the test)

- Training is the **sole side effect** of this face (the `eval!`-equivalent). Add `world.train`'s engine
  handle to the audited mutable-boundary list in `test/genmlx/mutation_boundary_test.cljs`.
- The reward fn passed to `train-step!` must be **pure** `(prompt, completion) → number` — it must NOT
  mutate, and must NOT run a forward pass on the *model being trained* (mirror the single-execution
  constraint at `core.cljs:43-45`; the training thread owns that model). A reward that needs LLM scoring
  must use a *different* model handle.
- `train-step!` returns a promesa **promise** — training is an *event*, so it crosses to async per the
  project's "sync math, async events" principle (like model load / `.chat`). The pure GFI math above
  stays sync.

## 8. Edge cases & invariants

- **KL-to-ref gap:** native `grpo/autograd.rs:76` rejects `beta > 0` under autograd → Phase-0/1 runs are
  **KL-free** unless `grpo_loss` is driven directly. `config :beta` is accepted but documented as
  no-op-under-autograd until the ref-logprob autograd path lands (a Phase-1.5 follow-up + upstream PR).
- **Determinism:** the engine owns its sampler (MLX RNG); this is a *boundary* — GenMLX's
  `rng/fresh-key` does not thread into native generation. `config :seed` controls native reproducibility;
  document that training RNG ≠ GenMLX inference RNG.
- **Teardown:** `with-trainer`'s `p/finally` disposes the engine (frees model + optimizer moments) on
  success or throw. The `make-trainer!` escape hatch hands that responsibility to the caller.
- **Model ownership:** the engine borrows the loaded model (`from_qwen35(&model)`); the model handle must
  outlive the trainer. Document; `with-trainer` should take a loaded model and not free it.
- **Timeout:** the reward callback can hang (CLJS scorer); `:timeout-ms` bounds it (native
  `withTimeout`, `grpo-trainer.ts:1280`).

## 9. Open decisions (for review)

1. ~~Bind `@mlx-node/trl` or `@mlx-node/core` directly?~~ **RESOLVED (2026-06-20):** bind
   `@mlx-node/core` directly — `GrpoTrainingEngine`/`trainStepAuto` are already exported by the addon
   GenMLX loads; `@mlx-node/trl` is unbuilt and unnecessary. No new dependency.
2. **Async shape:** confirm `train-step!` returns a promesa promise (vs a blocking deref helper for REPL
   ergonomics). Recommendation: promise + a `train-step!!` blocking convenience for the REPL.
3. **Quarantine record:** a `defrecord Trainer [engine model ^:mutable disposed?]` vs a bare JS handle in
   an atom. Recommendation: a record mirroring `CljsForwardModel`.

## 10. Done means

- [ ] `genmlx.world.train` ns with `available?`, `with-trainer`, `make-trainer!`/`dispose!`,
      `train-step!`, lifecycle passthroughs, checkpoint save/load.
- [ ] The reward bridge marshals a pure CLJS reward-fn into the native `ThreadsafeFunction`; a Phase-0
      length-reward `train-step!` measurably changes a (tiny) model's weights.
- [ ] `available?` true on this box (binds `GrpoTrainingEngine` from the existing `@mlx-node/core`; no new dep).
- [ ] Coverage matrix updated (training-orchestration omitted → wrapped); `membrane_coverage_test` green.
- [ ] `mutation_boundary_test` updated with the new audited boundary; purity property holds.
- [ ] `with-trainer` teardown verified (engine disposed on throw).
- [ ] Docs: the reward-bridge seam documented as Phase-1's plug point.
- [ ] **Gating acknowledged:** real-checkpoint training deferred to `genmlx-o94r`; KL-free documented.

---

**Per the milestone protocol, this spec is presented for review. No code will be written until it is
approved.** The natural review questions are the three open decisions in §9.

---

## 11. Spec corrections after native-surface verification (2026-06-20, IMPLEMENTED)

The spec above was written from inspection. Before writing the membrane, the actual `@genmlx/core`
training surface was verified against the typings + reflection, which forced the following corrections.
The implementation (`src/genmlx/world/train.cljs`, `test/genmlx/world_train_test.cljs`) is built against
the **verified** surface, not the inspected one. All Phase-0 acceptance assertions pass (25/25 on Metal).

**Module + naming (factual fixes):**
1. **Bind `@genmlx/core`, not `@mlx-node/core`** (§2/§9.1 named the wrong module). GenMLX loads
   `@genmlx/core` (mlx.cljs:27); both expose an identical training surface but they are separate addons.
   `train.cljs` requires `@genmlx/core` for consistency.
2. **Class is `GrpoTrainingEngine`** (not `GRPOEngine`); config is `GrpoEngineConfig` (camelCase);
   factories `fromQwen35`/`fromQwen35Moe`, ctor `(Qwen3Model, config)`.
3. **`step`/`epoch` are getters**, `startEpoch()` is no-arg, `endEpoch(secs)` takes seconds.

**Capability gaps (forced API changes):**
4. **No native `saveCheckpoint`/`loadCheckpoint`.** The engine persists only OPTIMIZER moments via
   `saveOptimizerState`/`loadOptimizerState`; model WEIGHTS are saved on the model handle (`saveModel`).
   So the face exposes honest `save-optimizer-state!`/`load-optimizer-state!` (not a lying "checkpoint").
5. **No native `dispose`.** `dispose!` is implemented in CLJS via the engine's terminal `reset()` — which
   releases the model-thread training run so the model can host a NEW trainer — plus marking the handle
   disposed. (The native model thread hosts only ONE active training run; freeing it on teardown is
   required for `with-trainer` re-entry.)
6. **No `:seed` in `GrpoEngineConfig`.** §8's "config :seed controls reproducibility" is unsupported;
   the engine owns its MLX sampler RNG (training RNG ≠ GenMLX inference RNG). `:seed` is dropped.
7. **`:beta` → `klCoef`.** Mapped explicitly. NOTE the §8 framing "accepted but no-op-under-autograd"
   is wrong: native autograd **errors** on `klCoef > 0` (`grpo/autograd.rs:75` returns
   `Err("KL penalty (beta > 0) requires reference model logprobs ... not yet supported")`). So a
   `train-step!` with `:kl-coef`/`:beta` > 0 **rejects**; leave it 0 (default) for KL-free training.
   `world_train_test` asserts this rejection.
8. **`registerBuiltinReward` is an instance method** (not a standalone export); `BuiltinRewardType`
   values are the strings `Length`/`ToolUse`/`XmlFormat`/`JsonSchema`.

**Reward bridge (design improvement, §5):** the spec's all-in-one `trainStepAuto` ThreadsafeFunction
callback is **not** used. `trainStepAuto` filters out `finish_reason='length'` completions as an
OOM guard (`engine.rs:1067`), which skips the entire step for any model whose rollouts never emit EOS
(e.g. a tiny random checkpoint). Instead `train-step!` composes the explicit
**`generateBatchForTraining` → score-in-CLJS → `trainStepWithGenerations`** flow: it keeps the pure
reward scorer in CLJS between two awaits (a cleaner Phase-1 GFI-reward seam, no native callback
marshalling) and trains UNCONDITIONALLY. The pure reward-fn is `(prompt, completion-text) -> number`,
mapped over completions in prompt-major order.

**Teardown primitive (§3/§4):** `with-trainer` uses **`p/handle`**, not `p/finally`. Under nbb a
`p/finally` teardown followed by a downstream `p/catch` double-settles (the catch handler runs yet the
promise stays rejected — verified). `p/handle` disposes on both arms and re-raises exactly once.
(`world/net.cljs`'s `with-server` shares the latent `p/finally` quirk — captured as a follow-up.)

**Quarantine record (§9.3):** `defrecord Trainer [engine model state]` where `state` is an `atom`
(CLJS `defrecord` has no `^:mutable`; the atom mirrors `CljsForwardModel`'s cache atom).

**Phase-0 test vehicle:** a tiny random Qwen3.5 checkpoint (`createRandomQwen35Checkpoint`, 2 layers /
hidden 64) with the **real Qwen3.5 vocab (248320) + a real tokenizer copied in** (generation needs a
tokenizer; the random checkpoint writes none). The length/content reward must give the group nonzero
variance (a constant reward ⇒ zero advantage ⇒ no update).
