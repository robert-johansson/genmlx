# Phase 1.5 — KL-to-base penalty through GRPO autograd (genmlx-65d5)

> **Status:** SPEC — awaiting approval. No code until approved (milestone protocol).
> **Bean:** `genmlx-65d5` (child of `genmlx-nv1t`). Builds on Phase 1 (`genmlx-ugkv`),
> which trains KL-free because the native autograd path rejects `klCoef > 0`.
> **Date:** 2026-06-21. Every file:line below verified against source.

## Done-means (from the bean)

1. `klCoef > 0` trains **without error** under autograd.
2. A KL term **measurably regularizes** the policy toward the base policy.
3. The `world.train` kl-coef test **flips from "rejected" to "applied."**

## Current state (verified)

- **The KL math already exists and is correct.** `grpo::loss::grpo_loss`
  (`grpo/loss.rs:120`, last arg `ref_per_token_logps: Option<&MxArray>`) implements
  the k3 estimator when `beta > 0` (`loss.rs:248-267`):
  `KL(ref‖new) = exp(ref−new) − (ref−new) − 1`, log-ratio clamped to `[-88,88]`,
  scaled by `beta`, added to the per-token loss. Shape of `ref_per_token_logps` is
  `[B,T]` — identical to `per_token_logps`.
- **The autograd path refuses `beta > 0`.** `compute_loss_and_gradients_autograd`
  (`grpo/autograd.rs:64`) early-returns an error at `autograd.rs:76-81`, and both
  loss closures call `grpo_loss(..., None /* no reference model */)`
  (`autograd.rs:328-329` standard, `autograd.rs:521` chunked). So the **only** thing
  missing is producing `ref_per_token_logps` and passing it in.
- **No reference/frozen-base model exists today.** `engine.rs` has no
  reference/ref-logprob/frozen-param machinery (`grpo/engine.rs`, 2027 lines).
- **Training orchestration is per-model.** `compute_loss_and_gradients_autograd` is
  called only from `models/{qwen3,qwen3_5,qwen3_5_moe}/model.rs`
  (`train_step_grpo_sync`, e.g. `models/qwen3/model.rs:5092` → call at `:5163`).
  `old_logprobs` = `ts.cached_completion_logprobs` — the logprobs cached **at
  generation time** (`models/qwen3/model.rs:4781,5117`), i.e. the **old policy**
  (the policy that produced the rollout), **not** the reference.
- **`beta` is wired from config at two engine sites** (`engine.rs:1188`, `:1777`),
  both building `GRPOLossConfig { beta: self.config.kl_coef.unwrap_or(0.0), .. }`,
  both flowing into `train_step_grpo_sync` → autograd. One fix covers both.
- **Param updates are functional.** `OptimizerImpl::apply_gradient(&mut self,
  param: &MxArray, grad: &MxArray) -> Result<MxArray>` (`optimizers/mod.rs:26`)
  **returns a new array**; the model replaces the layer weight with it. Weights are
  **replaced, not buffer-mutated** ⇒ an Arc-clone snapshot of the initial params is a
  valid frozen reference (no deep copy).
- **CLJS surface.** `world/train.cljs:106` documents the honest contract
  ("klCoef > 0 ERRORs under autograd"); `world/train.cljs:116-117` maps
  `:kl-coef`/`:beta` → `klCoef`. The test `world_train_test.cljs:174-181` asserts
  `kl-coef>0` ⇒ `:rejected`.

## Design

### Reference = frozen snapshot of the initial policy params

Standard GRPO/TRL: `π_ref` is the model **before** RL. GRPO starts from the
base/SFT checkpoint, so the **initial params ARE the reference** — no separate
checkpoint, no extra config.

- New field on each model's `TrainingState`:
  `reference_params: Option<HashMap<String, MxArray>>`.
- **Populated lazily on the first train step where `beta > 0`** — an **Arc-clone**
  of `get_parameters_sync()` (cheap; keeps the initial buffers resident). When
  `beta == 0` (the default) it is never populated ⇒ **zero cost when KL is off**.
- Lifecycle: persists across steps; dropped on `reset()`/dispose (alongside the
  existing training-state teardown). Survives in-place optimizer updates because
  updates replace weights, not the snapshot's buffers.
- **Memory:** `+1× params` (frozen reference, e.g. ~1.6 GB for 0.8B bf16), paid
  **only when `klCoef > 0`**.

### Reference logprobs (per step, no-grad)

Computed inside `compute_loss_and_gradients_autograd` when `beta > 0`:

1. Reuse the already-built `input_ids` (padded prompt+completion), `prompt_len`,
   `padded_completions`, `completion_masks`.
2. Forward through **reference_params** (reuse `functional::forward_functional_dispatch`
   / `functional::chunked_lm_head_selective_logprobs` — the same dispatch the loss
   closure uses) → completion logits → `efficient_selective_log_softmax` over
   `padded_completions` ⇒ `ref_per_token_logps` `[B,T]`.
3. `clip(Some(-20.0), Some(0.0))` (mirror the policy/old-logprob clamping at
   `autograd.rs:319,179`).
4. Computed **once, OUTSIDE `value_and_grad`** ⇒ a constant w.r.t. the trainable
   params (no gradient flows; `value_and_grad` differentiates only the `params`
   vector). `eval()` to materialize + detach before the closure captures it.
5. Pass `Some(&ref_per_token_logps)` into `grpo_loss` (standard closure
   `autograd.rs:322`; chunked closure `autograd.rs:514` — slice the ref array per
   chunk exactly like `padded_old_logprobs`).

The reference forward is **no-grad** (no backward graph) ⇒ much cheaper than the
policy forward; it honors `forward_chunk_size` / `lm_head_chunk_size` for the same
peak-memory bounds.

### Signature & call-site changes

- `compute_loss_and_gradients_autograd` gains
  `reference_params: Option<&HashMap<String, MxArray>>`; the chunked helper gains the
  precomputed `ref_per_token_logps: Option<&MxArray>` to slice per chunk.
- **Remove** the `beta > 0` rejection (`autograd.rs:76-81`). Replace with: if
  `beta > 0` and `reference_params` is `None`, return a clear error (defensive — the
  model layer guarantees it is set).
- `train_step_grpo_sync` (×3 models): when `loss_config.beta > 0`, ensure
  `ts.reference_params` is populated (snapshot on first use) and pass
  `Some(&ts.reference_params)`; otherwise pass `None`.

### CLJS + test

- `world/train.cljs`: update the `:106` contract note — `klCoef > 0` is now
  **applied** (true KL-to-base), no longer an error. No API change (`:kl-coef`/
  `:beta` mapping at `:116-117` stays).
- `world_train_test.cljs:174-181`: flip from asserting `:rejected` to asserting
  `klCoef > 0` **trains** (`gradients_applied = true`, no rejection) — done-means #3.

## Invariants & edge cases

- **`beta == 0` (default): byte-for-byte unchanged** — no snapshot, no ref forward,
  `grpo_loss(..., None)` as today. Perf/memory neutral.
- **No gradient into the reference** — ref logprobs are inputs (constants) like
  `old_logprobs`; assert via the autograd test that the ref forward params receive
  no gradient.
- **Shapes** — `ref_per_token_logps` is `[B,T]`, matches `per_token_logps`; padded
  over completions, masked by `completion_masks`.
- **`valid_indices` filtering** — the reference forward runs on the **same filtered
  completions** as the policy (build after filtering, consistent with `old_logprobs`
  at `models/qwen3/model.rs:5142-5161`).
- **Numerics** — ref logprobs clamped `[-20,0]`; k3 KL clamps log-ratio `[-88,88]`
  (already in `loss.rs:260`). NaN handling inherits the existing clip-replaces-NaN
  behavior.
- **Determinism** — ref logprobs are a logprob extraction (no sampling) ⇒
  deterministic given the completions.
- **Chunked autograd path** — `ref_per_token_logps` sliced per chunk (`start..end`)
  exactly like `padded_old_logprobs` (`autograd.rs:416`).
- **All three models** — mechanism is identical; MoE's reference forward routes
  through the MoE functional dispatch (heavier, but no-grad).

## Test plan

1. **Rust unit (`grpo/loss.rs`)** — `test_kl_coefficient_scaling` (`loss.rs:827`)
   already covers the KL term in isolation. Add an **autograd** test: `beta > 0`
   runs end-to-end with a frozen reference, ref logprobs threaded, and the KL term
   shifts loss/gradients vs `beta = 0`.
2. **KL-effect (done-means #2)** — train N steps at `kl-coef = 0` vs a large
   `kl-coef`; assert the large-KL run stays closer to the base policy (measured
   KL(ref‖policy) or mean param drift `‖θ−θ₀‖` is **smaller**, monotone in `beta`).
3. **`world_train_test.cljs`** — flip `:174-181` to assert `klCoef > 0` trains
   (`gradients_applied = true`).
4. **Regression** — `beta = 0` GRPO path unchanged (existing world.train /
   world_train_reward suites).
5. **Native rebuild gate** — after `cd mlx-node && yarn build:native`, run the
   membrane contract guards (`exact_test`, `gradient_fd_test`, `score_gradient_test`,
   `clip_contract_test`, `membrane_coverage_test`) per CLAUDE.md, since the native
   addon was rebuilt.

## Upstream PR

Self-contained mlx-node change (GRPO autograd gains reference-model KL — the TRL
parity feature). Clean PR to the mlx-node fork.

## Decisions (LOCKED 2026-06-21)

1. **Reference source** — ✅ **initial-params snapshot** (Arc-clone of the model at
   training start; standard GRPO, zero extra config, exact "KL-to-base").
2. **Model scope** — ✅ **all three** (qwen3 + qwen3_5 + qwen3_5_moe; consistent,
   cleanest upstream PR). MoE reference forward verified lightly (heavier + the MoE
   is guard-refused elsewhere).
3. **Freeze mechanism** — ✅ Arc-clone snapshot (evidence: `apply_gradient` returns
   a new array ⇒ replace-not-mutate). Deep copy only if a future in-place optimizer
   is added.
