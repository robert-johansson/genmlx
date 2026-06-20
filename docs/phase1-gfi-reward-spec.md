# Phase 1 Spec — GRPO where the REWARD is a GFI quantity (`genmlx-ugkv`)

> **Bean:** `genmlx-ugkv` (Phase 1 of milestone `genmlx-nv1t`). **Blocked-by:** `genmlx-zftr` (Phase 0, DONE).
> **Status:** SPEC — presented for review per the milestone protocol. **No implementation until approved.**
> **Date:** 2026-06-20. Built ON the verified Phase-0 `genmlx.world.train` boundary.

## 1. The idea (the highest-leverage novelty)

GenMLX already computes exactly the scalars policy-gradient RL wants as a reward. An LLM (the policy)
writes a **probabilistic program**; GenMLX evaluates it and returns **Bayesian model evidence** — the
marginal log-likelihood `log p(data | program)` — and GRPO uses that as the reward to update the policy
weights. This is, as far as we know, the **first system where Bayesian model evidence is the RL signal
for an LLM** — the *wake phase* of wake/sleep, operating at the level of program space.

The whole of Phase 1 is a **pure CLJS reward function** `(fn [prompt completion] -> number)` plugged into
the Phase-0 `train-step!` bridge. **No new Rust, no new autograd, no change to `world.train`.** The
reward-fn: parse the completion → SCI-eval the program → wrap as a GenMLX GF → score it → return the
scalar. The estimator family is the same REINFORCE-with-baseline already in `inference/adev.cljs` (GRPO's
group-mean baseline = the variance-reduction baseline).

## 2. Where it lands & dependencies

- **New ns:** `src/genmlx/world/train_reward.cljs` (Layer 9-ish, an LLM/training composition) — builds
  the reward-fn(s). It depends on `genmlx.llm.msa` (`score-model`), `genmlx.llm.codegen`
  (`verify-transition-fn`, `score-structure`), `genmlx.inspect` (compilation level), and the SCI eval
  helpers already in `msa.cljs`/`codegen.cljs`. It does NOT depend on `world.train` — it just produces a
  reward-fn that the caller passes to `train/train-step!`.
- **No new native dep.** Phase 0 already wraps the GRPO engine. Phase 1 is pure CLJS on top.
- **Real model required.** Unlike Phase 0's tiny random checkpoint, Phase 1 needs a model that emits
  *parseable programs*, so the reward signal is meaningful. Default: `qwen3.5-0.8b-mlx-bf16` (loads via
  the o94r loader; small enough for a single-box fine-tune). `qwen3.5-4b` is a config swap.

## 3. The reward functions (the GFI quantities)

Two reward families, each a pure `(fn [prompt completion] -> number)`. Both already return number-shaped
scalars and **neither forward-passes the policy model** (respecting the §7 model-ownership constraint):

### 3a. Bayesian model evidence (the headline) — `model-evidence-reward`
```
(prompt, completion) ->
  extract program text  →  SCI eval to a DynamicGF  →  msa/score-model gf observations
  →  log-ML  (+ λc · compilation-bonus  − λk · complexity)   [floored on failure]
```
- `observations` comes from the task (a fixed dataset the program must explain) — see §4.
- `msa/score-model` returns exact analytical marginal evidence for conjugate/eliminable models, else
  log-mean-exp IS (`msa.cljs:495`). Returns `##-Inf` on a nil/erroring GF.
- **Compilation bonus** (`λc · level`): `inspect/inspect` reports L1–L4; reward a program that compiles
  to a higher level (a structural prior toward GenMLX-idiomatic models). Optional, off by default.
- **Complexity penalty** (`λk · size`): subtract a small multiple of the program's form-size (Occam
  pressure). Optional.

### 3b. Program correctness + idiomaticity — `transition-fn-reward`
```
(prompt, completion) ->
  extract code  →  codegen/verify-transition-fn code transitions  →  :accuracy ∈ [0,1]
  (+ λs · normalized score-structure)                                [floored on failure]
```
- `verify-transition-fn` runs the generated fn against held transitions (`codegen.cljs:304`);
  `score-structure` rewards idiomatic ClojureScript (`codegen.cljs:455`).

### Invalid-program floor (critical for GRPO)
A parse/eval failure must map to a **finite floor** (e.g. `reward-floor = -100.0`), NOT `##-Inf` — GRPO
normalizes advantages by group std, and a `-Inf` in the group poisons the whole step (NaN advantages →
skipped step, the Phase-0 lesson). `score-model`'s `##-Inf` and `verify`'s `0.0`-on-error are clamped to
the floor by the reward-fn wrapper.

## 4. The task (what the policy generates, what it's scored against)

Phase 1 ships ONE task to prove the loop, parameterized so more can be added:
- **A task is `{:prompt-template, :observations, :system-prompt}`** (evidence reward) or
  `{:prompt-template, :transitions}` (correctness reward).
- **Model-synthesis task (default):** a small regression/mixture dataset; the prompt asks the LLM to
  "write a GenMLX `gen` model that explains this data"; reward = `score-model` log-ML against the held
  `observations`. The dataset is fixed CLJS data in the test (no I/O).
- The policy sees the prompt; the engine samples `group-size` completions; each is scored; group-relative
  advantages drive the update.

## 5. The training loop (composition, not new mechanism)

```clojure
(train/with-trainer model {:learning-rate 1e-5 :group-size 8 :max-completion-length 256
                           :temperature 0.8 :kl-coef 0.0}        ; KL-free (see §6)
  (fn [trainer]
    (p/loop [step 0, history []]
      (if (= step N)
        (p/resolved history)
        (p/let [r (train/train-step! trainer (batch-of task) (model-evidence-reward task))]
          (log-progress step r)
          (p/recur (inc step) (conj history (:reward-mean r))))))))
```
The loop is plain promesa over the Phase-0 effect. The **proof** is that `:reward-mean` (mean log-ML)
**trends up** over steps — the policy learns to write better-fitting programs.

## 6. Edge cases & invariants

- **KL is KL-free in Phase 1.** Native autograd ERRORS on `klCoef > 0` (`grpo/autograd.rs:75`, verified
  in Phase 0). So `:kl-coef` stays `0.0`. Wiring reference-model log-probs through autograd (true
  KL-to-base regularization) is **Phase 1.5** (a native change + clean upstream PR), explicitly deferred.
- **Reward purity / model ownership.** The reward-fn must be pure and must NOT run a forward pass on the
  *policy* model (the training thread owns it). Both reward families satisfy this: `score-model` runs
  GenMLX inference over the *generated* program (a fresh GF), `verify` runs the generated fn — neither
  touches the policy LLM. (If a future reward needs LLM scoring, it must use a *different* model handle.)
- **SCI sandbox.** Generated code is eval'd via the existing `msa.cljs`/`codegen.cljs` SCI evaluators
  (same sandbox the MSA/codegen paths already trust); eval errors → reward floor, never a crash.
- **Determinism.** The engine owns generation RNG (`config :seed` does not exist — Phase 0 finding); runs
  are not bit-reproducible across the native sampler. The reward-fn itself is deterministic given a
  completion.
- **Cost.** A real-model GRPO step (generate `group-size` × `max-completion-length` tokens + backprop on
  0.8B) is heavyweight; Phase 1's acceptance run is a SMALL number of steps on the small model, serial
  (no parallel GPU — the Metal-wedge constraint).

## 7. Done means

- [ ] `world/train_reward.cljs` with `model-evidence-reward` (and `transition-fn-reward`) — pure
      `(fn [prompt completion] -> number)` closures, with the invalid-program **finite floor**.
- [ ] A reward-fn, passed to the Phase-0 `train/train-step!`, drives a real `qwen3.5-0.8b` GRPO step whose
      reward is `msa/score-model` log-ML — the loop closes with **no new Rust**.
- [ ] **Reward improves:** over N steps, `:reward-mean` (mean log-ML) trends upward on the task — the
      policy demonstrably learns (the Phase-1 claim). Asserted with a tolerance, not a single noisy step.
- [ ] Invalid/un-parseable completions score the finite floor (no `##-Inf` poisoning the group).
- [ ] Reward purity test: the reward-fn does not touch the policy model handle; it is a pure
      `(prompt, completion) -> number`.
- [ ] KL-free documented; `:kl-coef 0.0`; Phase-1.5 (ref-logprob autograd) captured as a follow-up bean.
- [ ] Docs: the "Bayesian model evidence as RL signal = wake-phase" framing + the reward-shaping knobs.

## 8. Open decisions (for review)

1. **Reward signal for the Phase-1 proof:** (a) Bayesian model evidence (`score-model` log-ML) — the
   crown novelty; (b) program correctness (`verify` accuracy + `score-structure`); (c) ship both.
   **Recommendation: (a)** as the headline, with (b) implemented but exercised only lightly.
2. **Base model:** `qwen3.5-0.8b-mlx-bf16` (cheap, the point is the loop + reward trend) vs `qwen3.5-4b`
   (better programs, heavier). **Recommendation: 0.8b** for the proof; 4b is a one-line config swap.
3. **Reward shaping:** ship plain log-ML first, or include the compilation-bonus / complexity-penalty
   knobs from day one? **Recommendation:** ship plain log-ML + the finite floor; add λc/λk as
   off-by-default knobs (so the shaping is available but the proof is clean).
4. **Acceptance bar for "reward improves":** a strict monotonic-ish trend over N steps can be noisy on a
   0.8B model. **Recommendation:** assert `mean(last k) > mean(first k)` over a modest N, plus the loop
   closing + finite-floor + purity tests as the hard gates (the *capability* is the deliverable; the
   *magnitude* of improvement is model/compute-bound).

---

**Per the milestone protocol, this spec is presented for review. No code will be written until it is
approved.** The natural review questions are the four open decisions in §8.
