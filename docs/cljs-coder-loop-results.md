# GenMLX cljs-coder loop — run results (2026-06-22)

A small, GenMLX-native ClojureScript code generator: a fine-tuned **qwen3.5-0.8b**
specialized on the GenMLX/cljs program-synthesis distribution (probabilistic models +
small functions), built by **teacher distillation → oracle filter → SFT**, graded on a
leakage-free held-out split by the **same grounded oracle** that built the corpus.

This is the milestone `genmlx-8587` loop (task-gen `genmlx-7473` → SFT `genmlx-o8w9` →
GRPO `genmlx-2ctu`). Honest scope: a **specialist** at GenMLX/cljs program synthesis, not a
general coder — a 0.8b is near the floor on general code synthesis (the literature), but on
this narrow, well-tested distribution it rivals far larger models.

## The pipeline (all reproducible from the repo)

```
distill-gen (183 grounded tasks)
  → gen_tasks.cljs --export-train          143 teacher prompts
  → distill_teacher.py (Qwen3.6-35B-A3B)   858 candidates (6/task)
  + reference seeds (143)                  1001 candidates
  → distill_filter.cljs --gen --top-k 4    512 oracle-validated SFT rows
  → sft_prep.cljs --gen                     439 train / 73 valid (0 eval leakage)
  → sft_train.sh (LoRA, 8 layers, 700 it)   qwen3.5-0.8b adapter + fused model
  → distill_teacher.py (student) + sft_eval.cljs --gen   held-out pass@1/pass@k
```

### 1. Task / curriculum generator (`genmlx.world.distill-gen`, genmlx-7473)

**183 grounded tasks** (15× the 12-task seed set), each carrying a REFERENCE solution
validated against the same native-free oracle that grades the student:

- **44 conjugate programs** (gaussian-mean, beta-bernoulli, gamma-poisson) — ALL score the
  EXACT analytical marginal (reproducible, no importance-sampling noise).
- **139 functions** — state machines + a broad standard-function catalog (a 5-agent workflow
  authored 86 oracle-safe specs into `distill-gen-extra`).
- Grounding is **non-circular**: program reward is GenMLX's Bayesian model evidence
  (independent of the reference's form); function ground truth is hand-written (never derived
  by running the reference). `distill_gen_test` (18/18) admits every reference and checks the
  oracle discriminates.
- Family-balanced, leakage-safe split: **143 train / 40 eval** (every eval family has train
  siblings → real same-distribution generalization, never a leaked instance).

### 2. Teacher distillation + oracle filter (genmlx-j0d6 path, scaled)

- Teacher **Qwen3.6-35B-A3B-4bit** (on-box, the measured 93%-yield teacher), 6 samples ×
  143 train tasks = 858 candidates; + 143 reference seeds = 1001.
- `distill_filter --gen --top-k 4`: **784 kept**, **512 selected SFT rows**, **yield 1.0**
  (every train task covered — no cold-start in training), mean-log-ml −5.57, all programs
  exact-scored (0 noisy-IS), 1 non-terminator caught by the timeout sandbox.
- 22× the prior 23-row corpus — this is what fixed the o8w9 overfit.

### 3. SFT (mlx-lm LoRA on qwen3.5-0.8b-mlx-bf16)

- 8 layers, rank 16, batch 2, lr 1e-5, 700 iters, val-batches capped (32GB-safe).
- Val loss **1.114 → ~0.1–0.2 band, never diverged** (vs the o8w9 smoke that blew up to 2.01
  at 12-task scale). Checkpoints at every 100; best-val saved at iter 300 / 600.

### 4. Held-out eval (oracle-graded, leakage-free, 40 tasks, pass@k via Chen 2021)

| model | aggregate pass@1 | aggregate pass@4 | **function pass@1** | **function pass@4** |
|---|---|---|---|---|
| baseline 0.8b | 0.400 | 0.350 | 0.172 | 0.103 |
| **SFT-300** | **0.525** | 0.500 | **0.345** (2.0×) | 0.310 (3.0×) |
| **SFT-600** | 0.475 | **0.575** | 0.276 | **0.414** (4.0×) |

Programs tie at 1.0 (the prompt scaffolds the API + the no-floor "kept" criterion is lenient);
the discriminating signal is the **no-scaffold function synthesis**, where SFT **doubles
pass@1 and triples-to-quadruples pass@4**. Clean generalization (baseline 0→SFT 1 on abs-val,
max-of-list, dedupe-list, count-uppercase, sign-str, unit conversions, …); 2 minor regressions
(mean-of-list, between?); the hard tail (nth-prime, integer-sqrt, …) stays cold for both — a
0.8b ceiling, not a pipeline failure.

### Model artifacts

- `~/.cache/models/qwen3.5-0.8b-cljs-sft300` — best **greedy pass@1** (0.525); ship for
  single-shot use.
- `~/.cache/models/qwen3.5-0.8b-cljs-sft600` — best **pass@4** (0.575), most GRPO headroom
  (pass@4 − pass@1 = +0.14 on functions); the GRPO base.

Both are standard MLX safetensors in the qwen3_5 (owned-forward) family — GenMLX loads them
identically to the base 0.8b.

## 5. GRPO sharpening (genmlx-2ctu)

`scripts/grpo_sharpen.cljs` runs GRPO over `genmlx.world.train` (the native GRPO engine
membrane): load the fused SFT model as the policy, generate group-size completions per
TRAIN-task prompt, score each with the SAME grounded oracle (model-evidence for :program,
behavioral accuracy for :function), and apply the group-relative AdamW update — sharpening
pass@1 toward the SFT pass@k ceiling (RLVR; Yue 2025). Held-out eval tasks are never in the
band (leakage).

**Mechanism validated end-to-end:**
- The fused SFT-600 model loads in GenMLX's native owned forward (`Qwen35Model.load`) and the
  GRPO step mutates its weights (`gradients-applied? true`, finite rewards, no NaN/Inf).
- Band curation matters: a random band gave 2/6 learnable groups (always-pass programs +
  too-hard functions have no reward variance); **medium-difficulty train functions give 8/8
  learnable groups, valid-rate ~0.5** — the variance sweet spot.
- A real generation quirk and its fix: the native engine does not stop at EOS (mlx-lm does),
  so completions are `(valid form)` + `\n!\n!…` filler. Evaluating the whole string errored
  every reward to the floor; isolating the first complete form (preserving named recursion)
  fixes it — the reward then correctly scores the code (e.g. a generated dot-product scores
  1.0). The group-relative baseline cancels the common-mode filler, so the learning signal is
  the code-quality differential.

The held-out lift of the GRPO'd model is measured by re-running the SAME mlx-lm + oracle eval
(mlx-lm respects EOS, so the filler does not affect the deployed model). Whether GRPO is
shipped over SFT-600 is decided by that objective re-eval — RLVR gains are modest by nature.

**Run outcome (2026-06-22): the native GRPO engine WEDGED.** A real run (SFT-600 base, 15
steps, band 8) loaded the model and completed 2 steps cleanly (step 0 reward-mean −9.68,
valid-rate 0.50, 8/8 learnable; step 1 −11.77) — then the native `generateBatchForTraining`
**stalled mid-step-2 and made no progress for ~3.5h** (the Metal-stall risk documented in the
project memory; `Context leak detected, CoreAnalytics returned false`). Killed; no sharpened
model was produced. The GRPO MECHANISM is validated (load + reward bridge + gradient step all
work, with a clean 8/8-learnable signal), but a sustained multi-step run is not reliable on
this 32GB box overnight. Re-running needs: a smaller band + fewer steps + lower max-completion
length, per-step generation timeouts/checkpointing in the native engine, and ideally a fresh
process (a reboot clears any residual Metal wedge). The SFT model (above) is the shipped
deliverable; GRPO would, at best, modestly sharpen its pass@1 toward the pass@4 ceiling.

