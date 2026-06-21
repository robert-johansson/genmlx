# Research memo: a small ClojureScript coder via SFT+GRPO, and Qwen3-Coder-Next as teacher

> Two parallel research workflows (web literature + local grounding), 2026-06-21.
> Question 1: can SFT + iterated GRPO (SCI oracle) make a small Qwen a strong cljs
> coder? Question 2: can Qwen3-Coder-Next run on 96GB + integrate with GenMLX (incl.
> as a teacher)? Written for an engineer who wants honesty, not optimism.

## Verdict (one paragraph)

A fine-tuned **0.8B will not become "almost perfect"** at ClojureScript — the
literature is consistent and the bottleneck is base capacity + data, which RL does
not manufacture. But **a meaningfully better, locally-useful small cljs coder is
realistic**, and the single highest-leverage move is **distillation from a strong
teacher** (Qwen3-Coder-Next), oracle-filtered through GenMLX's existing SCI/msa
validators, used as the SFT corpus — with GRPO as *amplification*, not the engine.
Qwen3-Coder-Next runs **comfortably on 96GB at 4-bit** and should be used as an
**offline teacher via Python mlx-lm**, not a native GenMLX forward (that path is the
deferred MoE-SIGTRAP, high-risk, unnecessary).

## Part 1 — the small-model SFT+GRPO loop

### The ceiling (sourced)
- Contamination-free **LiveCodeBench** pass@1, Qwen2.5-Coder-Instruct: 0.5B≈2.0,
  1.5B≈6.1, 3B≈10.8, 7B≈18.2, 32B≈31.4 (arXiv:2409.12186). Sub-2B is near the floor
  on *genuine novel* code. The 0.8B↔4B gap is large, not incremental.
- **HumanEval lies** for small fine-tunes: the same 0.5B scores 61.6 HumanEval vs
  2.0 LiveCodeBench; fine-tuned open small models are *the* overfitting cohort
  (LiveCodeBench, arXiv:2403.07974). This is almost certainly what burned prior cljs
  fine-tunes.
- **Lisp-family is hardest + lowest-resource.** Racket (best public Clojure proxy)
  is among the lowest MultiPL-E languages; Clojure is even lower-resource
  (arXiv:2208.08227).
- **SFT on a niche language works but the absolute ceiling stays low.** MultiPL-T
  (execution-validated synthetic data) lifted **Racket 4.7→11.3 pass@1 on a 1B**,
  11.8→21.0 on 15B (arXiv:2308.09895). Real gains, bounded ceiling.
- **Distillation/synthetic-data SFT > raw-corpus SFT** for code (NeMo KD; Case2Code
  arXiv:2407.12504). ~100K lines is small for continued-pretraining (studies use
  100M+ words); quality/diversity > raw line count.

### The RLVR/GRPO crux (the decisive part)
- **RLVR sharpens pass@1 but does not raise pass@k beyond the base** (Yue et al.
  2025, arXiv:2504.13837). GRPO re-weights what SFT already made reachable.
- **Cold-start kills it**: if the SFT'd base never samples valid/passing cljs for a
  task, all rollouts score 0 → group advantage 0 → gradient vanishes. **The loop's
  floor is the SFT'd base's pass@k, not RL.**
- **DeepSeek-R1's own finding**: RL on a small model loses to distillation (47% vs
  72.6% AIME). *Lead with SFT.*
- Real GRPO-on-small-coder successes exist (a **Lisp** GRPO run 0.57→0.88; Oxen
  Rust-1.5B build 61→80%) — but only because the base already partly worked.
  Realistic gain: **"polish a working base by ~15–25 pass@1 points."**
- Counter-evidence (ProRL, arXiv:2505.24864) shows RL *can* expand the boundary in
  low-competence domains — but needs prolonged training + KL control + reference
  resets + task diversity. Not free; mostly 7B+ on math.

### Iterated loops + reward design
- Verifier-bounded SFT↔RL loops **plateau fast**; ReST-EM code gains come at
  iteration 1 then *regress* at iteration 2 when the task set is small. Cap at ~3–6
  rounds, **grow the task set each round**, watch for entropy collapse.
- **SCI oracle = 4 tiers**: parses < runs < passes tests < correct+idiomatic. It
  certifies only up to tier 3. **Idiomaticity is invisible to the oracle** → must
  come from SFT (teacher) data, plus optionally a clj-kondo/lint reward term.
- **Reward hacking is real and fast** (degenerate programs, gaming tests). Harden:
  hidden held-out tests, anti-triviality guards, timeouts/sandbox, a random-reward
  control.
- **FIM vs instruct**: the user's instinct is **correct** — use instruct/chat for a
  generate-execute-score loop; FIM is for IDE infill and is degraded by
  instruction-tuning.

### Local reality (corrects the premise)
- **Corpus `/Users/robert/code/cljs` is ~1.9M lines (930 MB, ~17 OSS projects), not
  100K** (100K is just the genmlx subproject). High-quality, idiomatic; **prepared,
  deduped SFT datasets already exist** (FIM `train.jsonl` 70K + instruction
  `training_pairs.jsonl`). Data volume/quality is NOT the bottleneck. (License mix:
  AGPL/MPL present — fine for private, document if distributed.)
- **Prior fine-tunes failed structurally, not "the model is bad":** they targeted
  **Qwen2.5-Coder (`model_type qwen2`)** which GenMLX's *differentiable* owned forward
  cannot score (only qwen3/qwen3_5), and used a **FIM** objective (wrong for an
  execution loop). Fix: SFT **qwen3.5-0.8b / qwen3-4b** (owned-forward families) on
  **instruction** data.

### Infra already in GenMLX (~60–70% of the loop)
- **Built + demonstrated**: the GRPO boundary (`world.train` over `GrpoTrainingEngine`),
  the pure `(prompt,completion)->number` reward seam, the **SCI cljs oracle**
  (`codegen.eval` + `msa-score`) certifying parse/eval/test + a scalar GFI reward
  (real trend −13.4→−9.0).
- **Gaps**: a native **SFT membrane** (the `SftTrainingEngine` exists in `@genmlx/core`
  but isn't wired in cljs) — *or* use the already-wired **mlx-lm LoRA** path for SFT;
  **task/curriculum generation** (one hardcoded task today); an **iterate
  orchestrator** (saveModel→reload). **KL-to-base is now available** (genmlx-65d5,
  just merged) — the stability mechanism an iterated loop needs.

## Part 2 — Qwen3-Coder-Next

### What it is (config.json + HF card, fetched)
- **80B total / ~3B active MoE**, `model_type=qwen3_next`, `Qwen3NextForCausalLM`,
  built on Qwen3-Next-80B-A3B-Base. Apache-2.0, Feb 2026, non-thinking,
  code-specialized (SWE-Bench Verified ~70% per card).
- **Hybrid attention**: 48 layers = 12×[3× Gated DeltaNet (linear) + 1× full
  attention]; `full_attention_interval=4`. **MoE**: 512 experts, top-10 + 1 shared.
  hidden 2048, vocab 151936, **256K context**.
- Distinct from Qwen3-Coder-480B-A35B and the 30B-A3B "Flash" (only the -Next line is
  hybrid Gated DeltaNet).

### Runs on 96GB? Yes (4-bit)
- Local `~/.cache/models/Qwen3-Coder-Next-4bit` = **42 GB on disk** (44.84 GB index,
  79.67B params, mlx-community MLX-native affine 4-bit).
- Single-session footprint ~48–53 GB; **KV cache tiny** (only 12/48 layers are full
  attention → ~6.4 GB even at 262K). **~33–38 GB free** on 96GB (raise
  `iogpu.wired_limit_mb`). Decode **~40–70 tok/s** (3B active). 8-bit is too tight.
- All 512 experts must be resident (total params drive memory), but A3B sparsity
  keeps decode fast.

### Integration: teacher-only, offline (recommended)
- mlx-node has **no `qwen3_next` dispatch**, but already implements the same family
  under `qwen3_5_moe` (Gated DeltaNet + MoE). **That MoE forward SIGTRAPs** on real
  large-expert checkpoints (genmlx-5luk; fix deferred genmlx-s8fi). So a native
  GenMLX forward = routing (~0.5d) + fixing the uncatchable-crash MoE forward
  (medium-high, days, risky). **Not worth it near-term.**
- **Python mlx-lm fully supports `qwen3_next`** (dedicated `qwen3_next.py`; the local
  4-bit was converted with mlx-lm) → MLX's C++ ops exist; the gap is only mlx-node's
  Rust forward.
- **Cheapest viable = the user's idea: run the teacher OFFLINE via Python mlx-lm**,
  generate cljs for distillation/SFT + few-shot, with no native GenMLX forward in the
  hot loop and no second MLX addon in-process (sidesteps the nldo two-addon SIGTRAP).

## The convergent recommended architecture (the real "genmlx loop")

Both halves point to the same design:

1. **Teacher (Qwen3-Coder-Next, offline, Python mlx-lm)** → generate diverse,
   idiomatic ClojureScript for a task set (256K ctx holds GenMLX/cljs exemplars).
2. **Oracle-filter (GenMLX, exists)** → run teacher output through SCI eval + held-out
   tests + `msa/score-model` log-ML; keep only valid/high-quality. *This is exactly
   MultiPL-T's winning recipe (execution-validated distillation) — the proven
   niche-language method.*
3. **SFT the small student (qwen3.5-0.8b or 4b, owned forward, instruct format)** on
   the distilled corpus. **This sets the ceiling.** (mlx-lm LoRA works today; a native
   SFT membrane is a nice-to-have.)
4. **GRPO sharpen (GenMLX, ~built)** with the SCI oracle as verifiable reward +
   **KL-to-base (65d5)** for stability, hardened against reward-hacking, on a curated
   difficulty-band task set where the base "sometimes succeeds."
5. **Iterate SFT↔GRPO**, re-mixing fresh teacher data each round; stop when held-out
   pass@1 plateaus / pass@k ceiling stops moving (~3–6 rounds).

## Honest outcome expectation
- **High-probability:** a solid, locally-useful small cljs coder — clearly better than
  the prior (structurally-broken) attempts — especially for **narrow, well-tested,
  program-synthesis** tasks (the MSA/codegen use case GenMLX actually needs).
- **Low-probability:** a "near-excellent, general" cljs LLM at 0.8B. **4B materially
  raises the ceiling**; use it if the budget allows.
- The lever that matters most is the **teacher-distilled SFT corpus**, not loop count.

## Smallest-first build order
1. Teacher harness (Python mlx-lm gen script) — trivial; mlx-lm supports qwen3_next.
2. Oracle-filter distillation pipeline — reuse `codegen.cljs` SCI + `msa_score.cljs`.
3. SFT the student — mlx-lm LoRA first (already wired); native SFT membrane later.
4. Task/curriculum generator (the teacher can generate tasks too).
5. GRPO sharpening with KL + reward-hardening; iterate orchestrator (saveModel→reload).
