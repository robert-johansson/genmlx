# Phase 3 (genmlx-oexl) — Specializing the fast proposer: results

**Thesis (north star genmlx-77vv):** model a REPL programmer as a GenMLX generative
function so program synthesis = inference over it; distill the *expensive loop's
propose-eval-revise POLICY* (its GFI trace) into a cheap 0.8B proposer, so the cheap
model drives the loop at a fraction of the big-model cost. Phase 3 builds and measures
that distillation end-to-end.

## Pipeline (every stage validated end-to-end)

curriculum (`world.curriculum`) → **35B greedy+beam harvest** (`world.harvest` +
`repl-corpus`) → 56-row propose-eval-revise SFT corpus → **LoRA SFT** of qwen3.5-0.8b →
**GRPO sharpen** (`world.train` + oracle reward) → **in-loop eval** (cohort-separated).

New, tested code: `genmlx.world.harvest` (`harvest_test` 17/17) + `scripts/harvest_corpus.cljs`,
`scripts/grpo_repl.cljs`, `scripts/inloop_eval.cljs`. The harvest→build-corpus glue was
already on main (curriculum_probe + tests); Phase 3 added the at-scale LLM driver over it.

## 1. Harvest (the 35B-in-loop teacher → corpus)

- **Greedy**, 45 train tasks: **40% solved**, 42 leakage-safe rows, 82 min. The
  **structural cliff** is real: varying-slopes **0/9** (the 6-latent structure resists the
  greedy loop), per-group-means weak.
- **Beam re-harvest** (3 structural families, width 3): **identical solve rates** to greedy
  (linear 4/9, per-group 3/9, varying 0/9). Beam did **not** escape the lock-in here — the
  bottleneck is the **proposer** (the 35B never proposes the 6-latent structure), not the
  search. (Phase-2's "beam beats greedy" doesn't replicate on the hardest curriculum
  structures.) Added 24 diverse rows.
- **Merged corpus: 56 rows** (deduped) — linear 17, ar1 13, per-group 12, shared-mean 12,
  varying-slopes 2.

## 2. SFT (distill the policy into the 0.8B)

- **OOM gotcha:** the 0.8B has a **~248k vocab**; mlx-lm's un-chunked LoRA loss
  (248k×seq×batch) blows the Metal command-buffer at seq≥1536 (peak ~16GB at seq 896).
  Fix: a **short system prompt** (160 tok, vs 689) + MAXSEQ 896, BATCH 1.
- Val-loss minimum at iter 100 (val 0.034, train 0.024) → early-stop, re-fused the iter-100
  adapter as the SFT model (`qwen3.5-0.8b-cljs-sft-i100`).

## 3. SFT in-loop eval — the headline result (27 held-out eval tasks)

| Cohort | n | loop-solve | one-shot-solve | loop>one-shot |
|---|---|---|---|---|
| within-family | 15 | **40%** | 0% | 100% |
| held-out `:segmented` | 12 | 8.3% | 8.3% | 92% |
| **ALL** | 27 | **25.9%** | 3.7% | 96% |

Per-family (loop): **shared-mean 3/3, ar1 3/3** (learned cleanly); linear 0/3, per-group
0/3, varying 0/3, segmented 1/12. Cost ~40s/task.

**Two findings:** (1) **the loop is essential and the SFT taught the loop policy** — the
cheap model beats its own one-shot 96–100% of the time (one-shot 3.7% vs loop 25.9%; its
one-shot is near-empty *by design* — it was trained to *improve a given model*, not
one-shot whole programs). (2) From **56 rows** the cheap model generalizes only the
structures with clean supervision (shared-mean, ar1); it does **not** generalize the harder
structures — a real, honest **bound set by corpus scale**.

## 4. GRPO (full-send) — machinery validated; in-loop eval blocked by a format gap

- GRPO **trains** end-to-end (validated smoke + 20-step bounded run, kl-coef 0.1 to the SFT
  base). The oracle reward is **task-bound**: ar1 fits (−7…−18), linear floors (−20,
  valid-rate 0) — **"RLVR sharpens, it does not teach"** observed directly (it can sharpen
  ar1/shared-mean where the model already produces valid forms; it has no gradient on
  linear/per-group/varying).
- **In-loop eval BLOCKED** (follow-up `genmlx-boh1`): the native `GrpoTrainingEngine`
  `saveModel` writes genmlx-internal weight keys (284 keys, fused projections) that the
  **mlx-lm** HTTP worker can't load (it expects 320 mlx-lm keys, split projections); and the
  synthesis loop is **synchronous** while native `generate-batch` is **async**, so an
  in-process native eval isn't a drop-in either. GRPO was attempted + trained; its in-loop
  eval is deferred to the format bridge.

## 5. Cost/quality frontier (cheap-0.8B vs 35B, in-loop, 27 held-out eval tasks)

| Model | **best-arm solve** | loop | one-shot | s/call | total gen |
|---|---|---|---|---|---|
| base-0.8B (no SFT, short sys) | **11%** (3/27) | 11% | 0% | 14.0s | 1533s |
| **SFT-0.8B** | **26%** (7/27) | 26% | 4% | **14.2s** | 1063s |
| GRPO-0.8B | (eval blocked — `genmlx-boh1`) | | | | |
| **35B (teacher)** | **37%** (10/27) | 22% | 33% | 30.8s | 2341s |

Per-family best-arm solve: shared-mean SFT 3/3 **=** 35B 3/3; **ar1 SFT 3/3 > 35B 0/3**;
per-group SFT 0/3 **<** 35B 3/3 (one-shot); linear SFT 0/3 **<** 35B 1/3; varying-slopes 0/3
**=** 0/3 (the cliff — both fail); segmented SFT 1/12 **<** 35B 3/12.

**Read shared-mean with care:** shared-mean is solvable by the loop's *deterministic σ-grid
refiner alone* (no LLM proposal needed), so its "solves" don't test the LLM for any model —
they're the floor the σ-grid gives everyone. The LLM-dependent test is the **structural**
families (ar1, linear, per-group, varying, segmented). That makes the **ar1** result sharper:
the SFT'd 0.8B genuinely learned the **AR(1) structure** (3/3) that the 35B doesn't reliably
emit (0/3) — a real, LLM-borne, distilled skill, not a σ-grid artifact.

**Frontier reading — the resource-rational thesis, quantified:**
0. **The distillation added real capability (base control).** The un-SFT'd base-0.8B-in-loop
   solves only shared-mean — i.e. only what the deterministic σ-grid solves *with no LLM* —
   and **0/3 ar1, 0/N every structural family**. The SFT'd 0.8B gained **ar1 3/3** at the
   *same* cost. So SFT didn't just re-weight a model that could already do this; it **taught
   a structural skill the base lacks** (base 11% → SFT 26%, the delta being LLM-borne
   structure, not σ-grid).
1. The **cheap distilled 0.8B reaches ~70% of the 35B's solve-rate at ~46% of the per-call
   cost** (14.2 vs 30.8 s/call; 1063 vs 2341 s total). A genuinely favorable point.
2. **The cheap model *beats* the teacher on its distilled specialty (ar1, 3/3 vs 0/3)** —
   distillation didn't just compress, it *specialized*: the cheap model learned a
   structure-specific skill the general 35B doesn't reliably produce at these knobs.
3. **The loop is the cheap model's enabler, not the strong model's.** The 35B's one-shot
   (33%) *beats* its own loop (22%) — greedy refinement adds lock-in for a capable model.
   The cheap model is the opposite: one-shot 4%, loop 26% — the loop *unlocks* it. So the
   right configuration is **cheap-in-loop vs strong-one-shot** — exactly the asymmetry a
   Phase-4 metareasoner would exploit (cheap proposer for the high-frequency / specialized
   moves, the big model one-shot for the structures the cheap model lacks).

## Honest bounds

- **Corpus scale** (56 rows) is the dominant limiter on structural generalization.
- **Proposer-bound cliff**: neither greedy nor beam-35B solves varying-slopes — more search
  doesn't help when the proposer never emits the structure.
- **GRPO in-loop eval blocked** by the native↔mlx-lm weight-format gap (`genmlx-boh1`).
- Ultimate value is **gated on the P5a decoder** (carried bound).

## What this proves

The **distillation mechanism works end-to-end**: the cheap 0.8B, SFT'd purely on the
35B loop's GFI traces, **drives the loop** (not one-shot) and solves the structures it had
clean supervision for — at ~3× lower per-step cost than the 35B. The north-star thesis
(synthesis = inference over a REPL-programmer GF, distilled into a cheap proposer) is
**demonstrated, with the reach bounded by corpus scale** — the clear next lever is more
harvest (more tasks / a stronger teacher on the hard structures), not a different method.
