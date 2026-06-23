# R2: The Loop-vs-Rerank Decision Gate — Results

*Crown-jewel roadmap (genmlx-3g6i), Phase R2 (genmlx-xrps). 2026-06-23.*
*Worker: `Qwen3.6-35B-A3B-4bit` (mlx-lm, out-of-process). Native-free oracle.*

## Verdict: GO-RERANK

The fixed loop does **not** beat compute-matched one-shot + exact-evidence rerank — decisively, on
the held-out-family split, with the accept-rule fix *validated and not the culprit*. The crown
(resource-rational synthesis = inference) is preserved; the Phase-4 **substrate is the rerank
allocator**, not the loop. The loop survives only as the bounded validity-repair sub-component it
provably is.

## Setup

R2 is the Phase-4-substrate gate. Same model, same compute, same exact oracle on both arms — the only
variable is loop-vs-one-shot. Four cells (a 2×2 of {loop, one-shot} × {fixed-σ, co-refined-σ}):

- **OS-fixed** — one-shot best-of-K, reranked by verbatim exact evidence (the de-masked R0 25% bar).
- **OS-coref** — one-shot best-of-K, reranked by **σ-co-refined** evidence (the SAME σ tuning the loop
  gets → the only difference vs L-coref is loop-vs-one-shot; this defuses the σ-laundering hazard).
- **L-strict** — the legacy strict-greedy loop (the Phase-3 search operator).
- **L-coref** — the R1 (genmlx-8smp) co-refined-accept loop (this session's fix), LLM-only proposer.

Compute-matched `K_OS = K_STEP×MAX_STEPS×(1+REVISE) = 40`. Natural difficulty. Family-split is the
headline (held-out `:segmented` family = leakage-immune cohort). Bootstrap CI (B=2000) on the paired
`L-coref − OS-coref` evidence advantage. Numbers are **log model evidence** (exact marginal
likelihood; higher = better); `✓` = cleared the structure-finding solve-bar.

## Results (35B, 9 eval tasks = 5 within + 4 held-out-family)

| cohort | OS-fixed | OS-coref (rerank) | L-strict (Phase-3) | L-coref (R1 fix) | paired Δev (L-coref − OS-coref) | gate |
|---|---|---|---|---|---|---|
| **FAMILY** (n=4) | 100% | **100%** | 0% | **50%** | −2.15, 95% CI **[−3.95, −0.72]** | **GO-RERANK** |
| WITHIN (n=5) | 80% | 80% | 20% | 20% | −10.52, CI [−19.57, −1.47] | GO-RERANK |
| ALL (n=9) | 89% | **89%** | 11% | **33%** | −6.33, CI [−12.61, −1.45] | GO-RERANK |

(solve-rate columns; CI on paired evidence delta.)

Per-family OS-coref solve: `linear 100% · per-group 100% · ar1 100% · segmented 100% · shared-mean
100% · varying-slopes 0%`.

## Three findings

1. **GO-RERANK is robust and statistical, not marginal.** On all three cohorts the loop is *worse*
   (CI upper bound < 0), not merely "not better." One-shot+rerank solves 89% overall to the loop's 33%.
   And the budget effect confirms the mechanism: K=18→40 lifted one-shot (per-group, ar1 went unsolved
   → solved) while the loop stayed stuck — **best-of-K converts compute into solved tasks; greedy
   construction does not** (Gu 2026; ModelSMC 2026 — the loop's only edge is proposer-diversity, which
   greedy commit lacks).

2. **The co-refine fix is validated — and that *strengthens* the verdict.** On the held-out family,
   `L-coref` 50% vs `L-strict` 0%: the R1 accept-rule fix *doubled* the loop's family solve rate from
   zero (it flips segmented-i0 from −14.00✗ to −12.70✓). So we cannot dismiss the result as a broken
   accept rule — we fixed it, the fix demonstrably helps, and rerank *still* wins everywhere.

3. **The one genuine wall is a coverage ceiling, not a loop-vs-rerank distinction.**
   `varying-slopes` is 0% for **both** strategies — the 6-latent coordinated model is outside the
   35B-A3B's one-shot *support*, so best-of-K can't select it and greedy can't construct it. This is
   the proposer-bound wall, and it is the same for both arms (the lone `loop>rerank=true` there is an
   artifact: one-shot emitted no valid candidate, the loop emitted a scoring-but-not-solving one).

## Cost (the Phase-4 gradient)

`oneshot 360 samples / 2803 s · loop-strict 64 / 497 s · loop-coref 48 / 382 s · wall ~2.46 h`. The
one-shot arm dominates wall-clock — exactly the cost gradient the resource-rational metareasoner
optimizes (more K = more solved but more compute; the loops are cheaper because they plateau early).

## What this means for Phase 4

- **Substrate = rerank allocator.** Control sites `[:tier]/[:K]`: allocate {cheap small one-shot,
  expensive teacher one-shot, oracle, how-many-K-before-rerank}. The loop is retained ONLY as bounded
  validity-repair (its one proven win: weak-model one-shot 4%→26%), never as evidence search.
- **Raise the ceiling two ways:** a stronger/more-diverse proposer (Qwen3-Coder-Next → higher one-shot
  coverage on the compositional families where the 35B is thin — directly attacks the varying-slopes
  wall) and a broader exact verifier (extend exact/RB scoring to more families).
- **The Jetson spend re-points:** harvest Coder-Next's one-shot best-of-K *solved programs* (its
  verified value is one-shot quality) for distillation into the cheap fast tier — not loop trajectories.

## Honest open edges (the only things that could revisit GO-RERANK)

The plan flagged two loop-hopes that were **not** tested here, both low-probability per the data
direction and the literature:

1. **Soft-accept SMC over construction** (not strict/co-refine *greedy*) — a particle population with
   resampling can carry a temporarily-worse intermediate through a valley. Needs reversible birth-death
   edits. Untested. (ModelSMC shows population>greedy on the same objective — but proposer-diversity-driven.)
2. **A real non-conjugate cliff family *in the loop*** (GMM/hierarchical). The RB scorer
   (`genmlx.inference.rb_mixture`, R4.5-minimal) is built and verified (exact collapsed marginal,
   monotone partial-credit path), but wiring a mixture into the loop needs the synth DSL to *represent*
   mixtures (genmlx-akvp). The loop's theoretical niche is exactly this complement (where one-shot
   coverage is thin) — so it's the one place left to look, and the RB scorer makes it testable.

Neither is required for the GO-RERANK substrate decision on the natural families, which is decisive.

## Reproduce

```
scripts/run_r2_bakeoff.sh Qwen3.6-35B-A3B-4bit 35b-family   # worker up → run → teardown
# env: ROUND INSTANCES EVAL_FAMILIES K_STEP MAX_STEPS REVISE MAX_TOKENS NP SEED BOOT  (MOCK=1 = native-free dry run)
```
Artifact: `~/genmlx-loop-artifacts/r2/r2_bakeoff_35b-family.json`.
