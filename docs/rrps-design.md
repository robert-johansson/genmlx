# Resource-Rational Program Synthesis (RRPS) — Design

**Status:** design / not yet built. The aspirational "title (B)" crown result for the
TOPML paper (`genmlx-wdop`). This document is the output of a 26-agent research +
design + adversarial-critique workflow (run `wgz54tgog`, 2026-06-16) that read the
actual GenMLX source (file:line citations below), researched the load-bearing external
facts (nbb/SCI runtime macros, Bun-Workers-vs-MLX, the synthesis-as-inference and
metareasoning literature), generated four competing architectures, and attacked each.

**One-line thesis:** *close the probabilistic-program synthesis loop under an explicit
value-of-computation controller that resource-rationally allocates synthesis-and-scoring
compute over **model space** — how many programs to propose × how deeply to score each ×
when to stop — scored by an exact marginal-evidence oracle.*

---

## 1. Why this is a real contribution (and where it is not)

The synthesis-as-Bayesian-inference framing is fully established prior art:

- **DreamCoder** (Ellis et al., PLDI 2021): neural recognition model amortizes *where*
  to search + library learning. Amortizes **search direction**, never *how much compute
  is worth spending*.
- **Saad et al. 2019** (POPL, "Bayesian Synthesis of Probabilistic Programs"): automatic
  data modeling as posterior inference over programs, scored by **collapsed/marginal
  likelihood** (the same conjugate-elimination family GenMLX's Level-3 uses). Uses a
  **fixed, uniform** MCMC budget.
- **Loula et al. 2025** (ICLR, "Syntactic and Semantic Control of LLMs via SMC") /
  **LLaMPPL** (Lew/Zhi-Xuan/Mansinghka 2023) / **twisted SMC** (Zhao et al., ICML 2024):
  token-level SMC steering of **one** sequence with an **external** black-box LM + a
  **fixed particle budget**.

GenMLX's crown jewel (`z9eo`: LLM+grammar → `(gen …)` program → **exact** marginal-log-ML
scoring → recover the generating model) is therefore a strong **re-instantiation** of
Saad-2019 on a novel substrate (GFI / MLX / ClojureScript) — defensible as "title (A)",
but **not itself the new science.**

**The open gap — verified across the metareasoning literature** (Russell & Wefald 1991;
Hay & Russell, UAI 2012; Callaway et al., UAI 2018; Lieder & Griffiths, BBS 2020):
*resource-rational metareasoning / value-of-computation has never been applied with
probabilistic-program synthesis / Bayesian model-structure search as the object level.*
That meta-level — a controller that adaptively allocates **#proposals × inference-per-
candidate × stopping** over a population of candidate programs, scored by exact evidence,
with proposer + synthesized models + scorer + controller all generative functions in one
GFI algebra — is the honest novel slice.

> **Novelty claim (paper-ready, one sentence):** *We close the probabilistic-program
> synthesis loop under an explicit value-of-computation controller that resource-rationally
> allocates synthesis-and-scoring compute over model space, scored by an exact
> marginal-evidence oracle, demonstrated by measured net-utility frontier-dominance over
> fixed-budget enumeration and meta-greedy under intrinsic conjugacy-driven heterogeneity
> with seeds + bootstrap CIs — the meta-level that DreamCoder, Saad-2019, and Loula all
> structurally lack.*

The VOC mechanism itself (myopic value-of-computation) is **cited, non-novel** (Russell-
Wefald / Callaway). The contribution is the *synthesis-over-model-space instantiation +
the measured win*, never the mechanism.

---

## 2. Honest verdicts on the originating intuitions

These corrections are load-bearing — three of the five intuitions are false *as stated*,
each with a real underlying truth. The design is built on the corrected versions.

| Intuition | Verdict | Corrected mechanism |
|---|---|---|
| "particles generate code **on the GPU**" | ❌ **false** | `qwen3_forward.cljs` hardwires `batch=1`; generation is host-coupled per token (`core.cljs:61` EOS via `mx/item`); distinct programs have distinct trace addresses and **cannot share a batch axis** (`vectorized.cljs` requires homogeneous leaves; the combinator-batched path drops to a host N-loop). Particles are **host-resident program values**; the GPU **scores** them (LLM logits + per-candidate exact/IS graphs), **serialized across distinct candidates, no throughput scaling in N**. On EASY conjugate instances, L3 elimination is host-cheap scalar arithmetic — the GPU is *incidental*, not load-bearing, to the resource-rational claim. |
| "nbb **hot-reloads** generated code, incl. **macros**" | ❌ false as a name, ✅ real as a capability | "Hot reload" (file-watch) is **not** an nbb feature and **is not needed**. SCI **does** support runtime `defmacro` + `&env`/`&form` (empirically verified in pinned nbb **1.4.208**). But the load-bearing primitive is **runtime SCI eval of generated *forms*** (`sci/eval-string` / `sci/eval-form` into a forked context) — and `msa.cljs` deliberately *avoids* live gen-macroexpansion: it hand-builds the source form (`code->source-form`, `msa.cljs:104`) → `make-gen-fn`, reconstructing the schema separately. Frame as *"runtime interpretation of synthesized forms into a forked SCI context."* |
| "**Bun Workers** for parallel synthesis" | ❌ dead end (explicit non-goal) | **MLX is not thread-safe** (maintainer, `ml-explore/mlx` #2133, #3078); **streams are thread-local since 0.31.2**; MLX graphs/arrays cannot cross Worker (per-worker `napi_env`) or process (#2457) boundaries; multi-process forces a **full model-weights copy per process** (linear memory, no KV-cache share). The single Metal GPU **time-slices, doesn't multiply FLOPs**. `proc.cljs/worker-pool` correctly throws. The correct substrate is **adaptive serial allocation over a serial-GPU + serial-host system** — which is what resource-rationality *is for*. |
| "**malli** is a match made in heaven" | ⚠️ partly, oversold | malli types the **leaves** — the per-trace-site slot grammar (which of 27 distributions, which scalar args) via `schema_grammar.cljs` schema→regex→DFA→byte-mask, with exact teacher-forced log-density (`structured.cljs`). But `schema->regex` **throws on recursive/registry refs** (`schema_grammar.cljs:25`) → **no single malli schema can express the recursive `(gen …)` AST.** The recursive program *tree* grammar is the **edamame reader-as-grammar** (`codegen/eval.cljs` `prefix-status`) **+ Instaparse**, which import zero malli. **malli types the leaves; the reader/Instaparse types the tree.** |
| "**ClojureScript writing ClojureScript** / closure" | ⚠️ true as substrate, not as the result | Real and shipped: a synthesized program is simultaneously data (`schema.cljs` walks the quoted form) and code (SCI evals it); proposer + models + scorer + controller are all GFI values in one algebra — the genuine "closure under the system's own operations" affordance. **But** resource-rational/PPL reviewers discount "nice Lisp" — lead with the measured net-utility frontier; relegate closure/homoiconicity to a one-paragraph enabling-architecture affordance. |

---

## 3. Architecture — five components, all reusing existing code

RRPS = a VOC controller allocating synthesis+scoring compute over a **host pool** of
candidate `(gen …)` programs, scored by an exact-evidence oracle, with an exact
closed-form held-out reward.

### (1) Proposer — host LLM decode on GPU logits
Reuse `msa.cljs/generate-knowledge-candidate` (Instaparse `math-spec-grammar`,
`msa.cljs:218`) → `eval-model` (SCI, `msa.cljs:142`) → `code->source-form`
(`msa.cljs:104`) → `dyn/make-gen-fn` (`dynamic.cljs:1210`), so each candidate is a real
DynamicGF with keyword trace-sites and schema/conjugacy firing identically to
hand-written models.

To kill proposer nondeterminism in the seed sweep (preserve paired-seed CIs): widen the
Instaparse grammar modestly to span **beta-bernoulli / gamma-poisson / normal-normal +
structural depth** (1-vs-2-latent, with/without a redundant covariate), then generate a
**large per-task candidate stream once with seeded proposer keys** (`dyn/with-key`) and
**freeze it as an ordered stream per (task, proposer-seed)**. The `:propose` action
reveals the next element of *that seed's* stream. This is honest "live synthesis under a
paired seed" — the stream order, length, and *arrival time of the correct structure* are
seed-dependent and the controller cannot see ahead — not "reading a static list." A small
separate **fully-live confirmation sweep** (fewer seeds, no CI) shows transfer.

### (2) Evidence scorer + schema-read cost predictor — the single strongest core
*(kept by all four critiques.)* Reuse `msa.cljs/score-model*` (`msa.cljs:461`):
- conjugate candidate (conjugacy-table, 7 families) → `score-exact` (`msa.cljs:437`) = one
  analytical `p/generate` weight = **exact marginal log p(obs)**;
- non-conjugate → `score-is` (log-mean-exp; the `:deepen` knob = n-particles).

**Crucial:** `make-gen-fn` runs schema extraction + conjugacy classification at
**construction with zero GPU** (`dynamic.cljs:1210`, `conjugacy.cljs`), so the controller
reads each candidate's **cost/quality *class*** (L3-exact-cheap-deterministic vs
IS-noisy-expensive) **before any GPU work** — an intrinsic, schema-readable difficulty
signal. Cross-check `score-exact` against the independent closed form
`nn-marginal-closed` (`synthesis_occam.cljs:260`, agreement ~5e-7) **and** a high-N IS
oracle.

### (3) Synth steppable (new: `src/genmlx/control/synth_steppable.cljs`, ~140 LOC)
Implements the `{:init :step :done? :best}` contract that `proc.cljs/with-deadline` +
`meta_mdp/controlled-steppable` already consume.
- State `= {:pool [{:gf :code :log-ml-est :score-method :score-depth :class}] :stream
  :stream-idx :task :key}`.
- `:step` executes ONE controller-chosen action:
  - `:propose` — reveal next stream element → `make-gen-fn` → shallow score → `conj` pool;
  - `:deepen` — re-score the current top-uncertain IS candidate at 2× n-particles;
  - `:stop`.
- `:best` `= argmax-log-ml-est :gf`.

This is the unbuilt `SMCState`→synthesis adapter the code map names.

### (4) Genuine K-action VOC controller (extend `meta_mdp.cljs` — **real work**)
Today `meta_mdp.cljs:65-81` hardcodes ONE `bstep`, one scalar VOC, a 2-element `[voc 0.0]`
EU, a binary branch. Generalize `controlled-steppable` to:
- **per-action trial-advance** (K separate metered lookaheads — `:propose` vs `:deepen`
  have different cost and `dv`-delta);
- per-action cost from `cost/measure-step`;
- EU vector `[voc-propose voc-deepen 0.0]` → `h/softmax-action` (`agents/helpers.cljs:70`,
  already K-vector-capable);
- policy gen-fn (`meta_mdp.cljs:105`) unchanged.

This makes the deferred `:add-particle`/`:refine` slot (`meta_mdp.cljs:32`) real for
synthesis. **Not** the "60-line vector-widening" the early designs understated.

### (5) Exact held-out reward + host cost meter + scheduler
- **Reward** (`decision_value.cljs`, new `dv-fn`): held-out predictive log-likelihood of
  the **selected** program on a fresh split. For the conjugate demo this is **exact and
  closed-form** via posterior-predictive `= nn-marginal-closed(train+test) −
  nn-marginal-closed(train)`. This **resolves the cross-cutting fatal flaw** ("reward is
  noisy exactly where adaptivity must work"): the demo is conjugate-structure-search where
  the HARD axis is search-depth + IS-ranking-variance of the *ranking*, while the held-out
  *reward* oracle stays exact even on HARD instances. `assert-downstream!`
  (`decision_value.cljs:18`) guards against log-ML-as-reward; the discipline is enforced by
  experiment construction (the guard is only a key-name check).
- **Cost meter** (`cost.cljs`): ADD host-synthesis counters `:llm-tokens` (proposer
  decode) + `:sci-evals` (eval-model), since `cost/measure` sees only GPU-membrane deltas
  and synthesis is **host-dominated**; fold via `cost+`. Within a frozen-stream seed,
  `:llm-tokens` is deterministic per candidate, preserving paired-seed bootstrap CI rigor;
  wall-ns is the non-deterministic cross-check only, reported as a secondary frontier.
- **Scheduler** = `world/proc.cljs/with-deadline`, **unchanged**, the sole side effect.
- **Bun Workers = explicit non-goal** (worker-pool unproven; MLX not thread-safe). GPU does
  LLM logits + per-candidate exact/IS scoring (host-serialized across distinct programs, no
  batched-across-distinct-programs); host proposes/evals/decides — stated honestly.

---

## 4. The demonstration experiment

**RRPS-over-conjugate-structure-search:** resource-rational synthesis over a
bimodal-difficulty distribution of data-modeling instances, where the HARD axis lives in
**proposal-discovery + IS-ranking-variance** but the **reward oracle stays exact**.

**Task distribution (intrinsic, schema-readable heterogeneity — the make-or-break).**
Each instance is a **multi-observation** dataset (k ≥ 8 points, *not* a single datum)
drawn from a known ground-truth gen-program. 50/50 bimodal, paired across methods via host
LCG (`seed → difficulty`, mirroring `anytime_control.cljs:290`):

- **EASY:** ground truth is normal-normal conjugate **and** first-stream-correct. One
  `score-exact` (zero-variance evidence) decisively ranks it; ~2 proposals suffice.
  Bayes-optimal: few proposals, exact score, stop early.
- **HARD:** ground truth is a structure (e.g. **2-latent regression vs spurious-covariate**,
  or a beta-bernoulli the proposer emits late) that (a) appears **late** in the proposer
  stream (needs many `:propose` to discover) **and** (b) competes with conjugate-looking-
  but-wrong alternatives whose IS evidence estimate is **high-variance at shallow depth**,
  so shallow scoring **mis-ranks** — needs `:deepen`. Bayes-optimal: many proposals + deep
  IS scoring, stop late.

No single fixed `(n-proposals, n-particles)` serves both: small under-serves HARD (misses
/ mis-ranks), large over-pays EASY.

> **Pre-registered heterogeneity check (mandatory, before any controller claim):** derive
> the per-type Bayes-optimal `(n-proposals, depth)` by exhaustive grid and **report that no
> fixed point lies within the 95% CI of both types**. Without this the win is a hand-tuned
> artifact.

**Metric.** `net-utility = held-out-predictive-LL(selected) − λ·compute`, with
`compute = :llm-tokens + :forced-evals + :sci-evals` (deterministic within seed). Held-out
LL is **exact closed-form** (posterior-predictive) for conjugate selected programs; for
non-conjugate selected programs, a high-N posterior-predictive IS estimate with enough
held-out points that reward noise ≪ difficulty signal.

**Oracle.** (a) `score-exact` cross-checked ~5e-7 vs `nn-marginal-closed` + high-N IS;
(b) ground-truth program known by construction → recovery rate; (c) exact closed-form
held-out-LL of the *true* generating program = achievable-utility floor; (d) **SBC** (Talts
et al. 2018) over the synthesis/scoring loop certifies evidence calibration (watch
U-shapes) BEFORE any controller claim.

**Baselines (controller must beat the whole set).**
1. fixed-budget enumeration over the full `(n-proposals × scoring-depth)` grid — the
   **binding baseline is the best-tuned fixed point**;
2. meta-greedy / myopic-VOC at hysteresis=1 (Callaway 2018 / Russell-Wefald) = the
   controller's own myopic policy → the contribution is the *instantiation + measured win*,
   not the mechanism (walled non-novel);
3. entropy/evidence-gap threshold stopper (non-VOC heuristic);
4. LLM-only-no-scoring (ablates the exact-evidence scorer);
5. **adaptivity ablation:** controller replaced by its own mean budget — proves per-instance
   adaptation, not the substrate, carries the win.

**Rigor.** ≥30 paired seeds (same data + same proposer stream per seed across all methods,
differing only in allocation policy). Deterministic non-overlapping bootstrap CIs (B=2000,
reuse `anytime_control.cljs:153`). Pareto frontier (held-out-LL vs compute) over
λ ∈ {0, 0.01, 0.03, 0.08, …}; claim = **frontier dominance**. Headline-win predicate
(`anytime_control.cljs:648`, reused verbatim): controller mean > EVERY fixed point AND
paired 95% CI vs best-tuned-fixed excludes 0 AND ≥ meta-greedy.

> **Honest gate.** This demo *counts* only if it produces an actual **CI-lo > 0** win —
> which the `gdtq` SSM bench (single knob, clean oracle) did **not**. The two coupled knobs
> (proposal-count AND scoring-depth) + the larger exact-vs-IS variance gap are the bet to
> clear the bar. **If it lands mean-only, report mean-only and DO NOT claim title-B.**

---

## 5. Build phases

**P0 — Heterogeneity proof + oracle (the GATE; build before any controller code).**
Multi-observation conjugate-structure task distribution; exact closed-form held-out
posterior-predictive oracle; `score-exact` vs `nn-marginal-closed` cross-check; SBC pass;
pre-registered per-type Bayes-optimal grid showing no fixed point serves both. **If this
fails, STOP** — title-B is unreachable; the paper floors at title-A.
*Reuses:* `msa.cljs` `score-model*`/`score-exact` (`:461,:437`), `synthesis_occam.cljs`
`nn-marginal-closed` (`:260`), `conjugacy.cljs`, `schema.cljs` `extract-schema`.

**P1 — Frozen seeded proposer stream.** Widened Instaparse grammar; per-(task,seed)
ordered candidate stream generated live (LLM-on-GPU) and frozen; small live-confirmation
harness.
*Reuses:* `msa.cljs` `generate-knowledge-candidate` (`:385`) / `math-spec-grammar`
(`:218`) / `code->source-form` (`:104`) / `eval-model` (`:142`); `dyn/with-key`.

**P2 — Synth steppable + host cost meter.** `synth_steppable.cljs` (propose/deepen/stop
over frozen stream + pool); `cost.cljs` `:llm-tokens`/`:sci-evals` counters; schema-read
cost-class predictor wired in.
*Reuses:* `steppable.cljs` `{:init :step :done? :best}` contract; `cost.cljs`
`measure-step`/`cost+`; `make-gen-fn` zero-GPU schema/conjugacy classification.

**P3 — Genuine K-action VOC controller.** `controlled-steppable` generalized to
per-action trial-advance + per-action cost + K-EU softmax; exact held-out `dv-fn` plugged
as `:decision-value-fn`; AR(1) bench regression-green (no regression).
*Reuses:* `meta_mdp.cljs` `make-metareasoner`/`controlled-steppable` (`:36,:85`);
`agents/helpers.cljs` `softmax-action` (`:70`); `decision_value.cljs` `assert-downstream!`.

**P4 — Rigor sweep + frozen artifact.** ≥30 paired seeds, baselines + ablations, Pareto
frontier, headline-win predicate, recovery study; frozen `results/control/rrps.md` with
honest no / mean-only / YES verdicts.
*Reuses:* `anytime_control.cljs` `bootstrap-ci` (`:153`) / `seed→rs` heterogeneity
(`:290`) / paired design (`:438`) / `emit-results` headline predicate (`:648`);
`world/proc.cljs` `with-deadline`.

**Minimal vs ambitious.** *Minimal (1-2 wks, the floor):* conjugate-only structure-search,
HARD axis = pure scoring-depth on a single clean knob, one action distinction that matters
(`:deepen`-vs-`:stop` on IS-scorable candidates); prove the pre-registered heterogeneity
divergence, then show CI-lo>0 frontier-dominance vs fixed-depth sweep + meta-greedy +
adaptivity ablation. Removes the noisy-HARD-reward + IS-bias confounds, and does *better*
than `gdtq`'s own null on a clean problem first. *Ambitious (title-B target):* two coupled
knobs (proposal-count AND scoring-depth), live-but-seeded stream where the correct
structure arrives late on HARD, multi-latent regression-vs-spurious-covariate search, full
K-action VOC, Pareto frontier over a λ sweep, recovery study, SBC, live-confirmation
transfer sweep, GFI-closure framing as a one-paragraph affordance. **Build P0 first as a
hard gate.**

---

## 6. "done means" checklist

- [ ] `src/genmlx/control/synth_steppable.cljs` implements `{:init :step :done? :best}`
      with actions `{:propose :deepen :stop}`; `:best = argmax-log-ml-est`; self-test that
      `proc/with-deadline` drives it to a committed stop.
- [ ] `meta_mdp/controlled-steppable` GENUINELY generalized to K-action (per-action
      trial-advance + per-action metered cost + EU vector → `softmax-action`); existing
      2-action AR(1) bench still green.
- [ ] `cost.cljs` extended with `:llm-tokens` + `:sci-evals` host counters, folded via
      `cost+`; deterministic within a frozen proposer-stream seed; wall-ns secondary.
- [ ] EXACT held-out reward: closed-form posterior-predictive for conjugate selected
      programs, cross-checked vs high-N IS ~5e-7; `dv-fn` passes `assert-downstream!` AND is
      genuinely out-of-sample by construction.
- [ ] PRE-REGISTERED heterogeneity check committed BEFORE the controller sweep: per-type
      Bayes-optimal `(n-proposals, depth)` grid shows NO fixed point within 95% CI of both
      EASY and HARD.
- [ ] `score-exact == nn-marginal-closed` to ~5e-7 over MULTI-observation groups;
      independent high-N IS oracle agrees.
- [ ] SBC histogram over the synthesis/scoring loop is uniform (no U-shape); committed as
      the correctness gate that precedes any net-utility claim.
- [ ] ≥30 paired seeds; same data + same proposer stream per seed across all 5 methods;
      deterministic bootstrap CIs (B=2000).
- [ ] Pareto frontier (held-out-LL vs compute) over the λ sweep with frontier-dominance
      reported per λ.
- [ ] Headline-win predicate emits per-λ no/mean-only/YES exactly as
      `anytime_control.cljs:648`; at least one λ is an actual CI-lo>0 YES vs best-tuned
      fixed budget, OR the result is reported honestly as mean-only and title-B is NOT
      claimed.
- [ ] Adaptivity ablation (controller-at-own-mean-budget) reported: the win survives it.
- [ ] Recovery study: true-vs-selected program rate + CI across the difficulty sweep.
- [ ] Small fully-live (unfrozen) confirmation sweep shows the frozen-stream result
      transfers; frozen-stream assumption stated.
- [ ] A frozen `results/control/rrps.md` artifact exists with all tables, CIs, SBC,
      heterogeneity check, and honest no/mean-only/YES verdicts (the
      no-claim-without-frozen-artifact gate, `topml-a9v6`).

---

## 7. Risks

- **Dominant risk — it ties.** The published `gdtq` SSM bench (single clean knob, exact
  oracle) is currently mean-only / no-CI-win. RRPS adds confounds (LLM-stream variance,
  host-cost estimation) on a *harder* substrate, so a tie (the documented
  homogeneous-problem null) is the most likely single outcome. If it ties, title-B
  evaporates and the paper floors at title-A. This is the honest gate.
- **Genuine K-action VOC is real work**, not vector-widening: K separate per-action metered
  trial-advances per step, heterogeneous-unit cost commensuration (propose-cost vs
  deepen-cost), non-streak stop semantics. The `switch-method-translate` stub
  (`meta_mdp.cljs:124`) already flags cross-op peek-payload non-portability.
- **Myopic VOC provably under-explores** (Hay-Russell: stops too early) — bites HARD
  instances where the correct program arrives late; can degenerate "adaptive" into "give up
  on HARD". Report net-utility-beats-fixed *despite* this bias, never optimality.
- **Heterogeneity legitimacy is the make-or-break referee check** — it must be INTRINSIC
  (conjugacy off the schema; multi-observation IS variance), not hand-tuned to defeat fixed
  budgets. The pre-registered per-type divergence check is mandatory.
- **`score-is` catches all exceptions to `-Inf`** at n=50 (`msa.cljs:452`) — a
  silently-failing/high-variance IS estimate biases the controller toward
  cheaply-scorable conjugate programs and can corrupt the ranking. Track evidence
  *variance* in the VOC, not just the point estimate; SBC must certify the IS path first.
- **Host cost dominates and is invisible to `cost/measure`** (GPU-only); the new counters
  must be instrumented honestly or the controller meters a fiction.
- **Frozen-stream proposer weakens the "over the live loop" framing** — the small fully-live
  confirmation sweep must show transfer, else a referee reads `:propose` as "read down a
  static list."
- **The exact-evidence oracle is narrow** (7 conjugate families + nil-walled
  non-conjugate). Scope the claim to **conjugate-structure-search**, not "general program
  synthesis." `sci/fork` per candidate is needed for isolation; the SCI subset (no
  `deftype`) bounds the synthesis grammar.

---

## 8. The conceptual bridge worth keeping (twisted SMC ↔ VOC)

Twisted SMC (Zhao et al., ICML 2024) introduces a **twist function** ψ_t(x_<t) that
estimates the **expected future potential** — the optimal twist *is the value function*
(soft-RL value) of a partial sequence. This is **precisely what the VOC controller
estimates** over the synthesis loop: the expected future evidence of continuing to
propose/deepen. So the controller ↔ twist correspondence is a real, citable conceptual
unification — not a stretch — and is the kind of result that makes the closure story more
than "nice Lisp." Honest caveat: twist payoff requires *learning* the twist (contrastive /
value-function regression); an honest paper either learns it or uses a myopic VOC surrogate
and says so.

**Two SMC layers — do not conflate:** (1) token-level SMC *inside* one program's synthesis
(Loula's setting — grammar = `Phi_eff`, already built in `grammar.cljs`/`bytes.cljs`); (2)
program-level SMC *over* candidate programs (synthesis-as-inference; the RRPS object level).
GenMLX uniquely **owns its LLM forward as a GF**, so proposal `q` and target `p` live in
the *same* GFI algebra — Loula/genlm bolt an external LM onto an external grammar; GenMLX
can express proposer, exact-evidence potential, and synthesized model in one algebra. Claim
exactly that unification — no more.

---

## 9. References

- Russell & Wefald 1991, *Principles of Metareasoning*, Artificial Intelligence 49:361-395.
- Hay & Russell, *Selecting Computations: Theory and Applications*, UAI 2012;
  *Metareasoning for MCTS*, EECS-2011-119.
- Callaway, Gul, Krueger, Griffiths, Lieder, *Learning to Select Computations*, UAI 2018
  (arXiv:1711.06892).
- Lieder & Griffiths, *Resource-rational analysis*, Behavioral and Brain Sciences 2020;43:e1.
- De Sabbata et al., *Rational Metareasoning for LLMs*, arXiv:2410.05563.
- Saad, Cusumano-Towner, Schaechtle, Rinard, Mansinghka, *Bayesian Synthesis of
  Probabilistic Programs for Automatic Data Modeling*, POPL 2019 (arXiv:1907.06249).
- Ellis et al., *DreamCoder*, PLDI 2021 (arXiv:2006.08381).
- Lake, Salakhutdinov, Tenenbaum, *Human-level concept learning through probabilistic
  program induction* (BPL), Science 2015.
- Lew, Zhi-Xuan, Grand, Mansinghka (Loula et al.), *Sequential Monte Carlo Steering of LLMs
  using Probabilistic Programs* (LLaMPPL), arXiv:2306.03081.
- Loula et al., *Syntactic and Semantic Control of LLMs via Sequential Monte Carlo*, ICLR
  2025 (arXiv:2504.13139). genlm-control: github.com/genlm/genlm-control.
- Zhao, Brekelmans, Makhzani, Grosse, *Probabilistic Inference in Language Models via
  Twisted SMC*, ICML 2024 (arXiv:2404.17546).
- Talts, Betancourt, Simpson, Vehtari, Gelman, *Validating Bayesian Inference Algorithms
  with Simulation-Based Calibration* (SBC), arXiv:1804.06788.
- Grand et al., *Modeling Boundedly Rational Agents with Latent Inference Budgets* (L-IBM),
  arXiv:2312.04030.

---

*Provenance: workflow run `wgz54tgog` (2026-06-16), 26 agents, ~2.3M tokens. Source
citations verified against the tree at `de0aca2`/`858ed05`. External facts (nbb 1.4.208
runtime macros; MLX thread-safety) empirically checked / sourced to upstream issues.*
