# The Programmer as a GenMLX Model — REPL-driven, resource-rational program synthesis

*Design doc behind bean **genmlx-d4q4**. The north star for advanced GenMLX model synthesis:
model an experienced Clojure programmer who builds probabilistic models REPL-first — propose a
small form, eval it, read the feedback, revise, compose upward — **as a GenMLX generative
function**, so that program synthesis becomes *inference over the programmer-GF*. Empirical
motivation: `docs/cljs-coder-loop-results.md` §6–7 (whole-program best-of-K cliffs for BOTH the
0.8B and the 35B — it is the one-shot *interface*, not capability).*

---

## 1. The thesis: synthesis = inference over a programmer-GF

If the programmer is a GF, the search, the metareasoner, the LLMs, and the oracle stop being
separate machinery and become *parts of one generative function*. The deep structure is a
**fixpoint — a GenMLX model that synthesizes GenMLX models** — with GenMLX inference appearing at
two levels:

- **Object model** — the thing being built (linreg / hierarchy / GMM / Kalman). Trace = the
  data-generating latents + observations.
- **Meta model = the programmer** — a GF whose **latent is "a program"** and whose **observation
  is "the data is well-explained by that program."** Its trace = the *development trajectory*.
  Inference over it (targeting high model evidence) **is** program synthesis.

This is the natural consequence of the project's standing principle *an agent is a GF and
inference is pluggable* (`genmlx.agents`): the programmer is an agent whose environment is the
**REPL + the GenMLX oracle**; its state is the current partial program; its actions are edits;
its observations are eval feedback + evidence; its policy is the LLM proposers + the metareasoner.

## 2. The traces — the development trajectory

The programmer-GF's trace is the REPL session itself: an `Unfold`/`Scan` over construction steps,
each step a few trace sites. Illustrative skeleton (GenMLX `gen`/`trace`, aspirational):

```clojure
(def programmer
  (gen [data goal]
    ;; SYSTEM 2 — the architect sketches a decomposition (rare, big model)
    (let [plan (trace :plan (planner-llm (describe data goal)))]
      (loop [prog (empty-model), t 0]
        (let [act (trace [t :edit] (controller-policy prog plan))]    ; metareason: which edit?
          (if (= act :stop)                                            ; resource-rational stop
            prog
            (let [tier (trace [t :which] (escalate? prog act))         ; :fast 0.8b | :slow big ← System 1/2
                  form (trace [t :form]  (propose tier act prog        ; the concrete cljs form…
                                           {:grammar genmlx-instaparse}))  ; …constrained valid
                  prog' (apply-edit prog form)
                  chk   (check prog' data)    ; ← SELF-CHECK in-process: parse / schema / eval / evidence
                  keep? (trace [t :accept] (accept-policy chk))]       ; condition on verified feedback
              (recur (if keep? prog' prog) (inc t)))))))))
```

**Trace sites (random choices) = the programmer's decisions:** `:plan`; per step `:edit`
(add-latent / add-obs / wrap-combinator / refactor / fix / stop), `:which` (fast vs slow),
`:form` (the code), `:accept` (keep / revise / backtrack). **The observations it conditions on**
are the `check` results — parse status, eval value/error, schema validity, partial-model
**evidence**. **The weight/likelihood** of the whole trace is the oracle evidence of the
synthesized model, so `generate`/SMC over `programmer` given `data` *automatically targets
high-evidence programs*; the intermediate evidences are the SMC intermediate weights.

**The second meaning of "traces" — self-generated training data.** The GFI trace of a
*successful* run records `(partial-program, feedback) → edit-chosen` at every step. Run inference
over the programmer-GF (big model + oracle as teacher), keep the high-evidence trajectories, and
their traces **are** the `(partial + feedback → next edit)` SFT corpus for the fast proposer. The
programmer generates its own oracle-filtered training data *by being a GF you can run inference
over*. GRPO then sharpens the proposer at its `:form` sites with evidence as the verifiable reward
(the GFI-reward GRPO infra already exists, genmlx-ugkv/2ctu).

## 3. The role of each LLM — distributions at trace sites

Each LLM is a `DynamicGF` plugged in at a site, so the whole GFI applies:

| faculty | model | site(s) | cadence |
|---|---|---|---|
| **planner / architect** (decompose, pick structure) | big off-the-shelf (qwen3.5 → Coder-Next) | `:plan`, escalated `:form` | rare (System 2) |
| **proposer** (next form / local edit) | small fast specialist (0.8b-cljs) | high-frequency `:form` | very high (System 1) |
| **controller** (which edit / which tier / accept) | tiny model or learned softmax | `:edit` `:which` `:accept` | the metareasoner |
| **verifier** ("did it work / fit?") | **the GenMLX oracle — never an LLM** | the `check` node + the weight | every step, exact |

Fast/slow thinking is literally the `:which` trace site; the metareasoner is the *learnable
policy* over it. The search strategies become **GenMLX inference over `programmer`**, not bolted-on
heuristics: `simulate` = synthesize once; `generate` w/ constraints = "build a model that uses a
Kalman combinator" (condition `:plan`) or "finish this partial"; **SMC** = a *population of
programmers* exploring trajectories, resampled by partial evidence (best-of-K becomes a principled
particle filter over development paths); **regenerate/MCMC** = reconsider an earlier decision given
later feedback (backtracking *as inference*); the metareasoner's expected-utility lookahead reuses
the `genmlx.agents` machinery.

## 4. Self-checking — why it is a GF, not just an agent loop

`check` runs *inside the programmer's own body* — four complementary verifiers at four altitudes,
all in-process, and the fitness one is **exact math, not an LLM-judge**:

| level | checker | answers | substrate |
|---|---|---|---|
| **syntax** | instaparse / reader-as-grammar | well-formed GenMLX form? | `genmlx.llm.grammar`/`bytes`/`codegen` |
| **semantics** | **malli** + GenMLX schema | well-formed *model*? (covers data, types/arities, no delta-hack) | `schema.cljs` + malli |
| **behavior** | **SCI eval** | does it run / error? | `codegen.eval` (in-process) |
| **fit** | **the oracle** (`score-model*`) | does it explain the data? (the likelihood) | analytical evidence / IS |

The programmer is a **self-conditioning generative model**: its later choices (`:accept`, the next
`:edit`) are conditioned on the *verified* results of its earlier choices. It **contains its own
verifier**, and that verifier is *exact* — which is exactly why this beats a generic LLM agent loop
(whose "self-critique" is another unreliable generation). It also makes the trace *trustworthy as
training data*: only verified-good trajectories become SFT corpus.

### instaparse — the *support* of the proposal distribution
instaparse (with the reader-as-grammar) is the grammar constraint at `:form`, composed via the same
`dispatch/with-handler` middleware as analytical conditioning. It gives the proposer support **only
over syntactically-valid GenMLX** — `(trace :kw (dist/NAME args…))` with real distributions and real
`mx` ops — so it **kills a whole class of the §8 failures at generation time** (malformed parens,
hallucinated `mx/0`/`mx/infer`) instead of at eval. Two tiers: edamame = "valid cljs," instaparse =
"valid GenMLX *model* form." In GFI terms instaparse **types the action space**: "propose code"
becomes "sample from the grammar" — sample-efficient and a clean typed move-set for the metareasoner.

### malli — the *semantic* schema + the programmer's introspection
Where instaparse is syntax, malli is meaning:
1. **The semantic gate as schemas.** "Every observed address is a trace site" (the covering check),
   "dist args have the right arity/type," "the body returns an observation map," "no `delta` hack" —
   these are malli schemas over the parsed model. malli is the `:schema-ok?` checker.
2. **The programmer's *mental model* of the partial program.** `schema.cljs` already extracts
   trace-sites / dist-types / dependency sets; as malli, that is the programmer's introspection —
   *what's covered, typed, missing, depends on what* — and it **drives the next `:edit`** ("schema
   says `:y2` is uncovered → add a `:y2` site"). The schema is the programmer's working memory.
3. **Schema-guided generation + typing the meta-trace.** malli can constrain the proposer
   (schema → generators / as a predicate), validate data conforms to the model's inferred shape, and
   even type the programmer-GF's own development trajectory (each step a well-formed move).

## 5. The nbb / SCI substrate — the homoiconic in-process loop (a real pillar)

GenMLX runs on **nbb (SCI — the Small Clojure Interpreter), not compiled cljs**. This is not
incidental; it is what makes the REPL loop a *function call* instead of a subprocess. The
load-bearing property is **homoiconic, no-compile, sandboxed, in-process eval of generated GenMLX
code in the same image as the MLX binding and the oracle**:

- **The REPL loop is in-process and cheap.** `propose form` (the form is *data* — an s-expression)
  → `SCI eval` → a live `DynamicGF` (a *value*) → `oracle score` — all in one image, no compile/IPC,
  no marshaling (the eval'd model shares the MxArray graph the oracle reads). This is what makes
  *hundreds* of per-step evals affordable inside a search.
- **Interpret-don't-compile actually *beats* JVM Clojure here.** A synthesis loop eval's thousands
  of throwaway candidate programs. SCI just walks the form — **no per-candidate compile latency and
  no class-leak**, where JVM `eval` would generate thousands of classes (metaspace/JIT churn). And
  SCI's interpretation overhead lands on the *cheap* part (building the lazy MLX graph — "graph
  construction is value manipulation"); the *expensive* work (MLX compute, LLM decode) is native, so
  the interpreter overhead is essentially free for model execution.
- **Safe eval of untrusted LLM code.** SCI is *sandboxed* — controlled namespaces/capabilities
  (`codegen.eval` already does this) — so the agent eval's LLM-generated code thousands of times
  safely; the eval environment is *itself data* the metareasoner can shape per step (expose only the
  GenMLX DSL + the learned library). Pair with the existing process-group-kill watchdog
  (`distill_sandbox.cljs`) for non-terminating candidates.
- **Live library / abstraction growth — the antidote to the §8 length-cliff.** New combinators /
  sub-model GFs (and, where new *syntax* helps, macros — `gen` itself is a `.cljc` macro SCI runs)
  that the agent factors from successful syntheses are **immediately available in the same image, no
  recompile**. The agent compresses recurring structure into reusable abstractions, shrinking the
  program length that caused the cliff (DreamCoder-style library learning, but live).

**On "live-reload itself" and macros — don't over-index.** The useful version of self-improvement is
NOT the agent rewriting its own code; it is (a) **inference** (better trajectories), (b)
**weight-space learning** at the proposer's sites (SFT/GRPO), and (c) **growing a live toolbox of
reusable abstractions**. For (c), the *primary* mechanism is **GF / combinator composition** —
learned sub-models are first-class *values* you `splice`, scoreable under the GFI, no macros needed
and cleaner. **Macros are secondary**: useful because the object-DSL is already macro-based and for
growing compact *notation* / sub-DSLs, which SCI lets you do live — reach for them only when you want
new syntax, not as the core lever.

**GRPO vs nbb — orthogonal, both needed, not substitutes.** Two kinds of "update": *weight-space*
(SFT/GRPO — slow, amortized, changes the proposer's instincts, learns the **policy**) and
*inference-time structural* (nbb/SCI — the live eval/check/compose loop + library growth, no
retraining). **GRPO learns *from* the traces the SCI loop *produces*; SCI *runs* the loop the better
proposer acts in.** nbb is needed for the loop to exist at all (and to generate the training traces);
GRPO amortizes the policy. nbb is not replaceable by GRPO. (Constraint to preserve across the
Thor/CUDA move: keep the SCI-eval-next-to-MLX property on the new runtime.)

**The deepest point.** Code is data (forms); the agent *generates* forms (LLM), *evals* them live
(SCI) into GFs (values), *scores* them (oracle, exact), *composes* them (GFI `splice`/`edit`), and
*grows its DSL* (live combinators/macros) — all in one image where "graph construction is value
manipulation." The programmer, the programs it writes, its toolbox, and the data are **all the same
kind of thing — values/forms in a live evaluable image.** That homoiconic uniformity is the "extreme
feature": Lisp's promise (the program is data the program can manipulate and run live), finally
pointed at probabilistic-program synthesis with an exact verifier.

## 6. Resource-rationality — the metareasoner is where it all meets

The programmer-GF is the single point where the three axes meet: **model** (the object program it
builds), **inference** (the SMC/MCMC search *and* the exact oracle it checks itself with), and
**control** (`genmlx.control`: the metareasoner choosing edits, tiers, particle budget, and when to
stop). Resource-rationality is the policy at the control sites: **one-shot vs build-up** (simple
models stay one-shot — the K=16 tie shows it works there), **fast↔slow escalation** (`:which`),
**how many particles** to spend on a step, and **anytime stopping** ("one cycle, more if time"). The
expensive model is invoked *once or a few times* (the `:plan` site + hard escalations); the exact
oracle does the discrimination — that is the resource-rational unlock (you never pay an LLM to verify).

## 7. Plan + the decisive early experiments

Phases (full detail in `docs/cljs-coder-loop-results.md` §7):
- **Phase 0 — done.** Exact oracle verifier; whole-program best-of-K (the particle engine); the
  cliff diagnosis (both models cliff → it is the interface).
- **Phase 1 — the REPL loop.** State = partial model; action = edit; eval (SCI) + partial evidence →
  feedback; revise. Big model as proposer first (de-risk capability). Reuses GFI `edit`/`CompositeEdit`,
  `codegen.eval`, the oracle.
- **Phase 2 — search over edits.** Particles + SMC/beam over construction steps; resource-rational
  allocation; stop on evidence plateau.
- **Phase 3 — specialize the proposer.** REPL-trace SFT + GRPO → the 0.8b becomes the cheap
  high-frequency proposer; escalate to the big model only on hard steps.
- **Phase 4 — the controller (the crown).** Metareasoning, anytime, resource-rational.
- **Phase 5 — enablers (parallel).** Fix/port the batched decoder (→ Thor/CUDA); deeply integrate the
  big off-the-shelf model GFI-native (genmlx-9uyg) so the *planner* also composes natively.

**Decisive early experiments (experiment-first, before committing):**
- **(A) Does partial-model evidence GUIDE the search?** Build the 4 advanced models incrementally by
  hand, score each partial — does evidence climb informatively as correct sites are added, or is it
  flat/noisy until complete? **The single load-bearing assumption** of REPL-with-evidence. Cheap,
  decisive. If it fails, the intermediate signal must be eval-success + structural progress instead.
- **(B) Minimal REPL loop solves an advanced model** that whole-program best-of-16 got 0/16 on.
- **(C) The decoder pivot.** Fix EOS-stop + the cross-call leak; re-measure — does the cheap proposer
  become genuinely cheap? This single fact decides the small model's role.

## 8. Why this is cutting-edge

Two camps that don't meet: LLM program synthesis (strong proposer, *weak* verifier — tests/LLM-judges)
and probabilistic-program inference (exact methods, *no* LLM). This is **REPL-driven program synthesis
where the verifier is exact Bayesian model evidence, the search is SMC over construction steps with
per-partial reward, the proposer is a resource-rationally-allocated mix of a cheap specialist and a
rarely-invoked expensive planner, and the whole programmer is a GF that checks itself and generates its
own training data — all in one homoiconic, live-evaluable image.** That combination is ahead of where
either camp is.

---
*Related: genmlx-d4q4 (this), genmlx-9uyg (Route B — the GFI-native [K]-particle substrate),
genmlx.agents (agent = GF), genmlx.control (metareasoning). Data: `~/genmlx-loop-artifacts/particle/`.*

## 9. Experiment (A) RESULT (2026-06-22) — partial-model evidence is a calibrated guide ✅

The load-bearing assumption holds. `scripts/partial_evidence_probe.cljs` builds each advanced
model as a ladder of hand-written intermediate states (crude→correct + variants), all covering the
same data, scored EXACTLY by the oracle:

```
linreg:  shared-mean σ2.5 -17.50 | shared-mean σ1 -23.59 (bad, punished) | linear σ3 -17.45
         | linear no-intercept σ1 -11.05 | linear+intercept σ1 -11.96
kalman:  iid -7.01 | AR 0.3 -6.39 | AR 0.9 -5.37        (none < weak < correct — smooth ramp)
hier:    single mean -24.00 | indep groups -15.79 (+8.2) | hierarchical -21.77
```

**Verdict: evidence is an informative, well-calibrated MODEL-SELECTION gradient** — not just "a
gradient." Structure-blind baselines are clearly worst; the structurally-necessary component earns
a large correct jump; genuinely-bad moves are punished (−23.6); and crucially it implements **Occam**
— it does NOT blindly reward more structure. The two apparent "anomalies" are the oracle being
*right*: `no-intercept` ≳ full model (the intercept costs a parameter it barely needs), and
`independent-groups` ≫ `hierarchical` for well-separated groups (the partial-pooling prior is
mismatched, so the hierarchical complexity is unwarranted by *this* data — my hand-labeled "ground
truth" was not evidence-optimal).

**Design consequences (this sharpens §6's controller):**
- The search target is **the simplest model the data warrants**, not a pre-specified textbook
  structure. So stepwise synthesis is naturally **"add structure while evidence climbs, stop when it
  plateaus/drops"** — self-terminating and resource-rational by construction. The controller's
  **stopping rule (evidence plateau) is doing real work**, not decoration.
- Evidence is a *selection* signal, so a single-axis greedy climb can stall (linreg `shared-σ2.5 ≈
  linear-σ3` until the noise also tightens at `linear-σ1`) — structure and parameters must move
  together. This argues for **search over edits (SMC/particles), not greedy repair**, and matches the
  realistic REPL move "evidence didn't move → my noise scale is off."
- The high-evidence basin is reachable with the crude models well below it — a particle search finds
  it. Data: `~/genmlx-loop-artifacts/particle/partial_evidence.json`.

## 10. Phase 1 — the kernel + experiment (B): the loop solves the cliff ✅

The Phase-1 kernel is built (`src/genmlx/world/synth.cljs`, native-free, 36/36
`test/genmlx/synth_test.cljs`): the **model spec** (a partial GenMLX program as data —
ordered latent + observation trace sites with argument *forms*), the form-level **edit
ops** (`add-latent` / `add-obs` / `set-args` / `set-mean` / `set-noise` /
`wrap-combinator` — all pure `spec → spec`, the homoiconic build-up of §5, *not* the
trace-level GFI `CompositeEdit`, which is a different layer), the four-level **check
node** (§4: `:parses?` reader / `:schema-ok?` schema+no-delta+returns-a-map /
`:covered?` schema-vs-data / `:evals?`+`:error` SCI / `:evidence`+`:method` the exact
oracle), and the greedy **driver** (`synthesize`: propose → check every candidate →
accept the best evidence-improving valid one → stop on plateau). The check node's
evidence is pinned against an *independent* closed-form Gaussian-Gaussian marginal.

**Experiment (B) — `scripts/synth_repl_probe.cljs` — GREEN, 3/3.** Starting from the
crudest covering rung of the experiment-A ladder, the loop accepts the **key structural
edit** (oracle-selected over a distractor and the alternative structure) and
self-terminates on all three exact-scoreable advanced models that whole-program
best-of-16 got **0/16** on (the teacher got 1/4 and that one fit at junk −35):

```
                 crude start → LOOP final   (exp-A struct. rung, σ=1)   path (2 accepted edits, then plateau)
linreg-coupled   −17.50      → −10.14        (−11.05)                   add slope (no-intercept) +1.94 ; shared-noise +5.42
kalman-chain     −7.01       → −5.05         (−5.37)                    AR(1)-couple coef 0.9 +1.65 ; proc-noise +0.31
hier-group-means −24.00      → −12.36        (−15.79)                   independent group means +8.22 ; shared-noise +3.43
```

The **structural jump** is the result: the crude model *lacks* the slope / the temporal
coupling / the group split, and the oracle's exact evidence selects it over the
distractor and (for hier) over the more-complex *hierarchical* alternative
(independent groups are the higher-evidence structure for these well-separated groups —
exp-A). This is the load-bearing claim made concrete: **a feedback loop with an exact
verifier turns a 0/16 whole-program problem into a sequence of locally-checkable edits.**

**Honest framing (what this does and does NOT show):**
- The proposer here is a **structured move-vocabulary, not an LLM** — deliberately, to
  *isolate* the loop-vs-cliff claim from generation noise. The driver is
  proposer-agnostic (an injected `(fn [spec feedback] → candidates)`), so wiring a real
  LLM proposer is a drop-in; **Phase 3 learns it**. So experiment B proves the loop +
  oracle, not (yet) that a cheap LLM is the proposer.
- The monotone evidence climb is **true by construction** (the driver only appends an
  improving step), so it is *not* itself evidence of success — the discriminating facts
  are that the **structural** edit was accepted and the loop self-terminated.
- The final lands slightly **above** the σ=1 reference rung because the loop also tunes
  a single **shared observation-noise** scale — that is **type-II maximum-marginal-
  likelihood (empirical Bayes) over one fixed hyperparameter**, not added model
  structure, so it is *not* directly comparable to the fixed-σ rung. The structural
  step, not the noise tuning, is the result.
- **Greedy commits to the first improving *structural* move** (e.g. linreg locks into
  the no-intercept branch — a greedy local optimum, *not* a global-Occam claim; a
  full-intercept model may score higher under *joint* structure+noise search). This is
  the predicted greedy limitation and is exactly **why Phase 2 is SMC/beam over edits** —
  a population explores both branches. Data:
  `~/genmlx-loop-artifacts/particle/synth_repl_probe.json`.

The kernel ships **five** form-level edit ops (`add-latent`/`add-obs`/`set-args`/
`set-mean`/`set-noise`); the named **`wrap-combinator`** move is *deferred* — folding
repeated sites into a `Map` combinator needs the synthesis SCI sandbox to expose the
combinators — so the kernel ships `homogeneous-obs?`, the pure predicate that *detects*
the fold opportunity, rather than a rewrite that would render an un-evaluable form.
