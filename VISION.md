# GenMLX Vision: From Orchestrator to Compiler

## The Stack

GenMLX is ~10,800 lines of ClojureScript implementing the full Generative Function
Interface (GFI) on GPU via MLX. Three properties of this stack, taken together,
enable something no other PPL can do:

1. **ClojureScript is a Lisp.** Macros see model source code as data. Program
   analysis and code generation happen at macro-expansion time — no separate
   compiler infrastructure needed.

2. **MLX is lazy.** Operations build a computation graph before executing.
   The more you defer evaluation, the more MLX can fuse. Laziness is implicit
   whole-program compilation.

3. **Handlers are pure middleware.** The handler is `(state, addr, dist) →
   (value, state')` with no side effects. Analytical solvers, compilation
   passes, and interpretation all compose as middleware layers around the
   same unchanged model code.

The host language is just an orchestrator — it builds the graph, MLX executes it.
Today, the orchestrator does too much work. The vision: push everything into the
graph until ClojureScript builds one graph and then sits idle.

---

## The Levels

Each level describes how much of the total computation lives in a single MLX
lazy graph. Each level subsumes the previous.

```
Level 0 (done):   Model body is one graph, inference loop is host-driven
Level 1 (done):   Compiled gen fn — model compiles to single Metal dispatch
Level 2 (done):   Compiled inference — full SMC/MCMC sweep is one graph
Level 3 (done):   Auto-analytical — macro detects structure, eliminates sampling
Level 3.5 (done): Extended analytical — full GFI coverage, combinators, multivariate
Level 4 (next):   Single fused graph — model + inference + optimization
Level 5 (future): Cognitive architecture — LLMs as generative functions, theory search
```

---

## Level 0: Shape-Based GPU Batching (DONE)

**What it is:** Run the model body once with `[N]`-shaped arrays instead of
N independent traces. MLX broadcasting handles all batched arithmetic. No vmap,
no function transformation.

**What's in the graph:** One forward pass of the model for all N particles.
The inference loop (resampling, weight normalization, iteration) is host-driven.

**Status:** Complete. This is the current state of GenMLX.

### Completed work

| Item | Result |
|------|--------|
| Shape-based vectorization (vsimulate/vgenerate) | 10-100x over scalar |
| GPU resampling (cumsum + broadcasting) | O(N^2) mem, fine for N<=20K |
| Structured state resampling | Walk map, take-idx each array |
| Default vectorized IS/ADEV | 352-1520x IS speedup |
| Batched combinators (Unfold, Switch, Scan, Mix) | 19-80x over scalar loops |
| Compiled unfold + compiled particle filter | 6-15x over batched variants |
| Differentiable inference (grad through IS) | Automatic parameter learning |
| Analytical middleware (Kalman, EKF, HMM, conjugate) | Exact inference for substructure |
| Fisher information / Laplace approximation | Exact model comparison |
| PMCMC (Particle Gibbs, PMMH) | Joint parameter + state estimation |

### What Level 0 can't do

Every `mx/` call from ClojureScript crosses the FFI boundary. A model with 50
trace sites and 10 ops each makes 500 FFI calls per particle per step. MLX's
lazy graph amortizes GPU dispatch, but the FFI call overhead is pure host cost.
The inference loop (resample, iterate, check convergence) forces materialization
at every step, breaking the graph into fragments.

---

## Level 1: Compiled Gen Functions

**What it is:** The `gen` macro analyzes model source code at macro-expansion
time and emits a compiled path — a single MLX-compiled function that replaces
hundreds of FFI calls with one Metal dispatch.

**What's in the graph:** The entire model forward pass as a fused kernel.
The inference loop is still host-driven, but each step is a single call.

### Strategy: Macro-time analysis, not runtime tracing

The existing VISION.md proposed runtime tracing ("trace once, discover schema,
compile"). The Lisp approach is more powerful: the `gen` macro already sees the
model body as data. Analyze it at compile time.

```clojure
;; The gen macro sees this code as a data structure:
(gen [x]
  (let [slope     (trace :slope (dist/gaussian 0 10))
        intercept (trace :intercept (dist/gaussian 0 10))]
    (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1))
    slope))

;; At macro-expansion time:
;; 1. Extract trace sites: [:slope, :intercept, :y]
;; 2. Extract distribution types: [gaussian, gaussian, gaussian]
;; 3. Build dependency graph: :y depends on :slope, :intercept
;; 4. Emit BOTH the flexible handler version AND a compiled version
;; 5. The compiled version inlines log-probs as flat MLX ops
```

### Milestones

**L1-M1: Schema extraction from gen bodies.**
The `gen` macro extracts trace addresses, distribution types, and dependency
structure from the model body at compile time. Emits metadata alongside the
standard handler-based code. No behavior change — just analysis.

**L1-M2: Compiled simulate/generate for static models.**
For models where all trace sites are statically known (no data-dependent
branching), emit a compiled function that does sampling + scoring as flat
tensor operations. Distribution-specific noise transforms (Gaussian: `noise *
std + mean`, etc.). Single `mx/compile-fn` call. Full GFI bridge
(simulate, generate, update via compiled path).

**L1-M3: Partial compilation for dynamic models.**
Models with data-dependent branching: compile the static subgraph, interpret
only the dynamic parts. If 9 of 10 trace sites are static, compile 9 and
drop to handler for 1.

**L1-M4: Automatic branch rewriting.**
Recognize `if/else` over trace sites with the same address and rewrite to
`mx/where` — making branches branchless. This is what the `switch` combinator
does; the compiler automates the transformation.

**L1-M5: Combinator-aware compilation.**
The compiler understands combinator structure:
- `unfold` → compiled scan (single dispatch for T steps)
- `switch` → compute all branches, `mx/where` to select
- `mix` → pre-compile each component, weight at runtime
- `map` → reshape to [P*K] flat batch

Each combinator encodes a compilation strategy. A model built from
`unfold(switch(mix(...)))` compiles by composing strategies for each layer.

### Why this is uniquely GenMLX

Neither Gen.jl nor GenJAX can do macro-time program analysis. Julia macros
exist but Gen.jl doesn't use them for compilation — it has a separate static
DSL. Python has no macros at all; GenJAX relies on JAX's runtime tracing.
ClojureScript macros operate on code-as-data at compile time, which is strictly
more powerful than runtime tracing: you see the source structure, not just one
execution path.

---

## Level 2: Compiled Inference Sweeps

**What it is:** The inference algorithm itself — SMC steps, MCMC transitions,
resampling, weight normalization — becomes part of the MLX graph. The entire
inference sweep from initialization to final estimate is one lazy computation.

**What's in the graph:** Model + inference algorithm + all iterations. Host
language builds the graph once and calls `mx/eval!` once.

### The key obstacle: data-dependent control flow

Resampling in SMC depends on weights. MH accept/reject depends on the score
ratio. These create host-side decision points that force materialization,
breaking the graph.

### The solution: differentiable alternatives

Replace hard decisions with differentiable approximations that stay in the graph:

- **Differentiable resampling** — Gumbel-top-k, optimal transport, or soft
  systematic resampling. Particle selection becomes a continuous operation.
- **Soft MH** — Replace hard accept/reject with a continuous interpolation
  weighted by the acceptance probability. Or pre-generate all uniform random
  numbers and express accept/reject as `mx/where`.
- **Fixed-randomness sweeps** — Pre-generate all random noise (keys, uniforms)
  before the sweep. The entire computation becomes deterministic given the
  noise — pure graph, no decisions.

### Milestones

**L2-M1: Pre-generated randomness for SMC.**
Generate all PRNG keys for T steps x N particles upfront. Pass them into the
sweep as data. Resampling uses pre-generated uniform noise with stratified
indices. The sweep becomes a pure `reduce` over timesteps — no host-side
randomness decisions.

**L2-M2: Lazy SMC sweep.**
Build the entire SMC sweep (init, T steps of extend + weight + resample) as
one lazy graph. Single `mx/eval!` at the end. Verify correctness against
the current eager implementation.

**L2-M3: Differentiable resampling.**
Implement Gumbel-top-k or optimal transport resampling as pure MLX operations.
This replaces the weight-dependent host-side branching with a continuous
in-graph operation. Enables gradients through the full SMC sweep.

**L2-M4: Compiled MCMC chains.**
Express an MH chain as a compiled scan: pre-generate proposals and acceptance
uniforms, `mx/where` to accept/reject. The entire chain from initialization
to final samples is one graph. Compose with compiled gen fns from Level 1.

**L2-M5: Gradient through full inference.**
With differentiable resampling + compiled sweeps, compute gradients of the
final log-marginal-likelihood with respect to model parameters through the
entire inference procedure. This is end-to-end differentiable probabilistic
programming.

### Unified memory advantage

This is where Apple Silicon's unified memory becomes a structural advantage
over CUDA. GenJAX on CUDA pays a transfer cost every time the host reads a
weight or makes a decision. GenMLX on unified memory pays zero. But at Level 2,
this advantage is even stronger: because we're keeping everything in one graph,
there are literally zero transfers. CUDA can't match this even with aggressive
compilation, because XLA still materializes between traced functions.

---

## Level 3: Automatic Analytical Elimination (DONE)

**What it is:** The `gen` macro analyzes model structure and automatically
applies analytical middleware — eliminating sampling entirely for substructure
that has closed-form solutions. The system infers more and samples less.

**What's in the graph:** Only the irreducible stochastic computation.
Everything with a closed form has been algebraically eliminated. The graph
is smaller, the variance is lower, the inference is faster.

**Status:** Complete. 426 tests across 6 new files, 7 investigation gates passed.

### How it works

The user writes standard `gen` functions with standard distributions. At
construction time, the system automatically:

1. **Detects conjugate pairs** — scans trace site dependencies to find
   known conjugate families (Normal-Normal, Beta-Bernoulli, Gamma-Poisson,
   Gamma-Exponential, Dirichlet-Categorical)
2. **Classifies dependency types** — determines if the link between prior
   and likelihood is `:direct`, `:affine`, or `:nonlinear`
3. **Builds analytical handlers** — address-based dispatch that intercepts
   `p/generate` at conjugate sites, computing exact marginal likelihoods
   instead of sampling
4. **Wires handlers automatically** — no annotations, no manual middleware
   composition. The `gen` macro does everything.

```clojure
;; User writes exactly this — no annotations:
(def model
  (gen [x]
    (let [mu    (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/gamma 2 1))]
      (trace :y (dist/gaussian mu 1)))))

;; System detects: :mu → :y is Normal-Normal conjugate
;; p/generate analytically marginalizes :mu
;; Only :sigma needs sampling
;; Weight = exact marginal log-likelihood (zero variance)
```

### Architecture

**Address-based dispatch, not distribution-type dispatch.** L3 intercepts
standard distributions (`dist/gaussian`, `dist/beta-dist`, etc.) at specific
trace addresses detected as conjugate pairs. This means existing model code
works unchanged — no special distribution types needed.

**Score accounting:**
- Prior sites (marginalized): no score/weight contribution
- Observation sites (constrained): marginal LL added to both `:score` and `:weight`
- Observation sites (unconstrained): fallthrough to standard handler

**Fallthrough design:** Non-conjugate sites, unconstrained observations, and
dynamic models all fall through to the standard L2/L1/L0 handler paths.
A model that is 70% conjugate gets 70% eliminated and 30% sampled.

### Completed work

| Component | File | Lines |
|-----------|------|-------|
| Conjugacy detection (5 families) | `conjugacy.cljs` | 165 |
| Affine expression analysis | `affine.cljs` | 379 |
| Dependency graph + d-separation | `dep_graph.cljs` | 262 |
| Graph rewriting engine (3 rule types) | `rewrite.cljs` | 225 |
| Address-based analytical handlers | `inference/auto_analytical.cljs` | 393 |
| Auto-wiring in DynamicGF | `dynamic.cljs` | +133 |

**Rewrite rules** (applied in priority order):
1. **KalmanRule** — collapse linear-Gaussian chains via Kalman filter
2. **ConjugacyRule** — eliminate conjugate prior via marginalization
3. **RaoBlackwellRule** — sample from posterior mean (variance reduction)

### Benchmark results

Evaluation benchmark (`l3_evaluation_benchmark.cljs`) comparing L3 analytical
elimination against L2 standard prior-proposal IS:

**Observation scaling** (Normal-Normal, prior std=10):

| Obs | L3 log-ML | L2 IS (200 particles) | L2 ESS | ESS/N |
|-----|-----------|----------------------|--------|-------|
| 5 | -7.708 (exact) | -7.93 +/- 0.40 | 11.4 | 5.7% |
| 10 | -12.649 (exact) | -12.92 +/- 0.41 | 7.8 | 3.9% |
| 20 | -22.184 (exact) | -22.50 +/- 0.41 | 5.2 | 2.6% |
| 50 | -50.211 (exact) | -50.54 +/- 0.42 | 3.1 | 1.6% |

L3 is exact at every scale. L2 ESS degrades as observations increase.

**Multi-group Rao-Blackwellization** (3 NN groups + 2 non-conjugate params,
200 particles x 15 trials):

| | L3 (3/5 dims eliminated) | L2 (all 5 dims sampled) |
|---|---|---|
| log-ML std | **0.44** | 14.73 |
| ESS | **7.7** / 200 | 1.1 / 200 |

**33.5x lower variance**, **7.2x higher ESS**. L2 is essentially collapsed
(ESS=1.1); L3 still functions by eliminating the conjugate substructure.

### What Level 3 can't do (addressed by Level 3.5)

- ~~**MCMC:** Auto-handlers intercept `p/generate` and `p/assess` only~~ → L3.5 WP-0
- ~~**Combinators:** No combinator-aware conjugacy detection~~ → L3.5 WP-2
- ~~**Multivariate:** Only 1D conjugacy~~ → L3.5 WP-3
- **Non-static models:** Dynamic addresses (loops, data-dependent branching)
  prevent conjugacy detection. (Remains a limitation.)

---

## Level 3.5: Extended Analytical Elimination (DONE)

**What it is:** Extends L3's analytical elimination across the full GFI —
`p/regenerate` and `p/assess` now benefit from auto-handlers alongside
`p/generate`. Adds combinator-aware conjugacy, multivariate (MVN-MVN)
conjugacy, and reduced-dimension score functions for compiled MCMC.

**Status:** Complete. 150 tests across 5 test files, 5 investigation gates passed.

### How it extends Level 3

L3 proved that zero-annotation conjugacy detection works. L3.5 extends coverage:

1. **Regenerate auto-handlers (WP-0):** MCMC methods using `p/regenerate` now
   benefit from analytical elimination. Score-only semantics (no weight
   contribution) — the prior site's score is updated analytically, obs sites
   contribute marginal LL to score. 8.2% dispatch overhead (within 10% budget).

2. **Assess auto-handlers (WP-1):** `p/assess` computes exact marginal
   log-likelihood for conjugate substructure. Enables model comparison and
   scoring without sampling.

3. **Combinator-aware conjugacy (WP-2):** Kernel-internal conjugacy works for
   free — combinators like `unfold`, `map`, and `scan` call `p/generate`
   internally, which triggers auto-handlers. Cross-boundary conjugacy
   (prior in outer, obs in combinator) deferred to L4.

4. **Multivariate conjugacy (WP-3):** MVN-MVN conjugate family with Kalman gain
   form using `mx/solve` (numerically stable, no explicit matrix inverse).
   Scales with prior dimension.

5. **Score function integration (WP-4):** `make-conjugate-aware-score-fn`
   produces reduced-dimension score functions that exclude eliminated addresses.
   Compiled MCMC operates only on residual stochastic dimensions.

### Architecture decisions

- **Score-only regenerate semantics:** Regenerate weight = (new-score - old-score).
  Auto-handlers contribute to score, not weight. This preserves MH correctness.
- **O(1) dispatch guard:** Precomputed `:auto-regenerate-transition` on schema
  avoids per-step scanning overhead.
- **Kalman gain form for MVN:** `mx/solve` instead of 3x `mx/inv` — O(n^3) but
  better constant factor and numerical stability.
- **Unified Kalman handler core:** Generate and regenerate Kalman handlers share
  `make-kalman-handlers-core` with mode parameter, eliminating ~80 lines of
  duplication.

---

## Level 4: Single Fused Graph

**What it is:** Model specification, inference algorithm, and parameter
optimization are all expressed as one lazy MLX computation graph.
ClojureScript builds the graph, then sits idle while Metal executes.

**What's in the graph:** Everything. The model, the analytical eliminations,
the compiled inference sweep, the gradient computation, the optimizer step.
One `mx/eval!`.

This is the natural culmination of the compilation ladder. L0 batched the
model, L1 compiled it, L2 compiled inference, L3/3.5 eliminated analytically
— L4 fuses it all into one dispatch. The PPL becomes a compiler.

### What this looks like

```clojure
;; User writes:
(def model (gen [...] ...))
(def result (fit model data {:method :smc :particles 1000 :learn [:theta]}))

;; Under the hood:
;; 1. gen macro analyzes model structure (Level 3)
;; 2. Conjugate pairs detected, middleware auto-composed
;; 3. Remaining stochastic sites compiled (Level 1)
;; 4. SMC sweep compiled with differentiable resampling (Level 2)
;; 5. Gradient of log-ML w.r.t. :theta computed through the full sweep
;; 6. Adam step on :theta
;; 7. Steps 4-6 repeated for E epochs
;; 8. ALL of steps 4-7 is ONE lazy graph
;; 9. Single mx/eval! ships it to Metal
```

### Milestones

**L4-M1: Compiled optimization loop.**
Wrap the gradient computation + Adam update in `mx/compile-fn`. The inner
loop (compute loss, compute gradient, update parameters) is a single compiled
call per iteration. The epoch loop is still host-driven but each iteration is
minimal overhead.

**L4-M2: Fused inference + optimization.**
Compose compiled inference sweeps (Level 2) with compiled optimization
(L4-M1). The entire "run SMC, compute log-ML gradient, update params" cycle
is one graph. Multiple cycles can be unrolled into a single graph if MLX
compile supports it.

**L4-M3: Automatic method selection.**
Given a model and data, the system chooses the inference strategy:
- All conjugate → exact inference, no sampling
- Linear-Gaussian temporal → Kalman filter
- Nonlinear temporal → EKF or compiled SMC
- Static with few latents → compiled MCMC
- High-dimensional → compiled VI/ADEV

Selection based on the structure metadata from Level 3/3.5's analysis.
The user calls `(fit model data)` and the system figures out the rest.

---

## Level 5: Cognitive Architecture

**What it is:** The GFI becomes a cognitive primitive. Language models,
theory search, and adaptive reasoning are first-class generative functions
— composable with Gaussians, Kalman filters, neural nets, and all existing
combinators through the same interface.

L0-L4 make probabilistic programs run fast. L5 makes them think.

### Why this is possible

An LLM has a well-defined `log p(text)` — the sum of per-token log-probs.
That means it satisfies the GFI contract:

- **`simulate`**: sample text from the LLM (normal generation)
- **`assess`**: compute `log p(text)` for given text (the score)
- **`generate` with constraints**: conditioned generation (infilling/constrained decoding)

Once an LLM is a generative function, it composes with everything GenMLX
already has. Language becomes just another random variable you can condition
on — no different from a Gaussian or a Bernoulli.

```clojure
(def clinical-model
  (gen [symptoms lab-results]
    ;; LANGUAGE variable — diagnosis in words
    (let [diagnosis (trace :diagnosis
                     (llm-dist (str "Patient presents with " symptoms)))
          ;; CONTINUOUS variable — predicted labs from diagnosis
          predicted (trace :predicted (physiology-model diagnosis))]
      ;; OBSERVATION — actual lab results constrain both
      (trace :labs (dist/gaussian predicted noise) lab-results)
      diagnosis)))

;; Inference finds diagnoses that are BOTH linguistically coherent
;; (high LLM score) AND consistent with the lab data (high Gaussian score)
```

The LLM doesn't know it's inside a gen function. GenMLX doesn't care that
the distribution is an API call. The causal structure lives entirely in the
gen function's control flow. The LLM is a black-box conditional distribution
`p(text | prompt)`, just as `dist/gaussian` is `p(x | mean, std)`.

All existing combinators work naturally:

- **Map**: parallel independent LLM calls (N explanations for N data points)
- **Unfold**: sequential reasoning chains (chain-of-thought with particle filtering)
- **Switch**: discrete model selection (LLM vs physics model, posterior decides)
- **Scan**: dialogue modeling (multi-turn with full history as trace)
- **Mix**: LLM ensembles with learned mixture weights
- **Recurse**: tree-structured decomposition of complex problems
- **Dimap**: prompt engineering as pure function composition

### Milestones

**L5-M1: LLM as generative function.**
Wrap an LLM API (or local model via mlx-lm) as a `Distribution` with
proper `sample` and `log-prob`. Handle batching (multiple particles =
multiple API calls or batched local inference). Token-level log-prob
accumulation for `assess`. Constrained generation for `generate`.

This is the infrastructure milestone. Once done, all the composition
patterns above work immediately — they're just gen functions using a
new distribution type.

**L5-M2: Theory search.**
Express theory discovery as a meta-generative function that samples
architecture (which mechanisms to include) and evaluates fit. The space of
combinator compositions is the hypothesis space; the posterior concentrates
on the best-fitting theory. GPU-parallel evaluation of all candidate
architectures via compiled `switch`.

```clojure
(def theory-search
  (gen [data]
    (let [has-mechanism-a (trace :mech-a (dist/bernoulli 0.5))
          has-mechanism-b (trace :mech-b (dist/bernoulli 0.5))
          model (compose-model has-mechanism-a has-mechanism-b)]
      (splice :fit (model data)))))
;; Posterior over :mech-a, :mech-b = which theory fits the data
```

With L5-M1, the mechanisms themselves can be LLM-generated code,
scored against real data. The system searches over both structure
(which components) and implementation (what each component does).

---

## The Competitive Picture

### vs GenJAX (JAX/XLA on CUDA)

| | GenJAX | GenMLX at Level 0 | GenMLX at Level 2+ |
|---|---|---|---|
| Compilation | Full program via XLA trace | None (FFI per op) | Lazy graph = implicit compilation |
| Memory model | CUDA (explicit transfers) | Unified (zero transfers) | Unified (structural advantage) |
| Vectorization | vmap → compiled kernel | Shape broadcasting | Shape broadcasting (equivalent) |
| Host interaction | Zero during compiled execution | FFI per op | Zero (full graph) |
| Analytical inference | Manual | Composable middleware | **Auto-detected + composed** |
| Program analysis | Runtime tracing only | **Macro-time (code as data)** | **Macro-time** |

At Level 0, GenMLX is 2-5x slower than GenJAX for compiled workloads.
At Level 2, the gap closes to parity on Apple Silicon (unified memory wins).
At Level 3/3.5, GenMLX does things GenJAX cannot: automatic analytical
elimination across all GFI operations, multivariate conjugacy, macro-time
structure detection, and composable middleware stacking.
At Level 5, GenMLX enters territory no existing PPL occupies: LLMs as
first-class generative functions composable with continuous and discrete
models through a unified probabilistic interface.

### The honest bottleneck

**Model expressivity vs compilability.** The more dynamic the model (stochastic
control flow, variable-length traces, recursion), the less fits in one static
graph. Gen's power is handling arbitrary dynamic models. The art is keeping
full expressivity as a fallback while making the common cases fly.

The middleware architecture is perfectly positioned for this: each layer
handles what it can and passes the rest through. A model that is 70%
analytically tractable and 30% dynamic gets 70% eliminated and 30% sampled.
No all-or-nothing tradeoff.

---

## Architecture Summary

```
ClojureScript macros  →  static program analysis + code generation
Middleware handlers   →  algebraic graph rewriting engine
MLX laziness          →  implicit whole-program compilation
Unified memory        →  zero-cost host/device interleaving
Bun                   →  fast graph construction, then idle

gen macro sees code as data
  → extracts structure (trace sites, dependencies, conjugacy)
  → emits compiled path + analytical middleware
  → model body becomes one fused MLX kernel

inference algorithm expressed as lazy MLX ops
  → differentiable resampling, soft accept/reject
  → entire sweep is one graph
  → gradient flows through model + inference

result: ClojureScript builds one graph, calls mx/eval! once, Metal executes
```

The PPL becomes a compiler from probabilistic programs to optimal GPU
execution plans, where "compilation" emerges from the natural interaction of
Lisp macros, functional middleware, and lazy evaluation. No compiler
infrastructure needed — the stack is the compiler.

At Level 5, the same architecture extends to cognitive tasks: LLMs slot into
the GFI as distributions over language, composable with everything above
through the same `simulate`/`generate`/`assess` contract.

---

## Status

### Level 0: COMPLETE

- Full shape-based GPU batching, all combinators, all inference algorithms
- Analytical middleware: Kalman, EKF (1D + ND), HMM forward, conjugate priors
- Compiled combinators: unfold, particle filter
- Differentiable inference, Fisher information, PMCMC
- Formal verification: 10 GFI contracts, Lean mechanization (Phase A)

### Level 1: COMPLETE

- L1-M1: Schema extraction from gen bodies (174/174 tests)
- L1-M2: Compiled simulate/generate for static models (82/82 tests)
- L1-M3: Partial compilation for dynamic models (92/92 tests)
- L1-M4: Automatic branch rewriting via mx/where
- L1-M5: Combinator-aware compilation — unfold, scan, map, switch, mix (90/90 tests)
- Performance: 3-5x simulate speedup, compiled gen fns as single Metal dispatch

### Level 2: COMPLETE

- L2-M1: Pre-generated randomness for SMC and MCMC sweeps
- L2-M2: Compiled SMC — bootstrap particle filter, 77.5x speedup over handler
- L2-M3: Differentiable resampling — Gumbel-top-k (hard) + Gumbel-softmax (differentiable)
- L2-M4: Compiled MCMC chains with tensor-native score, 15.6x speedup
- L2-M5: Gradient through full inference — MH chains and SMC sweeps, sublinear memory
- New infrastructure: TensorTrace, TensorChoiceMap, tensor-native score function
- Bonus: fixed pre-existing PRNG seed bug affecting all multi-particle inference
- 881+ tests green across L0/L1/L2, all 6 investigation gates passed

Note: L2-M2 materializes at step boundaries for resampling (host-side systematic
resampling breaks the graph). With Gumbel-softmax resampling, larger chunks can
stay lazy, but true single-mx/eval! SMC sweep is not yet achieved.

### Level 3: COMPLETE

- WP-0: Conjugacy detection — 5 families (NN, BB, GP, GE, DC), dependency classification
- WP-1: Address-based analytical handlers — reuses conjugate.cljs math, address dispatch
- WP-2: Auto-wiring in DynamicGF — zero-annotation conjugate elimination in p/generate
- WP-3: Affine expression analysis + auto-Kalman — linear-Gaussian chain detection
- WP-4: Dependency graph + conditional independence — d-separation, Markov blankets
- WP-5: Algebraic graph rewriting — 3 rule types (Kalman, Conjugacy, RaoBlackwell)
- Memory fix: auto-sweep dead arrays in p/simulate and p/generate (prevents Metal exhaustion)
- 426 tests across 6 test files, 7 investigation gates passed
- Benchmark: 33.5x variance reduction, 7.2x ESS improvement on multi-group mixed models

### Level 3.5: COMPLETE

- WP-0: Regenerate auto-handlers — MCMC benefits from analytical elimination (8.2% overhead)
- WP-1: Assess auto-handlers — exact marginal LL scoring for conjugate substructure
- WP-2: Combinator-aware conjugacy — kernel-internal conjugacy works via p/generate dispatch
- WP-3: Multivariate conjugacy — MVN-MVN with Kalman gain form (mx/solve, no mx/inv)
- WP-4: Score function integration — reduced-dimension score fns for compiled MCMC
- Unified Kalman handler core (generate/regenerate), -98 lines cleanup
- 150 tests across 5 test files, 5 investigation gates passed

### Next

- Level 4: Single fused graph — compiled optimization, fused inference, automatic method selection
- Level 5: Cognitive architecture — LLM as generative function, theory search
