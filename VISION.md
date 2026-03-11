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
Level 1 (next):   Compiled gen fn — model compiles to single Metal dispatch
Level 2:          Compiled inference — full SMC/MCMC sweep is one graph
Level 3:          Auto-analytical — macro detects structure, eliminates sampling
Level 4:          Everything is one graph — model + inference + optimization
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

## Level 3: Automatic Analytical Elimination

**What it is:** The `gen` macro analyzes model structure and automatically
applies analytical middleware — eliminating sampling entirely for substructure
that has closed-form solutions. The system infers more and samples less.

**What's in the graph:** Only the irreducible stochastic computation.
Everything with a closed form has been algebraically eliminated. The graph
is smaller, the variance is lower, the inference is faster.

### The middleware is already built

GenMLX already has composable analytical middleware:
- Kalman filter (linear-Gaussian latent dynamics)
- EKF (nonlinear latent dynamics via linearization)
- HMM forward algorithm (discrete latent states)
- Conjugate prior updates (Normal-Normal, Beta-Binomial, Gamma-Poisson)

These are manually composed today:
```clojure
(compose-middleware h/generate-transition
  kalman-dispatch hmm-dispatch conjugate-dispatch)
```

The missing piece is **automatic detection and wiring**.

### Milestones

**L3-M1: Conjugacy detection in gen macro.**
At macro-expansion time, scan trace site pairs. When a site's distribution
parameters include another site's value, check if the pair forms a known
conjugate family (Gaussian-Gaussian, Beta-Bernoulli, Gamma-Poisson, etc.).
Emit metadata tagging conjugate pairs.

**L3-M2: Auto-wiring conjugate middleware.**
Use the conjugacy metadata from L3-M1 to automatically construct the
`compose-middleware` call. The user writes a normal `gen` function; the
system detects conjugacy and applies analytical inference without annotation.

```clojure
;; User writes:
(def model
  (gen [x]
    (let [mu    (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/gamma 2 1))]
      (trace :y (dist/gaussian mu 1)))))

;; System detects: :mu → :y is Normal-Normal conjugate
;; Auto-generates middleware that analytically marginalizes :mu
;; Only :sigma needs sampling
```

**L3-M3: Linear-Gaussian substructure detection.**
Analyze the operations between latent and observed trace sites. If the
computation graph consists only of affine operations (add, multiply, subtract
with constants), the substructure is linear-Gaussian. Auto-apply Kalman
middleware. This subsumes conjugacy detection for the Gaussian case and
extends to temporal models.

**L3-M4: Dependency graph for conditional independence.**
Build a full dependency graph from the gen body at macro time. Identify
conditionally independent subgraphs. These can be:
- Compiled independently (separate Metal dispatches that run in parallel)
- Updated independently in MCMC (parallel Gibbs blocks)
- Eliminated independently (each subgraph checked for analytical solutions)

**L3-M5: Algebraic graph rewriting.**
Handlers operate on the MLX computation graph, not just values. A handler
that sees a conjugate pair can replace two stochastic nodes with one
deterministic posterior computation and a marginal likelihood term. This is
Rao-Blackwellization as graph rewriting — applied automatically by the
middleware stack.

### The rewrite engine vision

The middleware stack becomes a rewrite engine that progressively simplifies
the model's computation graph:

```
Layer 1: Conjugacy elimination    → merge conjugate pairs into closed form
Layer 2: Linear-Gaussian collapse → Kalman where applicable
Layer 3: Discrete elimination     → HMM forward where applicable
Layer 4: Independence detection   → parallelize independent subgraphs
Layer 5: Everything else          → sample normally (compiled from Level 1)
```

Each layer reduces the graph. What emerges is the **minimal stochastic
computation** — the irreducible core that must be sampled because no
analytical solution exists. Inference runs only on this residual.

---

## Level 4: One Graph to Rule Them All

**What it is:** Model specification, inference algorithm, and parameter
optimization are all expressed as one lazy MLX computation graph.
ClojureScript builds the graph, then sits idle while Metal executes.

**What's in the graph:** Everything. The model, the analytical eliminations,
the compiled inference sweep, the gradient computation, the optimizer step.
One `mx/eval!`.

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

Selection based on the structure metadata from Level 3's macro analysis.
The user calls `(fit model data)` and the system figures out the rest.

**L4-M4: LLM as generative function.**
An LLM wrapped as a generative function with a proper score (log-probability
of the generated text). Enables inference over natural language: condition on
observations, sample explanations, compute posteriors over hypotheses expressed
in text. The GFI's `assess` gives the score; `generate` with constraints
gives conditional generation.

**L4-M5: Amortized combinator search.**
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
At Level 3+, GenMLX does things GenJAX cannot: automatic analytical
elimination, macro-time structure detection, composable middleware stacking.

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

---

## Status

### Completed

- Level 0: Full shape-based GPU batching, all combinators, all inference algorithms
- Analytical middleware: Kalman, EKF (1D + ND), HMM forward, conjugate priors
- Compiled combinators: unfold, particle filter
- Differentiable inference, Fisher information, PMCMC
- Formal verification: 10 GFI contracts, Lean mechanization (Phase A)

### In progress

- Depression modeling (D/E models) — testbed for middleware composition
- Formal verification Phase B — kernel and inference correctness proofs

### Next

- Level 1: Macro-time program analysis and compiled gen functions
- Level 2: Compiled inference sweeps with differentiable resampling
- Level 3: Automatic conjugacy detection and middleware wiring
- Level 4: End-to-end differentiable probabilistic programming
