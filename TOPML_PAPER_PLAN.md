# TOPML Paper Plan: GenMLX

> Detailed section-by-section plan for a paper I believe will be accepted
> at ACM Transactions on Probabilistic Machine Learning.

---

## Title Options

- **GenMLX: GPU-Accelerated Probabilistic Programming on Apple Silicon**
- **GenMLX: The Generative Function Interface on Apple Silicon via Broadcasting**
- **Probabilistic Programming Without CUDA: GenMLX on Apple Silicon**

Pick whichever the co-authors prefer. The third is the most attention-grabbing.

---

## Thesis

GenMLX is the first probabilistic programming system that implements the
complete Generative Function Interface on Apple Silicon, achieving
29–122x vectorized inference speedups through a novel broadcasting-based
approach that requires no program transformation. The system is 10,000
lines of ClojureScript, supports 22 distributions, 9 combinators, and
18+ inference algorithms, and passes 238 compatibility tests against
Gen.jl and GenJAX.

---

## Paper Structure (~28 pages + appendix)

### 1. Introduction (2.5 pages)

**Opening paragraph.** GPU-accelerated probabilistic programming has
been confined to the CUDA ecosystem: GenJAX runs on JAX, Pyro on
PyTorch, both requiring NVIDIA hardware. Meanwhile, Apple Silicon's
unified memory architecture — where CPU and GPU share the same physical
memory with zero transfer cost — is a natural fit for probabilistic
programming, where MCMC control flow runs on CPU while all numerics
stay on GPU. Yet no probabilistic programming system targets this
hardware.

**Problem statement.** Building a probabilistic programming system on
a new hardware backend is not merely a porting exercise. The standard
approach to vectorized inference — JAX's `vmap` — is a compiler
transformation that rewrites programs. Apple's MLX framework has no
`vmap`. The question: can we achieve the same performance gains through
a fundamentally different mechanism?

**Our answer.** Yes. GenMLX implements the complete Generative Function
Interface (GFI) — the same architecture as Gen.jl and GenJAX — on Apple
Silicon via MLX. Instead of `vmap`, GenMLX exploits **broadcasting**:
sample [N]-shaped arrays at each trace site, and all downstream
arithmetic (log-probability, score accumulation, weight computation)
just works, because the handler transitions never inspect array shapes.
This approach requires no program transformation, no compiler support,
and no changes to model code.

**Contributions.** (Numbered list, each with forward reference.)

1. **System.** GenMLX: a complete GFI implementation in ~10,000 lines of
   ClojureScript on Node.js, using MLX for GPU computation. 22
   distributions via an open `defdist` macro system, 9 structural
   combinators, 18+ inference algorithms from importance sampling to
   NUTS to SMCP3 to variational inference. (§3–4)

2. **Broadcasting-based vectorization.** A novel approach to parallel
   particle execution that replaces `vmap` with array broadcasting.
   We prove it correct using a logical relations argument adapted from
   the GenJAX formalization. Achieves 29–122x speedups. (§5)

3. **Handler architecture.** A two-layer design — pure state transitions
   + volatile! dispatch — that separates the semantics of each GFI
   operation from the mechanics of execution. This makes the system
   formally tractable and enables transparent batching. (§3.2)

4. **Evaluation.** Benchmarks on three standard probabilistic models,
   comparison with Gen.jl, vectorization scaling study, and validation
   via 238 compatibility tests. (§7)

5. **Formal foundation.** Correctness proofs for all GFI operations,
   combinator compositionality, and broadcasting. (§8, Appendix)

**Why TOPML.** This paper bridges probabilistic ML theory (GFI
formalization, broadcasting correctness) with practice (a working
system on consumer hardware). The broadcasting approach is a
methodological contribution: it shows that vectorized inference does
not require compiler transformations, only a shape-agnostic runtime.

---

### 2. Overview and Running Example (2.5 pages)

**2.1 A Bayesian Regression Model**

Present the complete model in GenMLX syntax:

```clojure
(def model
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))
```

Walk through what `dyn/trace` does: dispatches to the active handler,
which records the choice and accumulates the score. Emphasize: this is
the same code for simulation, inference, and vectorized inference — only
the handler changes.

**2.2 GFI Operations on the Example**

Show each GFI operation applied to the model:
- `simulate`: sample slope, intercept, y0–y4; return trace with score
- `generate` with observations: constrain y0–y4, return importance weight
- `update`: change slope from 2.0 to 2.1, get weight = score_new − score_old
- `regenerate`: resample slope, compute MH weight

Show actual MLX array values at each step (small concrete numbers).
The reader should understand GFI before any formal definitions.

**2.3 Vectorized Execution**

Same model, but now with `vgenerate` for N=100 particles. Show that
choices are [100]-shaped arrays, scores are [100]-shaped. Emphasize:
the model code is unchanged; only the handler samples [N]-shaped values
instead of scalars.

Include a timing comparison: sequential generate (100 calls) vs
vgenerate (1 call with [100]-shaped arrays). Show the 61x speedup.

---

### 3. System Design (4 pages)

**3.1 Architecture Overview**

Present the 8-layer architecture diagram:

```
Layer 0: MLX Foundation     (mlx, mlx.random)
Layer 1: Core Data          (choicemap, trace, selection)
Layer 2: GFI & Execution    (protocols, handler, edit, diff)
Layer 3: DSL                (gen macro, dynamic)
Layer 4: Distributions      (dist/core, dist/macros, dist — 22 types)
Layer 5: Combinators        (Map, Unfold, Switch, Scan, Mask, Mix, Recurse, ...)
Layer 6: Inference          (IS, MH, HMC, NUTS, Gibbs, SMC, SMCP3, VI, MAP, ...)
Layer 7: Vectorized         (VectorizedTrace, batched execution)
```

Explain each layer in 2–3 sentences. Emphasize the dependency structure:
higher layers depend only on lower layers; inference algorithms use only
the GFI protocol, not internal implementation details.

**Table: System at a glance.**

| Metric | Value |
|--------|-------|
| Language | ClojureScript (on Node.js via nbb) |
| Lines of code | ~10,000 |
| Hardware | Apple Silicon (M1/M2/M3/M4) |
| GPU framework | MLX via @frost-beta/mlx |
| Distributions | 22 (10 with native batch sampling) |
| Combinators | 9 (Map, Unfold, Switch, Scan, Mask, Mix, Recurse, Contramap, Dimap) |
| Inference algorithms | 18+ |
| Compatibility tests | 238 (165 Gen.clj + 73 GenJAX) |

**3.2 Handler Architecture**

This is the key design insight. Present the two-layer structure:

**Layer 1: Pure state transitions.**
```
transition : (State, Address, Distribution) → (Value, State)
```
Five modes: simulate, generate, update, regenerate, project. Each is a
pure function that takes immutable state, produces immutable state.
Show the simulate transition (6 lines of ClojureScript) and the
generate transition (10 lines).

**Layer 2: Volatile! dispatch.**
```
handler : (Address, Distribution) → Value
```
A thin wrapper that reads the current state from a `volatile!`, calls
the pure transition, writes the new state back. The `run-handler`
function scopes the volatile inside a `binding` block — no global state,
no aliasing.

**Why this matters:**
- Pure transitions are testable, composable, and formally tractable
- The volatile! boundary is an implementation detail invisible to model
  code and to the formal semantics
- Adding a new handler mode (e.g., batched variants) requires only
  writing new pure transitions — the dispatch machinery is reused

**3.3 The GFI Protocol**

List all 8 operations with their types:
- `simulate(g, args) → Trace`
- `generate(g, args, constraints) → {trace, weight}`
- `update(g, trace, constraints) → {trace, weight, discard}`
- `regenerate(g, trace, selection) → {trace, weight}`
- `assess(g, args, choices) → {weight}`
- `project(g, trace, selection) → weight`
- `propose(g, args) → {choices, weight}`
- `edit(g, trace, request) → {trace, weight, discard, backward}`

Explain how each operation is implemented by running the model body
under a different handler mode. The model code calls `dyn/trace` which
dispatches to whichever handler is active.

**3.4 Data Structures**

Brief description of:
- **ChoiceMap**: hierarchical address → value map. Two variants: Value
  (leaf) and Node (nested). Supports `get-choice`, `set-choice`,
  `get-submap`, merge.
- **Trace**: immutable record {gen-fn, args, choices, retval, score}.
- **Selection**: composable address selection algebra (AllSelection,
  select by keywords, complement, hierarchical, union).

**3.5 The `gen` Macro and DynamicGF**

Show how the `gen` macro transforms a ClojureScript function body into
a `DynamicGF` record that implements all GFI protocols. The body is
stored as a closure (`body-fn`) and re-executed under different handlers.

---

### 4. Distributions and Combinators (2.5 pages)

**4.1 The Distribution System**

Show the `defdist` macro:

```clojure
(defdist gaussian [mu sigma]
  :sample   (fn [key] (mx/add mu (mx/multiply sigma (rng/normal key []))))
  :log-prob (fn [v] ...))
```

Explain that distributions are a single `Distribution` record with open
multimethods. Adding a new distribution is 10 lines in any namespace.
Contrast with Gen.jl (requires modifying core) and Pyro (class
inheritance).

**Table: All 22 distributions** with columns: name, type (continuous/
discrete), support, reparameterizable?, native batch sampling?.

Note the rejection-sampling distributions (beta, gamma, poisson,
student-t, dirichlet) and their performance implications for
vectorization.

**4.2 Structural Combinators**

Present each combinator with a one-line description and a small example:

1. **Map(g)** — apply g to each element independently. Example: observe
   N data points from the same model.
2. **Unfold(g)** — sequential composition with carry state. Example:
   hidden Markov model.
3. **Scan(g)** — like Unfold but with explicit input sequence and output
   accumulation.
4. **Switch(g₁,...,gₙ)** — branch on a discrete index. Example:
   mixture model component selection.
5. **Mask(g)** — gate execution on a boolean. Example: optional model
   components.
6. **Mix(ws, g₁,...,gₙ)** — first-class mixture with learnable weights.
7. **Recurse(maker)** — recursive model structure. Example: tree-shaped
   probabilistic programs.
8. **Contramap(f, g)** — transform arguments.
9. **Dimap(f, h, g)** — transform arguments and return value.

Show one worked example: HMM via `Unfold`.

```clojure
(def hmm-step
  (gen [carry]
    (let [state (dyn/trace :state (dist/categorical transition-probs))
          obs   (dyn/trace :obs (dist/categorical (emission-probs state)))]
      state)))

(def hmm (comb/unfold hmm-step))
```

**4.3 The Edit Interface**

Briefly describe the three edit types:
- ConstraintEdit: like update (replace values)
- SelectionEdit: like regenerate (resample selected addresses)
- ProposalEdit: custom proposals with forward-backward GF pair (for
  SMCP3 and involutive MCMC)

The edit interface generalizes update and regenerate into a single
parametric operation, enabling advanced inference algorithms.

---

### 5. Broadcasting-Based Vectorization (3 pages)

This is the paper's key technical insight. It deserves its own section
because it's fundamentally different from the `vmap` approach in GenJAX.

**5.1 The Problem**

Particle-based inference (IS, SMC) requires running the model N times
independently. Sequential execution wastes GPU parallelism. JAX solves
this with `vmap` — a compiler transformation that rewrites the program
to operate on batched arrays. MLX has no `vmap`.

**5.2 The Insight: Shape-Agnostic Handlers**

The GenMLX handler transitions never inspect array shapes. They use
`mx/add` for score accumulation, `cm/set-choice` for storing values,
`rng/split` for key management — all of which work identically on
scalars and [N]-shaped arrays.

This means: if we sample [N]-shaped values at each trace site (via
`dist-sample-n`), all downstream computation automatically produces
[N]-shaped results. No program transformation needed.

**Figure: Broadcasting in action.** Show the dataflow for a 2-site
model with N=3 particles:
- `dist-sample-n(gaussian, key, 3)` → `[v₁, v₂, v₃]`
- `dist-log-prob(gaussian, [v₁,v₂,v₃])` → `[lp₁, lp₂, lp₃]`
- `score + [lp₁,lp₂,lp₃]` → `[s₁, s₂, s₃]` (broadcasting)

**5.3 Correctness**

State the broadcasting correctness theorem informally: batched execution
produces the same result as N independent sequential executions, packed
into struct-of-array format.

The formal proof uses logical relations (adapted from the GenJAX paper's
Theorem 3.3) but applied to a semantic property of the handler rather
than a syntactic transformation of the program. Full proof in Appendix.

**5.4 What Can and Cannot Be Vectorized**

Be honest about limitations:
- 25 of 22 distributions are broadcastable; wishart and inv-wishart
  are not (they inspect value shapes in log-prob)
- Model bodies must use only broadcasting-compatible operations (no
  `mx/item`, no branching on array values)
- Splice in batched mode only supports DynamicGF (not arbitrary
  combinator sub-GFs)
- Rejection-sampling distributions (beta, gamma, poisson, student-t,
  dirichlet) fall back to sequential sampling (log-prob still broadcasts)

**5.5 Performance**

Preview the vectorization speedup numbers (full evaluation in §7):

| Operation | N=100 | N=1000 |
|-----------|-------|--------|
| dist-sample-n | 29x | 78x |
| vgenerate | 61x | — |
| Vectorized IS | 81x | — |
| Vectorized SMC init | 65x | — |

---

### 6. Inference Algorithms (3 pages)

**6.1 Importance Sampling and SMC**

- Importance sampling with resampling
- Vectorized IS (single model execution for N particles)
- SMC with ESS-based resampling and MH rejuvenation
- Conditional SMC (CSMC)
- SMCP3: sequential proposals via the edit interface

Show: vectorized IS on the running example (4 lines of code + result).

**6.2 MCMC**

- Metropolis-Hastings (selection-based and custom-proposal)
- MALA (gradient-informed random walk)
- HMC (leapfrog integration, configurable steps and step size)
- NUTS (dual averaging for step size, adaptive tree depth)
- Enumerative Gibbs (exact block updates for discrete variables)
- Elliptical slice sampling (likelihood-free)
- Involutive MCMC (via ProposalEdit)
- MAP optimization (gradient ascent on log-joint)

**Table: MCMC algorithms** with columns: algorithm, gradient-free?,
discrete variables?, auto-tuning?, reference.

Show: NUTS on the running example (5 lines of code + trace plot).

**6.3 Variational Inference**

- ADVI with mean-field Gaussian guide
- Programmable VI objectives: ELBO, IWELBO, wake-sleep
- ADEV: automatic differentiation of expected values
- Amortized inference: train neural proposals

Show: ADVI on the running example (5 lines + ELBO convergence curve).

**6.4 Kernel Composition**

The kernel DSL enables building complex inference strategies:
- `chain(k₁, k₂)`: apply k₁ then k₂
- `cycle([k₁,...,kₙ])`: apply in sequence, repeat
- `mix(ws, [k₁,...,kₙ])`: randomly choose kernel by weight
- `repeat(k, n)`: apply k n times

Show: a Gibbs-within-MH kernel that alternates between discrete
and continuous updates.

**6.5 Gradient Infrastructure**

- Choice gradients: ∇_θ log p(trace) w.r.t. selected addresses
- Score gradients: ∇_θ log p(x | obs) for parameter arrays
- All gradients via MLX's `mx/grad` — stay on GPU, no materialization

---

### 7. Evaluation (4 pages)

**7.1 Benchmark Models**

Three models of increasing complexity:

**Model 1: Bayesian Linear Regression** (5 parameters, 20 observations).
Simple but standard. Analytic posterior available for validation.
Exercises: generate, update, importance sampling, MH, HMC.

**Model 2: Hidden Markov Model** (T=50 steps, K=3 states).
Sequential structure via Unfold combinator. Exercises: SMC, Gibbs,
forward-backward. Compares sequential vs vectorized SMC.

**Model 3: Gaussian Mixture Model** (K=3 components, D=2 dimensions,
N=200 data points). Exercises: Mix combinator, discrete+continuous
inference, component switching via involutive MCMC or Gibbs.

**7.2 Inference Accuracy**

For each model, compare inferred posteriors to ground truth:
- Model 1: analytic posterior (Normal-Normal conjugacy)
- Model 2: forward-backward algorithm (exact marginals)
- Model 3: synthetic data with known parameters

Report: KL divergence or Wasserstein distance to ground truth, as a
function of number of samples/particles. Show convergence curves.

**7.3 Vectorization Scaling**

For Models 1 and 2, measure wall-clock time as a function of particle
count N ∈ {1, 10, 50, 100, 500, 1000}:
- Sequential: N calls to `generate`
- Vectorized: 1 call to `vgenerate` with batch size N

Plot: time vs N (log-log), showing that vectorized scales sublinearly
while sequential scales linearly. Report the crossover point where
vectorized becomes faster.

**7.4 Comparison with Gen.jl**

Run Models 1–3 on both GenMLX (Apple Silicon M2/M3) and Gen.jl (same
machine, CPU). Report:
- Samples per second for MH
- Particles per second for IS/SMC
- Time to convergence (first N samples where posterior estimate is
  within ε of ground truth)

This is a fair comparison: same hardware, same algorithms, same models.
Gen.jl has a mature Julia implementation; GenMLX has GPU acceleration.

Note: GenJAX comparison would require NVIDIA hardware and is not
apples-to-apples. Acknowledge this limitation. If access to NVIDIA
hardware is available, include GenJAX numbers as well.

**7.5 Compatibility Validation**

Report: 165/165 Gen.clj compatibility tests pass, 73/73 GenJAX
compatibility tests pass. These cover: distribution log-probs, GFI
invariants (generate/update/regenerate consistency), MCMC convergence
to conjugate posteriors, gradient correctness, numerical stability.

This is not a benchmark but a correctness validation — evidence that
GenMLX implements the GFI faithfully.

**7.6 Hardware Profile**

Report for each model:
- GPU memory usage (peak, measured via MLX profiling)
- Fraction of time in GPU vs CPU
- Effect of `mx/eval!` placement on memory and throughput

If available, compare M1 vs M2 vs M3 vs M4 to show scaling across
Apple Silicon generations.

---

### 8. Formal Foundation (2.5 pages)

This section provides formal backing for the system's correctness.
It should be accessible to readers who skip the proofs (in the
appendix) and just want to know what is guaranteed.

**8.1 The λ_MLX Calculus (0.5 pages)**

Briefly present the type system extending λ_GEN:
- Handler state type H(σ, τ) — new
- EditRequest types — new
- Typing rules for update, regenerate, edit — new

Reference the POPL 2026 paper for the base calculus (λ_GEN) and
state what λ_MLX adds. Do not re-derive λ_GEN.

**8.2 Correctness Guarantees (1 page)**

State the four main theorems (informally, with precise statements in
appendix):

**Theorem 1 (GFI Correctness).** Each GFI operation (generate, update,
regenerate, project, propose, edit) produces correct traces and weights
as defined by the denotational semantics.

**Theorem 2 (Broadcasting Correctness).** Batched execution with N
particles produces the same result as N independent sequential
executions, for all models using broadcastable distributions and
shape-compatible operations.

**Theorem 3 (Combinator Compositionality).** If g satisfies the GFI
contract, then C(g) satisfies the GFI contract for all 9 combinators.

**Theorem 4 (Edit Duality).** The backward edit request correctly
inverts the forward edit, ensuring valid MH acceptance ratios.

For each: state the theorem in 2–3 sentences, explain what it
guarantees for the practitioner, and say "Proof in Appendix X."

**8.3 Broadcasting vs vmap (1 page)**

Compare our approach to the POPL paper's vmap_n:
- vmap_n is a source-to-source transformation (syntactic)
- Broadcasting is a runtime property (semantic)
- Both achieve the same result: N independent particles in SoA format
- vmap requires compiler support; broadcasting requires shape-agnostic
  handlers

Explain why this difference matters: broadcasting works on any
framework with element-wise operations, not just JAX. This makes
the approach portable to future hardware backends.

---

### 9. Related Work (2.5 pages)

**9.1 Probabilistic Programming Systems**

- **Gen.jl** (Cusumano-Towner et al. 2019): The reference GFI
  implementation. Julia, CPU-only. GenMLX implements the same interface
  on Apple Silicon with GPU acceleration.
- **GenJAX** (Becker et al. 2026): GFI on JAX with vmap-based
  vectorization. GenMLX achieves similar speedups via broadcasting
  instead of vmap, on different hardware.
- **Pyro/NumPyro** (Bingham et al. 2019, Phan et al. 2019):
  Effect-handler-based PPL on PyTorch/JAX. Similar handler architecture
  (Poutine ≈ our handler system) but different interface (not GFI).
  NumPyro uses JAX's vmap; Pyro uses PyTorch's vectorized distributions.
- **Turing.jl** (Ge et al. 2018): Julia PPL with compiler-based
  approach. No explicit GFI but similar capabilities.
- **Stan** (Carpenter et al. 2017): Mature HMC/NUTS system. Not
  programmable (fixed model language), no combinators, no custom
  proposals. Gold standard for NUTS performance.
- **Edward2/TFP** (Tran et al. 2018): TensorFlow-based. No GFI,
  limited composability.

**9.2 Vectorized Inference**

- **GenJAX's vmap_n**: Source-to-source transformation with logical
  relations proof. Our broadcasting approach achieves the same result
  without program transformation.
- **NumPyro's vectorized distributions**: Similar spirit (batch
  sampling) but applied at the distribution level, not at the handler
  level. NumPyro still uses vmap for model-level vectorization.
- **Pyro's plate**: Vectorized independent observations. More limited
  than our full-model broadcasting.

**9.3 Formal Foundations of Probabilistic Programming**

- **Ścibior et al. 2018**: Denotational validation of higher-order
  Bayesian inference using QBS. We build on their framework.
- **Heunen et al. 2017**: Quasi-Borel spaces. We use their results
  for function spaces and the probability monad.
- **Staton et al. 2016**: A convenient category for higher-order
  probability theory. Foundation for QBS approach.
- **Becker et al. 2026**: λ_GEN formalization. We extend with
  generate, update, regenerate, edit, and broadcasting.

**9.4 Hardware-Specific ML Systems**

- **MLX** (Apple, 2023): Array framework for Apple Silicon. Lazy
  evaluation, unified memory, automatic differentiation. GenMLX is
  (to our knowledge) the first probabilistic programming system on MLX.
- **Metal Performance Shaders**: Lower-level Apple GPU framework.
  MLX provides the higher-level abstractions we need.
- **CUDA ecosystem**: JAX, PyTorch, TensorFlow all target NVIDIA.
  GenMLX demonstrates that probabilistic programming does not require
  CUDA.

**9.5 Involutive MCMC and Advanced Proposals**

- **Cusumano-Towner 2020**: Involutive MCMC framework. Our ProposalEdit
  generalizes this.
- **Lew et al. 2023 (SMCP3)**: Sequential proposals. Our edit interface
  enables SMCP3 natively.
- **Neklyudov et al. 2020**: Involutive MCMC. Related to our
  ProposalEdit duality.

---

### 10. Discussion and Future Work (1 page)

**What works well:**
- Broadcasting is simpler than vmap (no compiler, no transformation)
- Unified memory eliminates CPU-GPU transfer overhead for MCMC
- ClojureScript's persistent data structures make traces cheap to copy
- The handler architecture cleanly separates concerns

**Current limitations:**
- Apple Silicon only (no CUDA, no AMD)
- Rejection-sampling distributions don't vectorize at sampling time
- No splice in batched mode for non-DynamicGF sub-GFs
- Model bodies must avoid shape-inspecting operations for vectorization
- Float32 only (MLX limitation for most operations)

**Future directions:**
- Vectorized update and regenerate (currently only simulate and generate)
- Automatic detection of non-broadcastable operations (static analysis)
- Extension to continuous-time models (SDEs via MLX)
- Amortized inference with neural network proposals
- WebGPU backend for browser-based probabilistic programming

---

### 11. Conclusion (0.5 pages)

GenMLX demonstrates that GPU-accelerated probabilistic programming does
not require CUDA, JAX, or compiler transformations. By implementing the
complete Generative Function Interface on Apple Silicon via a
broadcasting-based approach, we achieve 29–122x speedups for vectorized
inference while maintaining formal correctness guarantees. The system is
10,000 lines of ClojureScript — small enough to audit, expressive enough
to support 18+ inference algorithms, and fast enough to be practical.

---

### Appendix A: Formal Proofs (~10 pages)

**A.1 λ_MLX Calculus** — Full type grammar, term grammar, typing rules.
Corresponds to `formal/calculus.md`.

**A.2 Denotational Semantics** — QBS interpretation, handler transition
semantics. Corresponds to `formal/semantics.md`.

**A.3 Program Transformations** — All 8 GFI operations as
source-to-source transformations. Corresponds to
`formal/transformations.md`.

**A.4 Proof of Theorem 1 (GFI Correctness)** — By structural induction.
Corresponds to `formal/proofs/correctness.md`.

**A.5 Proof of Theorem 2 (Broadcasting Correctness)** — Logical
relations, four lemmas, main theorem. Includes distribution
categorization (25 broadcastable, 2 excluded). Corresponds to
`formal/proofs/broadcasting.md`.

**A.6 Proof of Theorem 3 (Combinator Compositionality)** — Per
combinator. Corresponds to `formal/proofs/combinators.md`.

**A.7 Proof of Theorem 4 (Edit Duality)** — Three edit types,
ProposalEdit detailed balance. Corresponds to
`formal/proofs/edit-duality.md`.

**A.8 Handler Soundness** — Induction on trace operations, volatile!
invisibility, splice soundness. Corresponds to
`formal/proofs/handler-soundness.md`.

**A.9 Diff-Aware Update Correctness** — Map VectorDiff, Unfold/Scan
prefix skipping. Corresponds to `formal/proofs/diff-update.md`.

---

### Appendix B: Additional Benchmarks

Raw data for all experiments. Timing methodology (warm-up runs,
`mx/eval!` placement, GC considerations).

---

## Section Budget

| Section | Pages | Content Focus |
|---------|-------|---------------|
| 1. Introduction | 2.5 | Motivation, contributions |
| 2. Overview | 2.5 | Running example, GFI walkthrough |
| 3. System Design | 4 | Architecture, handler, protocols, data structures |
| 4. Distributions & Combinators | 2.5 | defdist, 22 distributions, 9 combinators |
| 5. Broadcasting | 3 | Key insight, correctness sketch, limitations |
| 6. Inference Algorithms | 3 | IS, MCMC, VI, kernels, gradients |
| 7. Evaluation | 4 | Benchmarks, comparisons, compatibility |
| 8. Formal Foundation | 2.5 | Theorems (informal), broadcasting vs vmap |
| 9. Related Work | 2.5 | PP systems, vectorization, formal methods, hardware |
| 10. Discussion | 1 | Limitations, future work |
| 11. Conclusion | 0.5 | Summary |
| **Total body** | **~28** | |
| Appendix A (proofs) | ~10 | Full formal development |
| Appendix B (data) | ~2 | Raw benchmark data |

---

## What Must Be Created (Not Yet Existing)

| Item | Effort | Priority |
|------|--------|----------|
| Benchmark suite (Models 1-3 with timing harness) | Large | Critical |
| Gen.jl comparison (install, implement same models, run) | Large | Critical |
| Figures: architecture diagram, broadcasting dataflow, speedup plots, convergence curves, posterior comparisons | Medium | Critical |
| Vectorization scaling study (N=1 to N=1000) | Medium | Critical |
| Hardware profiling (GPU vs CPU time, memory) | Small | Important |
| Inference code examples (5-line snippets for §6) | Small | Important |
| LaTeX manuscript in TOPML template | Medium | Required |

## What Already Exists and Can Be Used Directly

| Item | Source |
|------|--------|
| System implementation | src/genmlx/ (~10K lines) |
| 238 compatibility tests | test/genmlx/*_compat_test.cljs |
| Vectorization speedup numbers | test/genmlx/vectorized_benchmark.cljs |
| GPU benchmarks (exploratory) | test/genmlx/gpu_benchmark.cljs |
| Formal proofs (appendix material) | formal/ (10 files, ~2700 lines) |
| Architecture description | ARCHITECTURE.md, CLAUDE.md |
| Model examples | README.md, test files |
| Distribution catalog | src/genmlx/dist.cljs |

---

## Reviewer Objections to Anticipate

1. **"ClojureScript is niche — who will use this?"**
   Response: The contribution is the broadcasting approach and the
   formal foundation, not the language choice. The technique applies to
   any framework with element-wise operations. ClojureScript demonstrates
   this with a minimal, auditable implementation.

2. **"No comparison with GenJAX on NVIDIA hardware."**
   Response: Acknowledge this limitation. The comparison is with Gen.jl
   on the same hardware. The point is not "faster than NVIDIA" but
   "competitive on consumer hardware without CUDA."

3. **"Float32 only — can you do real statistics?"**
   Response: Float32 is sufficient for most Bayesian inference (MCMC
   acceptance ratios, importance weights, ELBO estimation). Stan uses
   Float64 but acknowledges that Float32 is often adequate. Report any
   numerical issues encountered in the evaluation.

4. **"The formal proofs are in an appendix — is this really a theory
   contribution?"**
   Response: The primary contribution is the system and the broadcasting
   approach. The formal proofs provide confidence that the system is
   correct. For a full formal treatment, see [our companion formalization].

5. **"Only 22 distributions — Pyro has hundreds."**
   Response: The `defdist` macro makes adding distributions trivial (10
   lines). The 22 distributions cover the standard set used in Bayesian
   modeling. The open multimethod design means users can add distributions
   without modifying core code.
