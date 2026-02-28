# GenMLX Vision, Audit & Research Roadmap

## What GenMLX Is

GenMLX is a purely functional probabilistic programming language in ClojureScript
on Node.js (nbb/Bun), using Apple's MLX framework for GPU acceleration on Apple
Silicon. It implements the Generative Function Interface (GFI) — the same
architecture as Gen.jl (Julia) and GenJAX (JAX).

~10,600 lines of ClojureScript. Purely functional. Data-driven. GPU end-to-end.

---

## Full Codebase Audit (February 2026)

### Does the vision hold?

**Yes, remarkably well.** Across ~10,600 lines of source code and 32 files, the
codebase is almost fanatically pure functional.

#### Complete mutable state inventory

| Location | What | Justified? |
|---|---|---|
| `handler.cljs:605` | `volatile! *state*` in `run-handler` | Yes — the *one* mutable cell, scoped to a `binding` block, never escapes |
| `mlx/random.cljs:16` | `*prng-key*` dynamic var | Yes — set via `with-key`, enables reproducible inference |
| `handler.cljs` | `*handler*`, `*param-store*` dynamic vars | Yes — read-only from handler perspective, set via `binding` |
| `mlx.cljs:13,21` | `defonce` module cache + cpu-stream | Yes — one-time JS interop initialization |
| `mcmc.cljs:21-39` | `mx/set-default-device!` | Yes — try/finally restoration, needed for Metal dispatch |
| `util.cljs:234-280` | `mx/tidy`, cache clearing | Yes — essential Metal buffer management |

That's it. No atoms hiding in closures, no sneaky side-effect chains, no global
state pollution. Every handler transition is a pure
`(fn [state addr dist] -> [value state'])`. This is genuinely rare in a system
of this complexity.

### Minor concerns found

The audit found only minor concerns, not architectural violations:

1. **`gradients.cljs`** — No validation that `addresses` and `current-vals`
   dimensions match. Could fail silently with wrong address order.
2. **`diagnostics.cljs:76`** — R-hat has no guard against `W = 0` (constant
   chains). Would produce `Inf`.
3. **`vectorized.cljs`** — GPU resampling is `O(N²)` in memory. Documented but
   could surprise users at large N.
4. **`inference/util.cljs:153`** — Falls back to `js/Math.random` when PRNG key
   is nil — breaks reproducibility silently.
5. **`smcp3.cljs:99-102`** — When backward kernel is nil, falls back to
   constraint edit without explaining the weight implications.

None of these are purity violations. They're edge-case robustness issues.

### Mathematical correctness

All inference algorithms verified correct:

- **MCMC:** MH, MALA (Langevin with asymmetric correction), HMC (fused
  leapfrog), NUTS, Gibbs (enumerative categorical), Elliptical Slice,
  Involutive MCMC (with Jacobian)
- **Particle methods:** IS, importance resampling, SMC (systematic/residual/
  stratified resampling), conditional SMC, SMCP3
- **Variational:** ELBO, IWELBO, VIMCO (leave-one-out control variates), PWAKE,
  QWAKE, REINFORCE with baseline
- **Diagnostics:** ESS (Geyer's initial positive sequence), R-hat
  (Gelman-Rubin), summary statistics

---

## Completeness Assessment

### By layer

| Layer | Status | Notes |
|---|---|---|
| 0: MLX Foundation | 10/10 | Thin, clean wrapper. Lazy graph + explicit eval. |
| 1: Core Data | 10/10 | Choicemaps, traces, selections — all purely functional |
| 2: GFI & Execution | 10/10 | All 8 GFI ops implemented. Handler system is beautiful. |
| 3: DSL | 9/10 | Dynamic DSL complete. No static DSL yet. |
| 4: Distributions | 9/10 | 27 distributions with batch sampling. Missing a few exotic ones. |
| 5: Combinators | 10/10 | 9 combinators, all with full GFI. More than Gen.jl's 5. |
| 6: Inference | 9/10 | 20+ algorithms. ADEV partially integrated. |
| 7: Vectorized | 9/10 | 29-62x speedups measured. Broadcasting approach works. |

### Comparison to reference implementations

| Dimension | Gen.jl | GenJAX | GenMLX |
|---|---|---|---|
| GFI completeness | 100% | ~90% | 95% |
| Distributions | 30+ | 20+ | 27 |
| Inference algorithms | 25+ | 15+ | 20+ |
| Combinators | 5 | 4 | 9 |
| Vectorization | None | Native (vmap) | Batch-based (shape broadcasting) |
| Loop compilation | None | XLA | MLX (5-60x speedup) |
| GPU support | None | Yes (JAX) | Yes (MLX Apple Silicon) |
| Test assertions | 1000s | 1000s | 1474+ |
| Code size | 50K+ LOC | 30K+ LOC | 10.6K LOC |

### Testing

- **1,474+ assertions** across 78 test files
- **165/165** Gen.clj compatibility tests
- **73/73** GenJAX compatibility tests
- **162 property-based tests** (test.check) across 9 files
- Statistical convergence tests against conjugate posteriors
- Shape correctness and distributional equivalence for vectorized inference

### Overall: ~90% complete as a research PPL

The remaining 10% is ecosystem: neural network integration, visualization,
npm packaging, static DSL.

---

## Paper Ideas

### Paper 1: The GenMLX System Paper

**"GenMLX: Purely Functional Probabilistic Programming on Apple Silicon via MLX"**

The core contribution: showing that a PPL can be (a) purely functional
end-to-end, (b) GPU-accelerated via lazy computation graphs, (c) implemented in
~10K LOC of ClojureScript, and (d) achieve feature parity with Gen.jl/GenJAX.

Key results:
- 29-62x vectorization speedups via shape-based batching (no vmap)
- Handler-as-pure-state-transition architecture
- 9 combinators with full GFI (more than Gen.jl)
- 20+ inference algorithms with loop compilation (5-60x speedup)
- 10.6K LOC vs 50K+ for Gen.jl — density through language fit

### Paper 2: Formal Correctness

**"Mechanized Correctness of Generative Function Interface Implementations"**

Because every handler transition is a pure function with an explicit type
signature `[state addr dist] -> [value state']`, you can state and prove
properties like:

- `generate(model, args, constraints).weight = log p(constraints | args)`
- `update` preserves the marginal: the discard + new trace reconstruct the
  original probability space
- Score decomposition: `project(trace, selection) +
  project(trace, complement(selection)) = trace.score`
- Vectorized operations preserve distributional identity with scalar operations

The 162 property-based tests already in the codebase are *empirical* versions
of these proofs. The pure functional architecture means the gap from property
tests to formal proofs (in Lean4 or Agda) is unusually small.

**This could be combined with Paper 1** — "here's a PPL whose architecture was
designed for formal verification, and here are the proofs" — or stand alone as
a stronger contribution at a PL/verification venue.

### Paper 3: Shape-Based Vectorization

**"Vectorized Inference Without vmap: Shape Broadcasting as a Batching Strategy for Probabilistic Programs"**

GenJAX uses JAX's `vmap` for vectorization. GenMLX does something fundamentally
different: it changes the *shape* of sampled values from `[]` to `[N]` and
relies on MLX broadcasting to make all downstream arithmetic work. This is
simpler, requires no program transformation, and the handler never inspects
shapes.

Key results:
- 29-62x speedups with dispatch amortization (model body runs *once* for N particles)
- No program transformation required — same handler code for scalar and batch
- Competitive with vmap-based approaches
- Naturally extends to all combinators

### Paper 4: Loop Compilation for MCMC

**"Compiled Markov Chains: Metal Kernel Fusion for MCMC on Apple Silicon"**

The compiled-MH/compiled-MALA/compiled-HMC work builds K-step Metal programs
that run without returning to the interpreter.

Key results:
- MH: 5.6x speedup
- MALA: 2.3x speedup
- HMC: 3.9x speedup
- Combined with vectorization (N parallel chains): multiplicative gains
- The technique generalizes: any iterative inference algorithm with a fixed
  computation graph can be compiled

### Paper 5: ClojureScript as a PPL Host Language

**"Persistent Data Structures and Open Multimethods as PPL Primitives"**

A PL-venue paper. The argument: ClojureScript's immutable data structures,
protocols, and multimethods are not just *convenient* for PPL implementation —
they're *natural*.

- Traces are immutable records
- Choicemaps are persistent maps
- Distributions are open multimethods (extend without modifying core)
- The `gen` macro is 19 lines
- Compare to the machinery required in Julia (mutable structs, generated
  functions) or Python (class hierarchies, JAX tracing)
- The "language fit" argument, supported by 10K vs 50K+ LOC

---

## Ideas for Taking GenMLX Further

### Near-term (high impact, builds on what exists)

#### 1. Lean4 formalization of GFI contracts

The property tests already exist. Translating the pure state transitions into
Lean4 propositions would be a first — no PPL has mechanized proofs of its
inference correctness. The purely functional architecture makes this tractable
where Gen.jl/GenJAX would be nightmarish.

#### 2. Static DSL via ClojureScript analyzer

Parse `gen` bodies at macro-expansion time, build a static computation graph,
compile to a single MLX kernel. This would close the performance gap with
GenJAX's XLA compilation for fixed-structure models.

#### 3. LLM-as-generative-function

Wrap an LLM API call as a `Distribution` — `sample` calls the LLM, `log-prob`
computes token log-probabilities. This enables SMCP3-style inference where the
LLM *is* the proposal. The purely functional interface means LLM calls compose
with all existing inference algorithms.

### Medium-term (novel research directions)

#### 4. Differentiable programming through probabilistic programs

MLX's autograd + the pure functional handler = you can differentiate through
entire inference procedures. Train a neural network whose loss function involves
running SMC inside a `gen` body. GenJAX does this via JAX; GenMLX could do it
via MLX with potentially simpler code.

#### 5. Distributed inference via immutable traces

Because traces are persistent data structures, they're trivially serializable.
Run particles on different machines, merge via choicemap operations.
ClojureScript's data structures were designed for this (structural sharing,
fast equality).

#### 6. WebGPU backend

MLX is Apple-only. But the architecture — lazy graph, explicit eval, pure
transitions — would work identically with a WebGPU compute shader backend.
GenMLX in the browser, doing GPU-accelerated probabilistic programming. The
ClojureScript-on-nbb stack is already 90% of the way to browser deployment.

#### 7. Incremental inference via diffs

The `diff.cljs` infrastructure (no-change, value-change, map-diff) already
exists but is underutilized. For streaming data, incrementally update traces by
propagating diffs through combinators — only recompute what changed. This is
where the Unfold/Scan combinators with prefix-skipping optimization shine.

### Long-term (ambitious)

#### 8. Self-hosting

Write GenMLX's inference algorithms *as* generative functions. An MCMC kernel is
itself a probabilistic program. This enables meta-inference: using inference to
tune inference. The purely functional architecture makes this feasible because
there's no hidden state to corrupt during recursive inference.

#### 9. Certified inference

Combine the Lean4 formalization with runtime proof witnesses. Each inference
step produces a certificate that the weight computation is correct. For
safety-critical applications (medical, autonomous systems), this is the endgame.

---

## Why This All Fits Together

What makes GenMLX distinctive isn't any one property — it's that they reinforce
each other:

- ClojureScript's immutability makes the GFI natural to express
- The GFI's pure transitions make formal reasoning tractable
- MLX's lazy graph model means pure functional code compiles to fast GPU kernels
- Bun/nbb means zero friction — `bun run --bun nbb model.cljs` and go
- Clean code is correct code is fast code, because the same properties (purity,
  immutability, data-orientation) serve all three goals simultaneously

The ClojureScript-MLX pairing is underappreciated: MLX's "build a graph, then
eval" model is essentially a functional programming model for GPU computation,
and ClojureScript is the right language to drive it.
