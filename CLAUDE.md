# GenMLX — Claude Code Project Guide

## What is this project?

GenMLX is a probabilistic programming language in ClojureScript on Node.js (nbb),
using Apple's MLX framework for GPU acceleration. It implements the **Generative
Function Interface (GFI)** — the same architecture as Gen.jl (Julia) and GenJAX (JAX).

The thesis: probabilistic programming and functional programming are the same thing.
ClojureScript's immutable data, open multimethods, and macro system map perfectly
onto the GFI's mathematical structure. MLX's lazy graphs, unified memory, and
broadcasting reinforce this — functional-style array programming without penalty.

~22,500 lines of ClojureScript. Purely functional, data-driven, GPU end-to-end.

## Compilation ladder (VISION.md)

GenMLX follows a 5-level compilation ladder, progressively moving work from the
host interpreter into fused MLX computation graphs:

```
Level 0: Shape-based batching       ← DONE (certified, 68/68)
Level 1: Compiled gen functions      ← DONE (506+ tests)
Level 2: Compiled inference sweeps   ← DONE (881+ tests)
Level 3: Auto-analytical elimination ← DONE (426 tests, 33.5x variance reduction)
Level 3.5: Extended analytical       ← DONE (150 tests, MVN Kalman, combinator conjugacy)
Level 4: Single fused graph          ← DONE (260+ tests, 9.2x compiled Adam speedup)
```

Each level adds performance without breaking the GFI semantic contract. A model
written today runs unchanged at higher compilation levels. The handler system is
ground truth; compilation is optimization.

## How to run things

```bash
# Run with Bun (recommended — 3-4x faster than Node.js for iterative inference)
bun run --bun nbb <file.cljs>

# Run all core tests
for f in choicemap_test trace_test selection_test handler_test dist_test gen_test combinators_test inference_test; do
  bun run --bun nbb "test/genmlx/${f}.cljs"
done

# Level 0 certification (must pass 68/68)
bun run --bun nbb test/genmlx/level0_certification_test.cljs

# Level 1 compilation tests
bun run --bun nbb test/genmlx/schema_test.cljs                # 174/174 (L1-M1)
bun run --bun nbb test/genmlx/compiled_simulate_test.cljs      # 82/82  (L1-M2)
bun run --bun nbb test/genmlx/partial_compile_test.cljs        # 92/92  (L1-M3)
bun run --bun nbb test/genmlx/combinator_compile_test.cljs     # 90/90  (L1-M5)

# Compatibility suites
bun run --bun nbb test/genmlx/gen_clj_compat_test.cljs    # 162/165 (3 pre-existing edge cases)
bun run --bun nbb test/genmlx/genjax_compat_test.cljs      # 73/73

# Vectorized inference tests + benchmarks
bun run --bun nbb test/genmlx/vectorized_test.cljs
bun run --bun nbb test/genmlx/vectorized_benchmark.cljs
```

No build step, no compilation. nbb interprets ClojureScript directly.

**Requirements:** macOS with Apple Silicon (or Linux/Windows with CUDA — MLX supports both), Bun (or Node.js 18+), `npm install` for `@frost-beta/mlx`.

## Project structure

```
src/genmlx/
  mlx.cljs              Thin wrapper over @frost-beta/mlx (arrays, ops, grad, vmap)
  mlx/random.cljs       Functional PRNG keys (split, sample — no global state)
  choicemap.cljs        Hierarchical address -> value maps (Value/Node protocol)
  trace.cljs            Immutable Trace record {gen-fn, args, choices, retval, score}
  selection.cljs        Composable address selection algebra
  protocols.cljs        GFI protocols: simulate, generate, update, regenerate, assess
  handler.cljs          Pure state transitions (simulate, generate, update, regenerate, ...)
  runtime.cljs          Execution runtime: run-handler with volatile! dispatch
  gen.cljc              gen macro (converts fn bodies into DynamicGF)
  dynamic.cljs          DynamicGF record (full GFI via handlers), vsimulate, vgenerate
  schema.cljs           Schema extraction from gen body source forms (L1-M1)
  compiled.cljs         Compiled execution paths: noise transforms, compiled simulate (L1-M2),
                        partial prefix (L1-M3), branch rewriting (L1-M4)
  compiled_gen.cljs     Compiled generate with middle-tier score function (L1)
  rewrite.cljs          Source form rewriting for compiled paths (L1-M4)
  tensor_trace.cljs     TensorTrace — flat tensor-backed trace for compiled paths
  affine.cljs           Affine dependency analysis for conjugacy detection (L3)
  conjugacy.cljs        Conjugate prior detection and Rao-Blackwellization (L3)
  dep_graph.cljs        Dependency graph analysis for auto-analytical (L3)
  method_selection.cljs Method selection: decision tree from model metadata (L4)
  fit.cljs              One-call entry point: (fit model args data) (L4)
  dist/core.cljs        Distribution record + open multimethods (sample, log-prob, sample-n)
  dist/macros.cljc      defdist macro (zero-boilerplate distribution definition)
  edit.cljs             Parametric edit interface (constraint, selection, proposal)
  diff.cljs             Incremental change tracking
  dist.cljs             27 distributions (gaussian, uniform, bernoulli, beta, gamma, ...)
  combinators.cljs      Map, Unfold, Switch, Scan, Mask, Mix, Recurse, Contramap/Dimap
  vmap.cljs             Vmap combinator (vmap-gf / repeat-gf with full GFI)
  vectorized.cljs       VectorizedTrace, resampling, ESS, log-ML utilities
  gradients.cljs        Choice gradients, score gradients
  learning.cljs         Parameter stores, optimizers (SGD, Adam), wake-sleep
  custom_gradient.cljs  CustomGradientGF — custom gradient generative functions
  nn.cljs               Neural network generative functions (nn->gen-fn)
  serialize.cljs        Serialization/deserialization of traces and choicemaps
  contracts.cljs        GFI contract registry (11 measure-theoretic contracts)
  verify.cljs           Static validator (validate-gen-fn)
  inference/
    importance.cljs     IS, importance resampling, vectorized IS
    mcmc.cljs           MH, MALA, HMC, NUTS, Gibbs, Elliptical Slice, Involutive MCMC, MAP
    smc.cljs            SMC with rejuvenation, csmc
    smcp3.cljs          Sequential Monte Carlo with Probabilistic Program Proposals
    vi.cljs             ADVI, programmable VI (ELBO, IWELBO, wake-sleep)
    adev.cljs           ADEV gradient estimation (reparam + REINFORCE, vectorized, compiled)
    amortized.cljs      Amortized inference via trained neural proposals
    kernel.cljs         Kernel composition (chain, cycle, mix, repeat, seed)
    util.cljs           Weight materialization, normalization, MH accept
    diagnostics.cljs    ESS, R-hat, summary statistics
    analytical.cljs     Analytical handler middleware (wrap-analytical) (L3)
    conjugate.cljs      Conjugate update rules (5 families) (L3)
    auto_analytical.cljs  Auto-analytical elimination from model structure (L3)
    kalman.cljs         Kalman filter middleware for linear-Gaussian SSMs (L3)
    ekf.cljs            Extended Kalman filter for nonlinear SSMs (L3)
    ekf_nd.cljs         N-dimensional EKF with full matrix Jacobians (L3.5)
    hmm_forward.cljs    HMM forward algorithm for discrete latent SSMs (L3)
    enumerate.cljs      Exact enumeration for discrete latent variables (L3)
    fisher.cljs         Fisher information and Cramér-Rao bounds (L3)
    compiled_gradient.cljs  Tensor-native loss-gradient (L4)
    compiled_optimizer.cljs Compiled Adam optimizer (mx/compile-fn) (L4)
    compiled_smc.cljs   Compiled SMC with fused particle operations (L2)
    differentiable.cljs Differentiable inference primitives (L2)
    differentiable_resample.cljs  Differentiable resampling (L2)
    pmcmc.cljs          Particle MCMC (L2)

test/genmlx/
  *_test.cljs                     Unit tests (custom assert helpers, println output)
  level0_certification_test.cljs  68 checks across 10 gates (L0)
  schema_test.cljs                174 checks for schema extraction (L1-M1)
  compiled_simulate_test.cljs     82 checks for compiled simulate (L1-M2)
  partial_compile_test.cljs       92 checks for partial compilation (L1-M3)
  combinator_compile_test.cljs    90 checks for combinator compilation (L1-M5)
  l2_gate_test.cljs               L2 gate tests
  l2_mcmc_test.cljs               L2 compiled MCMC tests
  l3_5_*_test.cljs                L3.5 tests (assess, conjugacy, MVN, regenerate, score)
  l4_certification_test.cljs      41 checks (L4 certification)
  gen_clj_compat_test.cljs        165 tests from Gen.clj
  genjax_compat_test.cljs         73 tests for GenJAX compatibility
  vectorized_test.cljs            Shape correctness, statistical equivalence
  vectorized_benchmark.cljs       Speedup measurements
  benchmark.cljs, gpu_benchmark.cljs
```

## Architecture layers

```
Layer 0: MLX + Runtime    (mlx.cljs, mlx/random.cljs, runtime.cljs — mutable boundary)
Layer 1: Core Data        (choicemap, trace, selection — pure)
Layer 2: GFI & Execution  (protocols, handler, edit, diff — pure)
Layer 3: DSL + Schema     (gen macro, dynamic, schema — pure)
Layer 4: Distributions    (dist/core, dist/macros, dist — 27 types, pure)
Layer 5: Combinators      (Map, Unfold, Switch, etc. — pure)
Layer 6: Inference        (IS, MCMC, SMC, VI, ADEV, MAP + kernels — pure)
Layer 7: Vectorized       (batched execution — pure)
Layer 8: Verification     (contracts, verify — pure)
```

## Key design principles

1. **Purely functional.** State lives in persistent data structures. The only
   mutable boundary is the handler's `volatile!` (isolated within `run-handler`).

2. **Data-driven, open for extension.** Distributions are a single `Distribution`
   record with open multimethods. New distributions can be added via `defdist`
   without modifying core code.

3. **MLX arrays end-to-end.** Values stay as MLX arrays from sampling through
   scoring through gradient computation. Only extract to JS numbers with
   `mx/item` at inference boundaries.

4. **Lazy graph + explicit eval.** MLX operations build a computation graph.
   Call `mx/materialize!` at inference boundaries to evaluate. Direct
   `mx/eval!` and `mx/tidy` are confined to Layer 0 (`mlx.cljs`).

5. **Shape-based batching.** Vectorized inference works by changing array shapes
   (`[N]` instead of `[]`), not by transforming functions with `vmap`. MLX
   broadcasting handles all arithmetic naturally.

6. **Compose, don't duplicate.** Compiled paths compose on existing handlers and
   infrastructure — no parallel implementations. The handler is ground truth.

## How models work

```clojure
(def model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

;; GFI operations
(p/simulate model args)                  ;; => Trace
(p/generate model args constraints)      ;; => {:trace Trace :weight scalar}
(p/update model trace new-constraints)   ;; => {:trace :weight :discard}
(p/regenerate model trace selection)     ;; => {:trace :weight}
(p/assess model args choices)            ;; => {:weight scalar}
(p/project model trace selection)        ;; => scalar
(p/edit model trace edit-request)        ;; => {:trace :weight :discard}

;; Vectorized (runs model body ONCE for N particles)
(dyn/vsimulate model args n key)         ;; => VectorizedTrace
(dyn/vgenerate model args obs n key)     ;; => VectorizedTrace with weights

;; Schema introspection (Level 1)
(:schema model)                          ;; => {:trace-sites [...] :static? true ...}
```

## How the handler system works

The handler system has two parts:

1. **Pure transitions** in `handler.cljs` — 10 state transition functions
   (simulate, generate, update, regenerate, assess, project, plus batched
   variants). Each is `(fn [state addr dist] -> [value state'])`. Zero
   side effects.

2. **Execution runtime** in `runtime.cljs` — `run-handler` wraps a transition
   in a single `volatile!` cell, creating closure-based `trace`/`splice`/`param`
   operations that the `gen` macro binds as local names. Analogous to
   re-frame's app-db: one encapsulated mutable cell, everything else pure.

The `gen` macro injects a hidden runtime parameter. Inside gen bodies,
`trace`, `splice`, and `param` are local bindings (not namespace-qualified
calls), so they work with all Clojure constructs: `map`, `for`, HOFs, closures.

State flows through an immutable map: `{:choices :score :weight :key :constraints ...}`.
The handler never inspects value shapes — this is what makes batched execution
(`[N]`-shaped arrays) work transparently.

PRNG keys are threaded via metadata on gen-fns (`::key`). The single entropy
injection point is `rng/fresh-key` in `mlx/random.cljs`.

## Schema system (Level 1)

The `gen` macro captures the source form. At construction time, `schema.cljs`
walks this quoted form to extract:

- **Trace sites:** address, distribution type, dist-args, dependency set, static?
- **Splice sites:** address, gf reference, dependency set
- **Param sites:** name, default expression
- **Classification:** static?, dynamic-addresses?, has-branches?, has-loops?
- **Dep-order:** topological sort of static trace addresses
- **Return form:** the last body expression

The schema lives on the `DynamicGF` record as `:schema`. It enables Level 1
compilation: static models can be compiled into flat tensor operations because
all trace addresses, distribution types, and dependencies are known ahead of time.

## Vectorized inference

The key insight: MLX operations broadcast naturally. Sample `[N]` values
instead of `[]` at each trace site, and all downstream arithmetic (log-prob,
score accumulation, weight computation) just works.

- `dist-sample-n` multimethod: all distributions have native batch sampling
- Batched handler transitions: structurally identical to scalar ones
- `VectorizedTrace`: choices where leaves hold `[N]`-shaped arrays

**Limitations:** No `splice` in shape-based batched mode (`vsimulate`/`vgenerate`).
`vmap-gf` supports splice via combinator fallback.

## Test conventions

Tests use custom assertion helpers (`assert-true`, `assert-close`, `assert-equal`)
with `println` output. No test framework. Each test file is self-contained and
executable with `bun run --bun nbb`.

Pattern:
```clojure
(println "\n-- test section --")
(let [result (some-operation)]
  (assert-true "description" (predicate result))
  (assert-close "description" expected actual tolerance))
```

After any change, verify:
- All core tests pass (no FAIL lines in output)
- `level0_certification_test.cljs`: 68/68 (L0)
- `schema_test.cljs`: 174/174 (L1-M1)
- `compiled_simulate_test.cljs`: 82/82 (L1-M2)
- `partial_compile_test.cljs`: 92/92 (L1-M3)
- `combinator_compile_test.cljs`: 90/90 (L1-M5)
- `l4_certification_test.cljs`: 41/41 (L4)
- `gen_clj_compat_test.cljs`: 162/165 (3 pre-existing beta/gamma edge cases)
- `genjax_compat_test.cljs`: 73/73

## Common patterns when editing

- **Adding a distribution:** Use `defdist` in `dist.cljs`. Optionally add
  `defmethod dc/dist-sample-n` for batch sampling.
- **Adding inference:** New file in `inference/`. Follow existing patterns
  (pure functions, MLX arrays for weights, `u/materialize-weights` at boundaries).
- **Modifying handlers:** Edit transitions in `handler.cljs`. Keep them pure
  (`[state addr dist] -> [value state']`). The volatile! wrapper is separate.
- **Testing:** Create `test/genmlx/<name>_test.cljs`, run with `bun run --bun nbb`.

## What to avoid

- Don't call `mx/eval!`, `mx/materialize!`, or `mx/item` on values inside
  model bodies during batched execution (breaks vectorization).
- Don't use `mx/eval!` or `mx/tidy` directly outside `mlx.cljs` — use the
  boundary helpers (`materialize!`, `tidy-materialize`, `tidy-run`) — except
  in inference hot loops (Layer 6) where tidy scope and per-iteration cleanup
  are performance-critical.
- Don't introduce mutable state outside the runtime's `volatile!` in `runtime.cljs`.
- Don't import `genmlx.dynamic` from `genmlx.handler` (circular dependency).
- Don't modify existing GFI protocol signatures — everything downstream depends
  on them.

## Milestone delivery protocol

When working on a milestone (L1-M2, L1-M3, etc.):

1. **Spec first, code second.** Before writing any implementation or test code,
   produce a written spec that enumerates EVERY field, behavior, edge case, and
   invariant that "done" means for this milestone. Present the spec and STOP.
   Do not write code until the user has reviewed and approved the spec.

2. **Tests cover the full spec.** After spec approval, write tests for the
   COMPLETE spec — including parts not yet implemented. Failing tests are
   expected and show the gap between current state and done.

3. **Implement until all tests pass.** No partial delivery.

4. **Self-review before presenting.** Before showing results, check every item
   in the spec. Explicitly state completeness as "X/Y spec items done" — not
   just "N/N tests pass" (which hides incomplete specs). If anything is missing
   or deferred, state it upfront, not when asked.

5. **Compiled paths must match handler paths.** For Level 1+ work, compiled
   execution must produce identical traces, scores, and weights as the handler
   path. The handler is ground truth; compilation is optimization.

## Related documents

- `VISION.md` — Compilation ladder levels 0-5, the master development roadmap
- `README.md` — Quick start, examples, public API overview
