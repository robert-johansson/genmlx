# GenMLX — Claude Code Project Guide

## What is this project?

GenMLX is a probabilistic programming language in ClojureScript on Node.js (nbb),
using Apple's MLX framework for GPU acceleration. It implements the **Generative
Function Interface (GFI)** — the mathematical contract for probabilistic computation
from Cusumano-Towner's 2020 MIT PhD thesis — the same architecture as Gen.jl
(Julia) and GenJAX (JAX).

The thesis: probabilistic programming and functional programming are the same thing.
ClojureScript's immutable data, open multimethods, and macro system map perfectly
onto the GFI's mathematical structure. MLX's lazy graphs, unified memory, and
broadcasting reinforce this — functional-style array programming without penalty.

~29,600 lines of ClojureScript across 74 source files. Purely functional,
data-driven, GPU end-to-end.

## Compilation ladder

GenMLX follows a 5-level compilation ladder, progressively moving work from the
host interpreter into fused MLX computation graphs:

```
Level 0: Shape-based batching       ← DONE (certified, 68/68)
Level 1: Compiled gen functions      ← DONE (506+ tests)
  M1: Schema extraction from source forms
  M2: Full compilation for static models (noise transforms + mx/compile-fn)
  M3: Partial prefix compilation (static prefix + interpreted suffix)
  M4: Branch rewriting (if/if-not with same addr+dist → mx/where)
  M5: Combinator compilation (fused Map/Unfold/Scan loops)
Level 2: Compiled inference sweeps   ← DONE (881+ tests)
Level 3: Auto-analytical elimination ← DONE (426 tests, 33.5x variance reduction)
  7 conjugate families detected statically
  Kalman chain detection via affine dependency analysis
  Rao-Blackwellization for partial conjugacy
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
bun run --bun nbb test/genmlx/gen_clj_compat_test.cljs    # 165/165
bun run --bun nbb test/genmlx/genjax_compat_test.cljs      # 73/73

# Vectorized inference tests + benchmarks
bun run --bun nbb test/genmlx/vectorized_test.cljs
bun run --bun nbb test/genmlx/vectorized_benchmark.cljs
```

No build step, no compilation. nbb interprets ClojureScript directly.

**Requirements:** macOS with Apple Silicon, Bun (or Node.js 18+), `npm install`
for `@mlx-node/core` and `@mlx-node/lm`. Malli (included as git submodule,
temporary until upstream nbb/malli releases align).

## Project structure

```
src/genmlx/
  # Layer 0: MLX + Runtime (1,144 lines — the mutable boundary)
  mlx.cljs              Thin wrapper over @mlx-node/core (~200 ops, memory management,
                        auto-cleanup, tidy, grad/vmap/compile-fn). Only file that
                        touches the native MLX library.
  mlx/random.cljs       Functional PRNG keys (split, sample — no global state)
  runtime.cljs          Execution runtime: run-handler wraps a single volatile! cell,
                        creates trace/splice/param closures as a JS runtime object
  dispatch.cljs         IDispatcher protocol, stack resolution, with-handler, with-dispatch

  # Layer 1: Core Data (426 lines — pure immutable structures)
  choicemap.cljs        Hierarchical address→value maps (Value leaf / Node internal,
                        stack/unstack for batched execution)
  trace.cljs            Immutable Trace record [gen-fn args choices retval score]
  selection.cljs        Composable address selection algebra (all, none, SelectAddrs,
                        Hierarchical, Complement)
  diff.cljs             Change tracking for incremental computation (no-change sentinel)

  # Layer 2: GFI & Execution (653 lines — pure protocols and transitions)
  protocols.cljs        10 GFI protocols: IGenerativeFunction (simulate), IGenerate,
                        IAssess, IUpdate, IRegenerate, IPropose, IProject,
                        IUpdateWithDiffs, IHasArgumentGrads, IBatchedSplice
  handler.cljs          Pure state transitions: 6 scalar (simulate, generate, assess,
                        update, regenerate, project) + 4 batched variants.
                        Each is (fn [state addr dist] → [value state']).
                        merge-sub-result for splice. combinator-batched-fallback.
  edit.cljs             Parametric edit interface: ConstraintEdit, SelectionEdit,
                        ProposalEdit (for SMCP3 reversible kernels)
  tensor_trace.cljs     TensorTrace — flat [K] tensor-backed trace for compiled paths.
                        TensorChoiceMap provides IChoiceMap over the tensor.

  # Layer 3: DSL + Schema (1,883 lines — gen macro, model construction, introspection)
  gen.cljc              gen macro: injects ᐩrt runtime parameter, binds trace/splice/param
                        as local names, captures quoted source form for schema extraction.
                        Calls dynamic/make-gen-fn with body-fn and source.
  dynamic.cljs          DynamicGF record [body-fn source schema]. Implements all 10 GFI
                        protocols via 4-level dispatcher stack:
                        1. custom-dispatcher (with-handler / with-dispatch metadata)
                        2. analytical-dispatcher (L3 conjugacy, when obs constrained + not in grad)
                        3. compiled-dispatcher (L1 compiled or prefix paths from schema)
                        4. handler-dispatcher (L0 fallback, always resolves)
                        Execution paths: handler, compiled, prefix, analytical.
                        vsimulate/vgenerate/vupdate/vregenerate bypass dispatcher.
                        make-gen-fn: schema extraction → compilation attempts → conjugacy
                        augmentation → DynamicGF construction.
  schema.cljs           Schema extraction: walks quoted source form to extract trace sites,
                        splice sites, param sites, loop structure, branch detection,
                        dependency tracking via binding environment, topological sort.
                        Classification: static? (all keyword addrs, no branches/loops/splices)
  schemas.cljs          Malli schemas for handler state, GFI returns, trace structure
  inspect.cljs          Model introspection: compilation level, dispatch resolution, conjugacy

  # Layer 4: Distributions (2,003 lines — 31 types, open for extension)
  dist/core.cljs        Distribution record [type params]. 6 open multimethods:
                        dist-sample*, dist-log-prob, dist-reparam, dist-support,
                        dist-log-prob-support, dist-sample-n*. Mixture and Product
                        distributions. GFI bridge (dist-simulate, dist-generate, etc.)
  dist/macros.cljc      defdist macro: generates constructor + sample + log-prob methods
  dist.cljs             31 distributions:
                        Continuous (20): gaussian/normal, uniform, beta-dist, gamma-dist,
                          exponential, laplace, student-t, log-normal, cauchy, inv-gamma,
                          truncated-normal, multivariate-normal, broadcasted-normal,
                          gaussian-vec, von-mises, wrapped-cauchy, wrapped-normal,
                          piecewise-uniform, wishart, inv-wishart
                        Discrete (7): bernoulli/flip, categorical, poisson, geometric,
                          neg-binomial, binomial, discrete-uniform
                        Structural (4): delta, mixture, product, iid/iid-gaussian

  # Layer 5: Combinators (3,022 lines — 10 combinators, full GFI)
  combinators.cljs      Map, Unfold, Switch, Scan, Mask, Mix, Recurse, Contramap,
                        MapRetval/Dimap. Three execution tiers for Map/Unfold/Scan:
                        fused (single Metal dispatch), compiled (per-element), handler.
                        IBatchedSplice for Unfold/Switch/Scan/Mix.
  vmap.cljs             VmapCombinator (vmap-gf / repeat-gf): vectorized map via
                        physical loop or batched handler. [N]-shaped leaf arrays.

  # Layer 6: Inference (8,626 lines — 35+ algorithms across 25 files)
  inference.cljs        Re-export facade for all inference algorithms
  inference/
    importance.cljs     IS, vectorized IS (10-100x faster), importance resampling
    mcmc.cljs           MH (standard, custom, compiled, fused, vectorized), Gibbs,
                        Involutive MCMC, MALA (fused, vectorized, dual averaging),
                        HMC (mass matrix, fused, vectorized), NUTS, Elliptical Slice,
                        MAP (vectorized multi-restart). ~2,800 lines.
    smc.cljs            SMC (systematic/residual/stratified resampling, ESS threshold,
                        MH rejuvenation), CSMC, smc-unfold, batched-smc-unfold, vsmc
    smcp3.cljs          SMC with Probabilistic Program Proposals (forward/backward kernels)
    vi.cljs             Mean-field ADVI, compiled VI, programmable VI with 6 objectives
                        (ELBO, IWELBO, PWake, QWake, REINFORCE, VIMCO)
    adev.cljs           ADEV gradient estimation (reparam + REINFORCE, vectorized)
    amortized.cljs      Neural proposal training, neural importance sampling
    kernel.cljs         Composable kernels: chain, repeat, cycle, mix, seed.
                        with-reversal for symmetric kernels. DSL: random-walk, prior,
                        proposal, gibbs. collect-samples for burn/thin loops.
    util.cljs           Score function construction (3-tier: tensor-native > compiled-generate
                        > GFI handler), MH acceptance, resampling, weight normalization
    diagnostics.cljs    ESS (autocorrelation-based), R-hat, summary statistics
    analytical.cljs     wrap-analytical: ring-style handler middleware
    conjugate.cljs      Conjugate update rules: normal-normal, beta-bernoulli,
                        gamma-poisson, gamma-exponential, dirichlet-categorical
    auto_analytical.cljs  Auto-analytical elimination from model structure: address-dispatch,
                        7 conjugate families + MVN-normal + normal-iid-normal,
                        auto-Kalman handlers (cascade predictions, loading/offset),
                        generate and regenerate modes
    kalman.cljs         Kalman filter middleware for linear-Gaussian SSMs
    ekf.cljs            Extended Kalman filter (linearize via mx/grad)
    ekf_nd.cljs         N-dimensional EKF with scalar-decomposed covariance
    hmm_forward.cljs    HMM forward algorithm for discrete latent SSMs
    enumerate.cljs      Exact enumeration for discrete latent variables
    exact.cljs          Exact tensor enumeration: one model run builds full joint via
                        broadcasting. ExactGF with full GFI (all discrete latents
                        marginalized). exact-posterior, exact-joint, thinks, observes, pr.
    fisher.cljs         Fisher information (finite differences on autodiff gradient),
                        Laplace approximation, natural gradient, Cramér-Rao bounds
    compiled_gradient.cljs  Differentiable MH/SMC chains (straight-through via mx/where)
    compiled_optimizer.cljs Compiled Adam: 3-tier score (tensor-native > compiled-generate
                        > handler), fused MCMC+optimization
    compiled_smc.cljs   Compiled bootstrap PF with gumbel-softmax differentiable resampling
    differentiable.cljs Differentiable IS loss and gradient
    differentiable_resample.cljs  Gumbel-top-k, gumbel-softmax, soft-resample
    pmcmc.cljs          Particle MCMC: PMMH and particle Gibbs

  # Layer 7: Compiled Paths (3,941 lines — L1-L4 compilation pipeline)
  compiled.cljs         Noise transform system (10 distributions: gaussian, uniform,
                        bernoulli, exponential, log-normal, delta, laplace, cauchy,
                        iid-gaussian). Expression compiler (resolve ~45 MLX ops).
                        L1-M2 full compilation: reduce over step-fns + mx/compile-fn.
                        L1-M3 prefix: extract-prefix-sites + replay transitions.
                        L1-M4 branch rewriting: analyze-rewritable-branch + mx/where.
                        L0 compiled unfold/particle filter for temporal models.
  compiled_ops.cljs     Compiled GFI operations for all tiers (M2/M3/M4):
                        generate, update, assess, project, regenerate.
                        Fused loop compilation (unfold, scan, map).
                        Tensor-native score functions. SMC extend steps.
  compiled_gen.cljs     Compiled log-ML estimators (IS-based, keyed/unkeyed variants)
  rewrite.cljs          Algebraic graph rewriting: IRewriteRule protocol,
                        ConjugacyRule, KalmanRule, RaoBlackwellRule.
                        build-analytical-plan: progressive rule application.
  affine.cljs           Affine dependency analysis: determines if expression is
                        affine in target symbol (conservative, no false positives).
                        detect-kalman-chains for linear-Gaussian sequences.
  conjugacy.cljs        Static conjugate pair detection from schema: 7 families
                        (normal-normal, beta-bernoulli, gamma-poisson, gamma-exponential,
                        dirichlet-categorical, mvn-normal, normal-iid-normal)
  dep_graph.cljs        Dependency graph: DAG from schema, d-separation (Bayes-Ball),
                        Markov blankets, independent blocks for parallel Gibbs
  method_selection.cljs Automatic method selection: decision tree
                        (exact > kalman > smc > handler-is > hmc > vi > handler-is)

  # Layer 8: Supporting Systems (2,729 lines)
  vectorized.cljs       VectorizedTrace [gen-fn args choices score weight n-particles retval],
                        systematic resampling (GPU O(N) via cumsum+searchsorted),
                        resample-vtrace, vtrace-log-ml-estimate, merge-by-mask, ESS
  gradients.cljs        choice-gradients (d(score)/d(choice_i)), score-gradient (flat params)
  learning.cljs         Parameter stores, SGD (stateless), Adam (with momentum),
                        train loop, simulate/generate-with-params, wake-sleep learning
  custom_gradient.cljs  CustomGradientGF: deterministic GF with stop-gradient surrogate
                        trick for custom backward passes
  nn.cljs               NeuralNetGF: wraps MLX nn.Module as deterministic GF.
                        Layer constructors (linear, sequential, relu, gelu, etc.).
                        nn→gen-fn bridge. value-and-grad and optimizer factory.
  serialize.cljs        JSON v1 serialization: choices-only and full-trace formats.
                        Gen-fns NOT serialized — reconstruction via p/generate.
  contracts.cljs        11 measure-theoretic GFI contracts (deprecated, subsumed by gfi.cljs)
  verify.cljs           Static source analysis (DML restrictions: no external randomness,
                        no mutation, no HOF gen-fns) + runtime validation (duplicate
                        addresses, non-finite scores, halting)
  gfi.cljs              GFI algebraic theory: 53 laws from the thesis covering all GFI
                        operations, compositionality, gradients, inference, compiled path
                        equivalence. verify runs all laws against a model.
                        strip-compiled forces handler path for equivalence testing.
  fit.cljs              One-call entry point: method selection → hyperparameter tuning →
                        inference → optional parameter learning
  dev.cljs              Dev-mode Malli validation at GFI boundaries (start!/stop!)

  # Layer 9: LLM Integration (2,176 lines — neural networks as generative functions)
  llm/
    backend.cljs        Async wrapper over @mlx-node/lm: load-model, encode/decode,
                        forward-pass, next-token-logprobs, KV cache (init/reset/prefill/step),
                        generate-text via chat API
    core.cljs           make-llm-gf: wraps LLM as DynamicGF via gen macro. Each token is
                        a trace site (:t0, :t1, ...) sampling from dist/categorical(logits).
                        Uses KV cache for O(1) per-step. All GFI operations work automatically.
    grammar.cljs        Regex→NFA (Thompson)→DFA (subset construction)→alive states→
                        token mask precomputation→handler middleware (wrap-grammar).
                        BPE byte decoder for token-to-text mapping.
                        constrain: dispatch/with-handler composition.
    bytes.cljs          TokenByteTrie: prefix trie mapping byte sequences to token IDs.
                        byte-logprobs: marginalizes token logprobs to byte level via
                        logsumexp-gather on trie subtrees.
                        make-byte-llm-gf: trace sites :b0, :b1, ... sampling bytes.
                        constrain-bytes: DFA-constrained byte-level generation.
    codegen.cljs        Reader-as-grammar: edamame classifies prefix as complete/incomplete/
                        invalid. make-reader-constrained-gf: guaranteed valid ClojureScript.
                        generate-cljs (token-level or byte-level), eval-cljs (SCI sandbox),
                        synthesize-loop (generate-verify-revise), generate-and-score
                        (p/generate for principled scoring), generate-and-rank.
    msa.cljs            Model Synthesis Architecture: template mode (fine-tuned model +
                        regex parsing) and knowledge mode (base model + Instaparse grammar).
                        eval-model (SCI eval → DynamicGF), score-model (p/generate → log-ML),
                        synthesize-and-rank, importance-sample, msa end-to-end pipeline.

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
Layer 0: MLX + Runtime    (mlx.cljs, mlx/random.cljs, runtime.cljs, dispatch.cljs)
         Rust: genmlx.rs  (~105 NAPI exports, Either<&MxArray,f64> type coercion)
Layer 1: Core Data        (choicemap, trace, selection, diff — pure)
Layer 2: GFI & Execution  (protocols, handler, edit, tensor_trace — pure)
Layer 3: DSL + Schema     (gen macro, dynamic, schema, schemas, inspect — pure)
Layer 4: Distributions    (dist/core, dist/macros, dist — 31 types, pure)
Layer 5: Combinators      (combinators, vmap — 10 combinators, pure)
Layer 6: Inference         (25 files, 35+ algorithms — pure)
Layer 7: Compiled Paths   (compiled, compiled_ops, rewrite, affine, conjugacy, dep_graph,
                           method_selection — pure)
Layer 8: Supporting       (vectorized, gradients, learning, nn, serialize, gfi, verify,
                           contracts, fit, dev — pure except dev.cljs atoms)
Layer 9: LLM Integration  (llm/backend, core, grammar, bytes, codegen, msa — pure except
                           KV cache mutation in backend.cljs)
```

Strict dependency direction: higher layers depend on lower, never the reverse.
The only circular dependency risk is dynamic↔handler (resolved via the executor
pattern: runtime.cljs takes an `:executor` function in the state map, avoiding
direct import of dynamic.cljs).

## Key design principles

1. **Purely functional.** The only mutable boundaries are:
   - The handler's `volatile!` in `runtime.cljs` (scoped to a single `run-handler` call)
   - Two atoms in `mlx.cljs` (`tidy-depth`, `grad-depth` for nesting counters)
   - The `dispatch-fn` and `validate-fn` atoms (extension points for dev mode)
   - KV cache mutation in `llm/backend.cljs` (always in try/finally)

2. **Data-driven, open for extension.** Distributions are a single `Distribution`
   record with open multimethods. New distributions via `defdist`. New execution
   strategies via `dispatch/with-handler` or `dispatch/with-dispatch`. Grammar
   constraints compose via the same handler middleware as analytical inference.

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

7. **53 algebraic laws.** The GFI algebraic theory (`gfi.cljs`) encodes 53 laws
   from the thesis covering all operations, compositionality, gradients, and
   compiled path equivalence. `strip-compiled` forces handler path for testing.

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

;; GFI operations (10 protocols)
(p/simulate model args)                  ;; => Trace
(p/generate model args constraints)      ;; => {:trace Trace :weight scalar}
(p/update model trace new-constraints)   ;; => {:trace :weight :discard}
(p/regenerate model trace selection)     ;; => {:trace :weight}
(p/assess model args choices)            ;; => {:retval :weight}
(p/project model trace selection)        ;; => scalar
(p/propose model args)                   ;; => {:choices :weight :retval}
(edit/edit model trace edit-request)     ;; => {:trace :weight :discard :backward-request}

;; Vectorized (runs model body ONCE for N particles)
(dyn/vsimulate model args n key)         ;; => VectorizedTrace
(dyn/vgenerate model args obs n key)     ;; => VectorizedTrace with weights
(dyn/vupdate model vtrace constraints key) ;; => {:vtrace :weight :discard}
(dyn/vregenerate model vtrace selection key) ;; => {:vtrace :weight}

;; Schema introspection (Level 1)
(:schema model)                          ;; => {:trace-sites [...] :static? true ...}
(inspect/inspect model)                  ;; => {:compilation :L1-M2 :dispatch {...} ...}
```

## How the gen macro works

The `gen` macro (29 lines in `gen.cljc`) transforms user code:

```clojure
(gen [x] (let [s (trace :slope (dist/gaussian 0 10))] ...))
```

expands to:

```clojure
(dynamic/make-gen-fn
  (fn [ᐩrt x]                         ;; hidden runtime parameter injected
    (let [trace  (.-trace ᐩrt)         ;; closures from run-handler's volatile!
          splice (.-splice ᐩrt)
          param  (.-param ᐩrt)]
      (let [s (trace :slope (dist/gaussian 0 10))] ...)))
  '([x] (let [s (trace :slope ...)] ...)))  ;; quoted source for schema extraction
```

Key properties:
- `trace`, `splice`, `param` are **local bindings** (not namespace-qualified),
  so they work naturally with `map`, `for`, HOFs, closures
- The quoted source form enables static analysis without execution
- `make-gen-fn` extracts schema → attempts compilation → augments with conjugacy
  → returns `DynamicGF [body-fn source schema]`

## How the handler system works

The handler system has two parts:

1. **Pure transitions** in `handler.cljs` — 6 scalar state transition functions
   (simulate, generate, assess, update, regenerate, project) + 4 batched variants.
   Each is `(fn [state addr dist] -> [value state'])`. Zero side effects.

2. **Execution runtime** in `runtime.cljs` — `run-handler` wraps a transition
   in a single `volatile!` cell, creating closure-based `trace`/`splice`/`param`
   operations packaged as a JS object `#js {:trace :splice :param}`. The gen
   macro destructures this as local bindings. Analogous to re-frame's app-db:
   one encapsulated mutable cell, everything else pure.

**Handler state shape per mode:**

| Mode | Keys |
|------|------|
| simulate | `:key :choices :score :executor` |
| generate | + `:weight :constraints` |
| assess | + `:weight :constraints` |
| update | + `:weight :constraints :old-choices :discard` |
| regenerate | + `:weight :old-choices :selection` |
| project | + `:weight :old-choices :selection :constraints` |

Batched variants add `:batch-size` (int) and `:batched?` (true).
The handler never inspects value shapes — MLX broadcasting handles
`[N]`-shaped arrays transparently.

**Dispatcher stack (4-level priority, first non-nil wins):**

1. **custom-dispatcher** — `::custom-dispatch` or `::custom-transition` metadata
2. **analytical-dispatcher** — L3 conjugacy (generate/assess/regenerate only,
   when conjugate obs constrained and not inside `mx/in-grad?`)
3. **compiled-dispatcher** — L1 compiled or prefix paths (checks schema keys)
4. **handler-dispatcher** — L0 fallback, always resolves

PRNG keys are threaded via metadata on gen-fns (`::key`). Every sample splits
the key: one half for the sample, one half for the next operation. The single
entropy injection point is `rng/fresh-key` in `mlx/random.cljs`.

## Noise transform system (Level 1 compilation)

The key mechanism enabling compiled execution. For each distribution type:
1. Pre-generate standardized noise outside `mx/compile-fn` (N(0,1) or U(0,1))
2. Inside the compiled function, apply a pure deterministic transform
3. Compute log-probability via a pure function

10 distributions supported: gaussian, uniform, bernoulli, exponential, log-normal,
delta, laplace, cauchy, iid-gaussian. The expression compiler (`compile-expr`)
resolves ~45 MLX operations from namespace-qualified symbols in the source form.

## Schema system (Level 1)

The `gen` macro captures the source form. At construction time, `schema.cljs`
walks this quoted form to extract:

- **Trace sites:** address, distribution type, dist-args, dependency set, static?
- **Splice sites:** address, gf reference, dependency set
- **Param sites:** name, default expression
- **Loop sites:** type, bindings, homogeneous?, rewritable?, count-arg-idx
- **Classification:** static?, dynamic-addresses?, has-branches?, has-loops?
- **Dep-order:** topological sort of static trace addresses
- **Return form:** the last body expression

A model is **static** when all trace addresses are keyword literals, no branches,
no loops, no splices. Static models get L1-M2 full compilation.

The schema lives on the `DynamicGF` record as `:schema`. At construction time,
`make-gen-fn` runs the full pipeline:
1. `schema/extract-schema` — static analysis of source form
2. Compilation attempt — M2 (static), M4 (branch rewriting), or M3 (prefix)
3. `conjugacy/augment-schema-with-conjugacy` — detect conjugate pairs
4. `rewrite/build-analytical-plan` — Kalman chains, rewrite rules, auto-handlers

## Vectorized inference

The key insight: MLX operations broadcast naturally. Sample `[N]` values
instead of `[]` at each trace site, and all downstream arithmetic (log-prob,
score accumulation, weight computation) just works.

- `dist-sample-n` multimethod: all distributions have native batch sampling
- Batched handler transitions: structurally identical to scalar ones
- `VectorizedTrace`: choices where leaves hold `[N]`-shaped arrays

**Limitations:** No `splice` in shape-based batched mode (`vsimulate`/`vgenerate`).
`vmap-gf` supports splice via combinator fallback. No `mx/item` in model bodies
during batched execution (breaks vectorization).

## LLM integration

LLMs are wrapped as standard DynamicGF via the `gen` macro. Each token becomes a
trace site (`:t0`, `:t1`, ...) sampling from `dist/categorical(logits)`. All GFI
operations work automatically — simulate generates text, generate constrains tokens,
assess scores text, update/regenerate modify traces.

**Grammar constraints** compose via the same `dispatch/with-handler` mechanism used
for analytical inference. `wrap-grammar` is ring-style middleware that intercepts
categorical distributions and masks logits per DFA state.

**Three constraint levels:**
- Token-level (grammar.cljs): regex→DFA→token mask, operates on full vocab (~151K)
- Byte-level (bytes.cljs): TokenByteTrie + DFA, operates on individual bytes (~256)
- Reader-level (codegen.cljs): edamame parser as incremental grammar, guaranteed
  valid ClojureScript

**Code synthesis** (codegen.cljs): generate-verify-revise loop. Reader-as-grammar
ensures syntactic validity. SCI evaluates in the same runtime. generate-and-score
uses `p/generate` with full constraints for principled model-weight scoring.

**Model synthesis** (msa.cljs): LLM generates probabilistic programs from task
descriptions, evaluates with SCI, scores against data via `p/generate`, ranks
candidates by log-ML. Two modes: template (fine-tuned + regex) and knowledge
(base model + Instaparse grammar).

## Rust NAPI boundary (genmlx.rs)

~105 NAPI-exported functions in `mlx-node/crates/mlx-core/src/genmlx.rs`.
The core pattern: `Either<&MxArray, f64>` accepts both MLX arrays and JS numbers
transparently. `Vec<f64>` for shapes (no BigInt64Array needed). This makes
`mlx.cljs` extremely thin — most ops are direct property references to Rust exports.

Note: MLX has no float64 or int64. `mlx.cljs` silently aliases float64→float32
and int64→int32.

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
- `gen_clj_compat_test.cljs`: 165/165
- `genjax_compat_test.cljs`: 73/73

## Common patterns when editing

- **Adding a distribution:** Use `defdist` in `dist.cljs`. Implement `dist-sample*`
  and `dist-log-prob`. Optionally add `defmethod dc/dist-sample-n*` for batch
  sampling, `dc/dist-reparam` for reparameterized gradients, `dc/dist-support`
  for enumerable support.
- **Adding inference:** New file in `inference/`. Follow existing patterns
  (pure functions, MLX arrays for weights, `u/materialize-weights` at boundaries).
- **Adding an execution strategy:** Write a handler transition `(fn [state addr dist])`
  and attach via `dispatch/with-handler`. For full op-level control, use
  `dispatch/with-dispatch`. See `inference/exact.cljs` for the canonical example.
- **Modifying handlers:** Edit transitions in `handler.cljs`. Keep them pure
  (`[state addr dist] -> [value state']`). The volatile! wrapper is separate.
- **Adding a combinator:** Create a record implementing the GFI protocols.
  Implement `IBatchedSplice` for vectorized inference support.
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
  Use the executor pattern instead.
- Don't modify existing GFI protocol signatures — everything downstream depends
  on them.
- Don't add `ensure-mx` or `to-big-shape` calls — Rust `Either<&MxArray, f64>`
  and `Vec<f64>` handle type coercion and shape conversion at the NAPI boundary.
  Layer 0 ops accept both MxArray and JS number directly.

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

## TODO protocol

Project work is tracked in `dev/TODO.md` (read it at the start of each session).
Priority levels: P0 (fix now), P1 (build next), P2 (captured ideas), P3 (explorations).

Every P0/P1 entry has a **"done means"** checklist. When working on a task:
1. Read the full entry including context and "done means"
2. Implement using agent teams when the task is non-trivial
3. Review agent checks every "done means" criterion before presenting
4. Never declare a task complete until every checkbox is confirmed
5. Update `dev/TODO.md` status when starting and completing work

When the user captures a new idea, add it as a P2 entry with 2-3 sentences of
context. Promote to P1 when ready to spec (add "done means" criteria at that point).

## Related documents

- `dev/TODO.md` — Project tracker (P0-P3 priorities, "done means" checklists)
- `VISION.md` — Compilation ladder levels 0-5, the master development roadmap
- `README.md` — Quick start, examples, public API overview
- `dev/docs/` — System specification derived from line-by-line source code review
  (8,962 lines across 10 files). Read these when you need deep understanding of
  a specific subsystem — they document every public function, protocol, and record.
  Covers:
  - `SPEC_SYSTEM_OVERVIEW.md` — Complete system map, file inventory, design principles
  - `SPEC_CORE_ARCHITECTURE.md` — Layers 0-3: MLX, runtime, handlers, dispatch, gen macro
  - `SPEC_DISTRIBUTIONS_AND_COMBINATORS.md` — 31 distributions, 10 combinators
  - `SPEC_INFERENCE.md` — 35+ algorithms, every public function documented
  - `SPEC_COMPILED_PATHS.md` — L0-L4 compilation pipeline
  - `SPEC_LLM_INTEGRATION.md` — 6 LLM modules
  - `SPEC_SUPPORTING_SYSTEMS.md` — Vectorized, gradients, learning, 53 laws, verify
  - `SPEC_RUST_NAPI_BOUNDARY.md` — ~105 NAPI exports, type coercion
  - `INNOVATION_UNIFIED_STACK.md` — Future: LLMs as GFs, GRPO with GFI rewards
  - `INNOVATION_PROGRAM_SYNTHESIS.md` — Future: program synthesis as inference
