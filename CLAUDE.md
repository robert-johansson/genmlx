# GenMLX — Claude Code Project Guide

## What is this project?

GenMLX is a probabilistic programming language in ClojureScript on Node.js (nbb),
using Apple's MLX framework for GPU acceleration on Apple Silicon. It implements
the **Generative Function Interface (GFI)** — the same architecture as Gen.jl
(Julia) and GenJAX (JAX).

~2000 lines of ClojureScript. Purely functional, data-driven, GPU end-to-end.

## How to run things

```bash
# Run any ClojureScript file
npx nbb <filepath>

# Run a test
npx nbb test/genmlx/dist_test.cljs

# Run all core tests
for f in choicemap_test trace_test selection_test handler_test dist_test gen_test combinators_test inference_test; do
  npx nbb "test/genmlx/${f}.cljs"
done

# Compatibility suites (must pass 165/165 and 73/73)
npx nbb test/genmlx/gen_clj_compat_test.cljs
npx nbb test/genmlx/genjax_compat_test.cljs

# Vectorized inference tests + benchmarks
npx nbb test/genmlx/vectorized_test.cljs
npx nbb test/genmlx/vectorized_benchmark.cljs
```

No build step, no compilation. nbb interprets ClojureScript directly.

**Requirements:** macOS with Apple Silicon, Node.js 18+, `npm install` for `@frost-beta/mlx`.

## Project structure

```
src/genmlx/
  mlx.cljs              Thin wrapper over @frost-beta/mlx (arrays, ops, grad, vmap)
  mlx/random.cljs       Functional PRNG keys (split, sample — no global state)
  choicemap.cljs        Hierarchical address -> value maps (Value/Node protocol)
  trace.cljs            Immutable Trace record {gen-fn, args, choices, retval, score}
  selection.cljs        Composable address selection algebra
  protocols.cljs        GFI protocols: simulate, generate, update, regenerate, assess
  handler.cljs          Handler-based execution: pure transitions + volatile! dispatch
  gen.cljc              gen macro (converts fn bodies into DynamicGF)
  dynamic.cljs          DynamicGF record (full GFI via handlers), vsimulate, vgenerate
  dist/core.cljs        Distribution record + open multimethods (sample, log-prob, sample-n)
  dist/macros.cljc      defdist macro (zero-boilerplate distribution definition)
  dist.cljs             14 distributions (gaussian, uniform, bernoulli, beta, gamma, ...)
  combinators.cljs      Map, Unfold, Switch
  vectorized.cljs       VectorizedTrace, resampling, ESS, log-ML utilities
  inference/
    importance.cljs     IS, importance resampling, vectorized IS
    mcmc.cljs           MH, MALA, HMC, NUTS
    smc.cljs            SMC with rejuvenation, vsmc-init
    vi.cljs             ADVI (mean-field Gaussian guide)
    util.cljs           Weight materialization, normalization, MH accept
    diagnostics.cljs    ESS, R-hat, summary statistics

test/genmlx/
  *_test.cljs           Unit tests (custom assert helpers, println output)
  gen_clj_compat_test.cljs   165 tests from Gen.clj
  genjax_compat_test.cljs    73 tests for GenJAX compatibility
  vectorized_test.cljs       Shape correctness, statistical equivalence
  vectorized_benchmark.cljs  Speedup measurements
  benchmark.cljs, gpu_benchmark.cljs
```

## Architecture layers

```
Layer 0: MLX Foundation     (mlx, mlx.random)
Layer 1: Core Data          (choicemap, trace, selection)
Layer 2: GFI & Execution    (protocols, handler)
Layer 3: DSL                (gen macro, dynamic — DynamicGF, vsimulate, vgenerate)
Layer 4: Distributions      (dist/core, dist/macros, dist — 14 types)
Layer 5: Combinators        (Map, Unfold, Switch)
Layer 6: Inference          (IS, MH, MALA, HMC, NUTS, SMC, ADVI + vectorized variants)
Layer 7: Vectorized         (VectorizedTrace, batched execution, 29-122x speedup)
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
   Call `mx/eval!` explicitly to materialize (bounds memory, enables fusion).

5. **Shape-based batching.** Vectorized inference works by changing array shapes
   (`[N]` instead of `[]`), not by transforming functions with `vmap`. MLX
   broadcasting handles all arithmetic naturally.

## How models work

```clojure
(def model
  (gen [args...]
    (let [x (dyn/trace :addr (dist/gaussian 0 1))]  ;; sample or constrain
      (dyn/splice :sub sub-model arg1 arg2)          ;; call sub-generative-fn
      x)))

;; GFI operations
(p/simulate model args)                  ;; => Trace
(p/generate model args constraints)      ;; => {:trace Trace :weight scalar}
(p/update model trace new-constraints)   ;; => {:trace :weight :discard}
(p/regenerate model trace selection)     ;; => {:trace :weight}

;; Vectorized (runs model body ONCE for N particles)
(dyn/vsimulate model args n key)         ;; => VectorizedTrace
(dyn/vgenerate model args obs n key)     ;; => VectorizedTrace with weights
```

## How the handler system works

The handler is the heart of GenMLX. When `gen` body code calls `dyn/trace`,
it dispatches to whichever handler is active (simulate, generate, update,
regenerate, or batched variants). Each handler is a **pure state transition**
`(fn [state addr dist] -> [value state'])` wrapped in a thin volatile! dispatcher.

State flows through an immutable map: `{:choices :score :weight :key :constraints ...}`.
The handler never inspects value shapes — this is what makes batched execution
(`[N]`-shaped arrays) work transparently.

## Vectorized inference

The key insight: MLX operations broadcast naturally. Sample `[N]` values
instead of `[]` at each trace site, and all downstream arithmetic (log-prob,
score accumulation, weight computation) just works.

- `dist-sample-n` multimethod: 7 distributions have native batch sampling,
  the rest fall back to sequential
- Batched handler transitions: structurally identical to scalar ones
- `VectorizedTrace`: choices where leaves hold `[N]`-shaped arrays

**Benchmarks (N=100, 5-site model):**
- dist-sample-n: 29x speedup
- vgenerate: 61-122x speedup
- Vectorized IS: 81x
- Vectorized SMC init: 65x

**Limitations:** No `splice` in batched mode, no vectorized `update`/`regenerate`
yet, rejection-sampling distributions (beta, gamma, poisson, student-t,
dirichlet) fall back to sequential.

## Test conventions

Tests use custom assertion helpers (`assert-true`, `assert-close`, `assert-equal`)
with `println` output. No test framework. Each test file is self-contained and
executable with `npx nbb`.

Pattern:
```clojure
(println "\n-- test section --")
(let [result (some-operation)]
  (assert-true "description" (predicate result))
  (assert-close "description" expected actual tolerance))
```

After any change, verify:
- All core tests pass (no FAIL lines in output)
- `gen_clj_compat_test.cljs`: 165/165
- `genjax_compat_test.cljs`: 73/73
- `vectorized_test.cljs`: all pass

## Common patterns when editing

- **Adding a distribution:** Use `defdist` in `dist.cljs`. Optionally add
  `defmethod dc/dist-sample-n` for batch sampling.
- **Adding inference:** New file in `inference/`. Follow existing patterns
  (pure functions, MLX arrays for weights, `u/materialize-weights` at boundaries).
- **Modifying handlers:** Edit transitions in `handler.cljs`. Keep them pure
  (`[state addr dist] -> [value state']`). The volatile! wrapper is separate.
- **Testing:** Create `test/genmlx/<name>_test.cljs`, run with `npx nbb`.

## What to avoid

- Don't call `mx/eval!` or `mx/item` on values inside model bodies during
  batched execution (breaks vectorization).
- Don't introduce mutable state outside the handler's `volatile!`.
- Don't import `genmlx.dynamic` from `genmlx.handler` (circular dependency).
- Don't modify existing GFI protocol signatures — everything downstream depends
  on them.

## Related documents

- `README.md` — Quick start, examples, public API overview
- `ARCHITECTURE.md` — Internal design patterns and refactoring history
- `GAPS.md` — Long-term roadmap, gap analysis vs Gen.jl and GenJAX
