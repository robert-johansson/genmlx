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

~43,100 lines of ClojureScript across 105 source files. Purely functional,
data-driven, GPU end-to-end.

## Three-layer purity architecture

GenMLX has three layers of purity. Understanding this is essential for all work:

```
Layer A: Pure ClojureScript  — GFI protocols, handlers, inference (values)
Layer B: Pure MLX Graphs     — lazy computation descriptions (also values!)
Layer C: GPU Execution       — mx/eval! dispatches the graph to Metal (side effect)
```

**Layer B is the key insight.** MLX operations like `mx/add` do not compute — they
build lazy computation graph nodes. The graph is a value, like a Clojure data
structure. No GPU work occurs until `mx/eval!` is called. This means most of
`mlx.cljs` is purely functional: graph construction is value manipulation.

**`mx/eval!` is the sole side effect.** All GPU dispatch flows through `eval!`
(or wrappers like `item`, `->clj`, `materialize!`). Everything else — arithmetic,
reductions, autograd, vmap, compile — builds lazy graphs.

**mlx-node is at the heart of GenMLX.** The Rust/NAPI layer (214 @genmlx/core function exports, pinned by the coverage matrix; 5 crates)
is not "mutable substrate we contain" — it is a functional graph engine that aligns
naturally with ClojureScript's value semantics. `mlx.cljs` is the thin membrane
between them; `Either<&MxArray, f64>` in Rust handles type coercion so CLJS
doesn't need to.

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
Level 3: Auto-analytical elimination ← DONE (426 tests, exact marginal LL to float32 floor)
  7 conjugate families detected statically
  Kalman chain detection via affine dependency analysis
  Joint linear-Gaussian regression (coupled/affine multi-latent) elimination
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
for f in choicemap_test trace_test selection_test handler_test dist_test combinators_test inference_test; do
  bun run --bun nbb "test/genmlx/${f}.cljs"
done

# Native/membrane contract guard — MUST run after any mlx-node rebuild/bump.
# These exercise mx/array JS-array shaping (take/squeeze, value_and_grad,
# multivariate-normal cholesky) and the mx/clip Either<&MxArray,f64> bounds
# contract. A stricter MLX binary turns a malformed-array CLJS bug into a hard
# SIGTRAP; a more lenient one silently produces NaN/garbage; a narrowed NAPI
# signature rejects valid bounds. Either way these suites catch it, so a stale
# or drifted binary can never silently mask the regression. membrane_coverage_test
# is the SURFACE drift guard (genmlx-0vwn): it partitions every @mlx-node/core
# function export into wrapped ∪ intentional-omissions (both directions), so an
# upstream add/delete/rename surfaces as a red test naming exactly what moved.
# See docs/membrane-coverage.md for the matrix.
for f in exact_test gradient_fd_test score_gradient_test clip_contract_test membrane_coverage_test; do
  bun run --bun nbb "test/genmlx/${f}.cljs"
done

# Level 0 certification (must pass 68/68)
bun run --bun nbb test/genmlx/level0_certification_test.cljs

# Level 1 compilation tests
bun run --bun nbb test/genmlx/schema_test.cljs                # 227/227 (L1-M1)
bun run --bun nbb test/genmlx/compiled_simulate_test.cljs      # 82/82  (L1-M2)
bun run --bun nbb test/genmlx/partial_compile_test.cljs        # 92/92  (L1-M3)
bun run --bun nbb test/genmlx/combinator_compile_test.cljs     # 92/92  (L1-M5)

# Compatibility suites
bun run --bun nbb test/genmlx/gen_clj_compat_test.cljs    # 356/356
bun run --bun nbb test/genmlx/genjax_compat_test.cljs      # 73/73

# Vectorized inference tests + benchmarks
bun run --bun nbb test/genmlx/vectorized_test.cljs
bun run --bun nbb test/genmlx/vectorized_benchmark.cljs
```

No build step, no compilation. nbb interprets ClojureScript directly.

**Requirements:** macOS with Apple Silicon, Bun (or Node.js 18+), `npm install`
for `@mlx-node/core` and `@mlx-node/lm`, and nbb `1.4.208` (pinned via the `nbb`
script in `package.json`). Malli is a git submodule tracking **official upstream
`metosin/malli`** on the nbb classpath — the earlier robert-johansson/malli fork
existed only for nbb 1.4.206 compatibility, which 1.4.208 made unnecessary (its
SCI exposes `IPrintWithWriter`).

## Project structure

```
src/genmlx/
  # Layer 0: MLX + Runtime (the membrane — see mlx.cljs for section organization)
  mlx.cljs, mlx/random.cljs, runtime.cljs, dispatch.cljs

  # Layer 1: Core Data (pure immutable structures)
  choicemap.cljs, trace.cljs, selection.cljs, diff.cljs

  # Layer 2: GFI & Execution (11 protocols, 7+6 handler transitions)
  protocols.cljs, handler.cljs, edit.cljs, tensor_trace.cljs

  # Layer 3: DSL + Schema (gen macro, DynamicGF, 4-level dispatcher)
  gen.cljc, dynamic.cljs, schema.cljs, schemas.cljs, inspect.cljs

  # Layer 4: Distributions (36 constructors, open multimethods)
  dist/core.cljs, dist/macros.cljc, dist.cljs

  # Layer 5: Combinators (Map, Unfold, Switch, Scan, Mask, Mix, Recurse, etc.)
  combinators.cljs, vmap.cljs

  # Layer 6: Inference (35+ algorithms across 29 files)
  inference/ — importance, mcmc, smc, smcp3, vi, adev, amortized, kernel,
  util, diagnostics, analytical, conjugate, auto_analytical, kalman, ekf,
  ekf_nd, hmm_forward, enumerate, exact, fisher, compiled_gradient,
  compiled_optimizer, compiled_smc, differentiable, differentiable_resample,
  pmcmc, cost, steppable, translator

  # Layer 7: Compiled Paths (L1-L4 compilation pipeline)
  compiled.cljs, compiled_ops.cljs, compiled_gen.cljs, rewrite.cljs,
  affine.cljs, conjugacy.cljs, dep_graph.cljs, method_selection.cljs

  # Layer 8: Supporting Systems
  vectorized.cljs, gradients.cljs, learning.cljs, custom_gradient.cljs,
  nn.cljs, serialize.cljs, verify.cljs, gfi.cljs, fit.cljs, dev.cljs,
  schemas.cljs, inspect.cljs, sensorimotor.cljs, program.cljs

  # Layer 9: LLM Integration
  llm/ — backend.cljs, core.cljs, grammar.cljs, bytes.cljs, codegen.cljs,
         msa.cljs, vision.cljs

test/genmlx/  — one self-contained, executable test file per module
```

## Architecture layers

The implementation layers map onto the three-layer purity model:

```
── Layer C (GPU execution) ──────────────────────────────────────────────
  mlx-node Rust/C++   5 crates; 214 @genmlx/core function exports (coverage-matrix-pinned). MxArray = Arc<lazy graph node>.
                      eval! is the only operation that dispatches to Metal.

── Membrane (mlx.cljs) ─────────────────────────────────────────────────
  Layer 0: MLX + Runtime    (mlx.cljs, mlx/random.cljs, runtime.cljs, dispatch.cljs)
           mlx.cljs sections: Pure Graph Ops | Queries | Combinators | Effectful | Memory

── Layers A+B (pure ClojureScript + pure MLX graphs) ───────────────────
  Layer 1: Core Data        (choicemap, trace, selection, diff — pure)
  Layer 2: GFI & Execution  (protocols, handler, edit, tensor_trace — pure)
  Layer 3: DSL + Schema     (gen macro, dynamic, schema, schemas, inspect — pure)
  Layer 4: Distributions    (dist/core, dist/macros, dist — 36 constructors, pure)
  Layer 5: Combinators      (combinators, vmap — 10 combinators, pure)
  Layer 6: Inference         (29 files, 35+ algorithms — pure)
  Layer 7: Compiled Paths   (compiled, compiled_ops, rewrite, affine, conjugacy, dep_graph,
                             method_selection — pure)
  Layer 8: Supporting       (vectorized, gradients, learning, nn, serialize, gfi, verify,
                             fit, dev — pure except dev.cljs atoms)
  Layer 9: LLM Integration  (llm/backend, core, grammar, bytes, codegen, msa, vision —
                             pure except KV cache mutation in backend.cljs)
```

Strict dependency direction: higher layers depend on lower, never the reverse.
The only circular dependency risk is dynamic↔handler (resolved via the executor
pattern: runtime.cljs takes an `:executor` function in the state map, avoiding
direct import of dynamic.cljs).

## Key design principles

1. **Purely functional.** Layers 1-9 are referentially transparent. Mutation is
   confined to the membrane (Layer 0) plus a small audited set of caches and
   training state, verified by property tests (`mutation_boundary_test.cljs`).
   The mutable boundaries are:
   - The handler's `volatile!` in `runtime.cljs` (scoped to a single `run-handler`
     call — created fresh, consumed locally, never escapes)
   - Resource-management state in `mlx.cljs`: five atoms (`tidy-depth`,
     `grad-depth` nesting counters; `alloc-retry-count`, `proactive-sweep-count`
     telemetry; `buffer-count-threshold` hysteresis) and four `^:mutable`
     counters (`ops-since-check`, `allocs-since-count-check`, `proactive-armed?`,
     `gfi-ops-count`) — cleanup heuristics only, never affect computation results
   - Two atoms for dev mode extension (`dispatch-fn` in `dynamic.cljs`,
     `validate-fn` in `runtime.cljs` — only swapped by `dev.cljs` start!/stop!,
     no-ops in production)
   - Memoization caches of deterministic values: `fused-cache` atoms on
     Unfold/Scan combinator records (`combinators.cljs`), `with-cache` in
     `inference/exact.cljs`, and the construction-scoped expected-utility
     atoms in `agents/` built on it — write-once, invisible to results
   - Training state owned by the caller: `nn.cljs` layer refs and optimizer
     state atoms, the encoder atom in `inference/amortized.cljs`
   - The `defdist` registry bookkeeping atom in `dist/core.cljs` (never read
     by computation paths)
   - KV cache mutation in `llm/backend.cljs` (always in try/finally)
   - The live `Bun.serve` listener (an OS resource, not pure state) in the network
     face of the Bun world membrane (`world/net.cljs`): created by `serve!`, scoped
     and torn down by `with-server`'s `p/finally` (the blessed path) — analogous to
     the KV-cache try/finally. A bare `serve!` hands the lifecycle to the caller.
   - The native `GrpoTrainingEngine` handle in the TRAINING face of the world
     membrane (`world/train.cljs`, genmlx-zftr): an externally-mutating native engine
     that updates model weights + AdamW moments IN PLACE — the training `eval!`
     -equivalent, the sole side effect of that face. Fenced by the `with-trainer`
     blessed scope whose `p/handle` teardown disposes it on success OR throw
     (mirroring `llm/backend.cljs`'s KV-cache try/finally and `world/net.cljs`'s
     `with-server`); a bare `make-trainer!`/`dispose!` hands the lifecycle to the
     caller. Per-trainer mutable state lives in one `atom` (`:disposed?`), mirroring
     `CljsForwardModel`'s cache atom. The in-place weight updates are a *parallel*
     path that never composes back into the pure GFI-score gradient flow.
2. **Data-driven, open for extension.** Distributions are a single `Distribution`
   record with open multimethods. New distributions via `defdist`. New execution
   strategies via `dispatch/with-handler` or `dispatch/with-dispatch`. Grammar
   constraints compose via the same handler middleware as analytical inference.

3. **MLX arrays end-to-end.** Values stay as MLX arrays from sampling through
   scoring through gradient computation. Only extract to JS numbers with
   `mx/item` at inference boundaries.

4. **Lazy graph + explicit eval.** MLX operations build lazy computation graphs
   (Layer B values). `mx/eval!` is the sole side effect — it dispatches the
   graph to Metal for execution. Eval happens at three kinds of boundaries:
   - **API boundaries:** `mx/item`, `mx/->clj`, serialization
   - **Inference hot loops:** `mx/materialize!` to break graph accumulation
     (essential — without it, 1000 MCMC iterations build a 1000-node graph)
   - **Tidy scopes:** `mx/tidy-run`, `mx/tidy-materialize` for memory management

5. **Shape-based batching.** Vectorized inference works by changing array shapes
   (`[N]` instead of `[]`), not by transforming functions with `vmap`. MLX
   broadcasting handles all arithmetic naturally.

6. **Compose, don't duplicate.** Compiled paths compose on existing handlers and
   infrastructure — no parallel implementations. The handler is ground truth.

7. **The GFI algebraic laws.** The GFI algebraic theory (`gfi.cljs`) encodes
   the laws from the thesis (85 as of 2026-06; count the `laws` vector for the
   current number) covering all operations, compositionality, gradients, and
   compiled path equivalence. `strip-compiled` forces handler path for testing.

8. **Sync math, async events.** GenMLX core is synchronous: GFI ops, inference,
   distributions, combinators, local LLMs as GFs — all sync. Promesa appears
   only at genuine I/O boundaries (model loading, tokenizer encode/decode,
   `.chat`, streaming, external APIs). Runtime housekeeping (GC, Metal cleanup)
   stays internal via `mx/force-gc!` and never propagates to user API. The
   async event loop lives one layer up in the cognitive-architecture layer
   that embeds GenMLX. The sync/async choice follows semantic lines — math
   vs events — not runtime convenience.

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

;; GFI operations (11 protocols)
(p/simulate model args)                  ;; => Trace
(p/generate model args constraints)      ;; => {:trace Trace :weight scalar}
(p/update model trace new-constraints)   ;; => {:trace :weight :discard}
(p/update-with-args model trace new-args argdiffs constraints) ;; thesis x'
(p/regenerate model trace selection)     ;; => {:trace :weight}
(p/assess model args choices)            ;; => {:retval :weight}
(p/project model trace selection)        ;; => scalar
(p/propose model args)                   ;; => {:choices :weight :retval}
(edit/edit model trace edit-request)     ;; => {:trace :weight :discard :backward-request}

;; Vectorized (runs model body ONCE for N particles)
(dyn/vsimulate model args n key)         ;; => VectorizedTrace
(dyn/vgenerate model args obs n key)     ;; => VectorizedTrace with weights
(dyn/vupdate model vtrace constraints key) ;; => {:vtrace :weight :discard}
(dyn/vupdate-args model vtrace new-args constraints key) ;; batched thesis x'
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

1. **Pure transitions** in `handler.cljs` — 7 scalar state transition functions
   (the 6 GFI modes simulate/generate/assess/update/regenerate/project + a general
   retained-only `regenerate-transition-general`) + 6 batched variants.
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

**Regenerate has two transitions (genmlx-hmch, genmlx-yep2).** The *fast*
`regenerate-transition` (per-site convention) is used when the selection is
proven equivalent to the general path — no structure change (`has-branches?`
false) and the selected sites are mutually independent (no selected site feeds
another's distribution parameters). Otherwise `regenerate-transition-general`
builds the new trace (selected sites resample; unselected-&-absent sites sample
fresh — a structure change replacing the old throw, enabling branch flips;
unselected-&-present sites are retained), and `make-regen-result-general`
computes the retained-only weight `W = Σ_retained [lp(v; new ctx) − lp(v; old
ctx)]` with two project passes — `project(new-trace, retained) −
project(old-trace, retained)` — where `retained` = leaf addresses present in
both traces minus the selection. Selected, fresh, and removed sites cancel to
0; the project passes recurse through splices, so dependent joint moves and
spliced sub-models compose with no weight bookkeeping in the parent. The
compiled/prefix/branch-rewrite regen paths are skipped (deferred to the handler
general path) for non-fast-eligible selections; the analytical path is
unaffected.

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

9 distributions supported: gaussian, uniform, bernoulli, exponential, log-normal,
delta, laplace, cauchy, iid-gaussian (plus `:normal`/`:flip` aliases). The
expression compiler (`compile-expr`) resolves ~45 MLX operations from
namespace-qualified symbols in the source form.

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

**Limitations:** `splice` in shape-based batched mode (`vsimulate`/`vgenerate`)
is supported via three runtime paths: DynamicGF sub-gfs run a batched
sub-handler, combinators implementing `IBatchedSplice` take a fused fast path,
and other GFI values fall back to scalar-per-particle execution. No `mx/item`
in model bodies during batched execution (breaks vectorization).

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

~85 NAPI-exported functions in `mlx-node/crates/mlx-core/src/genmlx.rs`.
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
- `schema_test.cljs`: 227/227 (L1-M1)
- `compiled_simulate_test.cljs`: 82/82 (L1-M2)
- `partial_compile_test.cljs`: 92/92 (L1-M3)
- `combinator_compile_test.cljs`: 92/92 (L1-M5)
- `l4_certification_test.cljs`: 41/41 (L4)
- `gen_clj_compat_test.cljs`: 356/356
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
- Don't add no-op stubs or backward-compat shims that lie about what they do.
  If a function does nothing, remove it. If it returns a hardcoded value, document
  that honestly. The membrane must be transparent.
- MLX has no float64, int64, or bool dtypes. `mx/float64` silently aliases to
  `mx/float32`. Code expecting 64-bit precision will get 32-bit with no warning.
- `mx/compile-fn` is an identity pass-through (returns `f` unchanged). GenMLX's
  compilation uses noise transforms + the expression compiler, not MLX's
  graph-caching compile. See mlx.cljs docstring for details.
- Don't create scratch, experiment, or temp files inside this repo. It
  intentionally has no catch-all gitignored directories, so any stray file
  surfaces in `git status` by design — that visibility is the point (it's what
  kept a junk drawer from silently accumulating). Keep ephemeral work in an
  external scratch location; promote only deliberate keepers back into the tree.

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

## Task protocol

Project work is tracked with **beans** (a flat-file issue tracker; tasks live in
`.beans/`). A `SessionStart` hook runs `beans prime`, which injects the beans
usage guide — current types, statuses, and priorities — into context every
session, so you don't read a TODO file. The local guide `beans.md` documents the
workflow and the fixed type/status/priority boundary in depth.

Priorities map the old P-levels: `critical` ≈ P0 (fix now), `high` ≈ P1 (build
next); the `draft` status holds captured ideas and explorations (old P2/P3).

Every actionable bean should carry a **"done means"** checklist in its body.
When working on a task:
1. Read the full bean including context and "done means" (`beans show <id>`)
2. Implement using agent teams when the task is non-trivial
3. Review agent checks every "done means" criterion before presenting
4. Never declare a task complete until every checkbox is confirmed
5. Update bean status as you go (`beans update <id> -s in-progress` →
   `-s completed`), adding a `## Summary of Changes` section on completion

When the user captures a new idea, create a `draft` bean with 2-3 sentences of
context. Promote to `todo` (and add "done means" criteria) when ready to spec.

## Related documents

- `beans.md` — Task tracking guide (beans workflow, the fixed type/status/priority boundary)
- `ARCHITECTURE.md` — The GFI as external contract, the pure-handler mechanism, the compilation ladder, and the data-driven dispatch stack (as built)
- `README.md` — Quick start, examples, public API overview
