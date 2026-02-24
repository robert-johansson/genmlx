# Bun + MLX + ClojureScript: Compatibility & Deep Synergy Analysis

## Status: Bun is fully compatible with GenMLX

Branch `experiment/bun-compatibility` demonstrates full compatibility with a 2-line fix.

**Root cause of initial failures:** Bun's N-API layer doesn't correctly marshal `Slice`
class instances as `Index` type arguments to native C++ addons. This affected
`array.index(new Slice(...))` calls in `@frost-beta/mlx`.

**Fix:** Replaced 2 uses of `Slice`+`.index` with equivalent `take`-based alternatives
in `mlx/random.cljs` and `mlx.cljs`. Both alternatives produce bitwise-identical results
on Node and Bun, including correct gradient flow through `take`+`arange`.

### Test Results (Bun)

| Test Suite | Result |
|---|---|
| Core tests (8 suites) | All pass |
| Gen.clj compat | 165/165 |
| GenJAX compat | 73/73 |
| Vectorized tests | All pass |

### Performance (wall-clock time)

| Test | Node | Bun | Speedup |
|---|---|---|---|
| Startup (`-e :done`) | 0.47s | 0.13s | 3.6x |
| gen_clj_compat (165) | 1.20s | 0.54s | 2.2x |
| dist_test | 1.04s | 0.60s | 1.7x |
| inference_test | 1.57s | 1.26s | 1.2x |
| vectorized_test | 1.15s | 0.62s | 1.9x |

Bun is consistently faster, with the biggest gains on startup (3.6x) and moderate
gains on compute-heavy inference (1.2-1.9x).

---

## Three Philosophies, One Pattern

The deep match across all three systems is **deferral and sharing** — avoiding work
and avoiding copies at every level:

| Layer | Philosophy |
|---|---|
| **Apple Silicon** | "Don't move data" — unified memory, no CPU/GPU copies |
| **MLX** | "Don't compute until you must" — lazy computation graph, `eval()` materializes |
| **ClojureScript** | "Don't copy until you must" — persistent data structures, structural sharing |
| **bun:ffi** | "Don't wrap until you must" — pointers are bare JS numbers, zero-copy TypedArrays |

The entire stack from GPU metal through C API through JS runtime to ClojureScript
semantics is about the same thing: share what you can, defer what you can, materialize
only at boundaries.

## The `mlx-c` + `bun:ffi` Opportunity

Apple maintains an official C API for MLX called
[mlx-c](https://github.com/ml-explore/mlx-c) (`brew install mlx-c`), specifically
designed as "a bridge for binding other languages to MLX." It has 28 header files
covering the full API surface.

`@frost-beta/mlx` wraps the **C++ API** via N-API. The call path today:

```
ClojureScript -> nbb interop -> N-API boundary (~11ns) -> kizunapi C++ templates
  -> mlx::core::add() -> lazy graph node
```

A hypothetical `bun-mlx` via `bun:ffi` + `mlx-c` would be:

```
ClojureScript -> nbb interop -> bun:ffi JIT wrapper (~6ns)
  -> mlx_add() -> lazy graph node
```

The `mlx_array` type is `{ void* ctx }` — a single pointer. In `bun:ffi`, pointers
are represented as plain JS `number`s. An MLX array handle becomes just a number.
No object allocation, no GC pressure for the handle itself.

## The Isomorphisms

Three genuinely beautiful structural matches:

### 1. Lazy Graph ~ Lazy Sequences

MLX's computation graph and ClojureScript's lazy sequences are isomorphic:

```clojure
;; ClojureScript: build a lazy chain, realize when needed
(->> (range 1000) (map inc) (filter odd?) (take 10) (doall))

;; MLX: build a lazy graph, eval when needed
(-> x (mx/add 1) (mx/multiply 2) (mx/sum) (mx/eval!))
```

Both are descriptions of computation that haven't happened yet. `doall` is analogous
to `mx/eval!`. GenMLX already exploits this — model bodies build graphs, inference
boundaries call `eval!`.

### 2. Persistent Data ~ Unified Memory

ClojureScript's persistent data structures achieve "apparent mutation" through
structural sharing — old and new versions share most of their tree. Apple Silicon's
unified memory achieves "apparent ubiquity" through physical sharing — CPU and GPU
see the same bytes. Both eliminate copies by sharing.

With `bun:ffi`'s `toArrayBuffer()`, you get zero-copy from MLX array data to JS
TypedArray — and on Apple Silicon, that's the same physical memory the GPU wrote to.
No copies from GPU compute through C API through FFI through ClojureScript.

### 3. Transformations All the Way Down

```
MLX:           grad(f), vmap(f), compile(f)  — transform computation graphs
ClojureScript: (comp xf1 xf2), macros        — transform data flows / code
bun:ffi cc():  inline C compilation           — transform source to native at runtime
```

All three systems treat programs as data to be transformed.

## The Killer Feature: `bun:ffi` + `cc()` for Hot Paths

Bun embeds TinyCC and can compile C source code inline at runtime in ~5ms:

```javascript
import { cc } from "bun:ffi";

const { symbols } = cc({
  source: "./inference_kernel.c",
  library: ["mlxc"],
  symbols: {
    run_smc_step: { args: ["ptr", "ptr", "i32", "ptr"], returns: "void" }
  }
});
```

Model definition stays in idiomatic ClojureScript. The SMC/HMC inner loop is compiled
C calling `mlx-c` directly. No JS overhead, no N-API overhead, no interpreter overhead
for the hot path. This mirrors JAX's architecture: high-level Python for model
definition, XLA-compiled kernels for execution.

## Honest Assessment

| Aspect | Beautiful? | Practical? |
|---|---|---|
| Lazy graph ~ lazy sequences | Yes | Already exploited in GenMLX |
| Unified memory + zero-copy FFI | Yes | Real but marginal gain over N-API |
| `mlx_array` as bare `number` | Yes | Loses print/inspect without wrapping |
| `cc()` for hot inference loops | Yes | Genuinely transformative potential |
| 5ns vs 11ns per FFI call | Elegant | Negligible — graph ops take microseconds |
| JSC megamorphic IC for protocols | Yes | Real but modest (~6% on microbenchmarks) |

The FFI call overhead difference (5ns) is noise. The real bottleneck is nbb's SCI
interpreter, not the N-API boundary. A `bun-mlx` binding would gain very little over
`@frost-beta/mlx` for most operations.

**However**, the `cc()` inline C compilation is genuinely interesting — it could let
you write inference hot paths as compiled C calling `mlx-c`, while keeping model
definitions in idiomatic ClojureScript. That's not a marginal optimization; it's a
different architecture. It maps onto GenMLX's existing layering: Layers 0-5 stay in
ClojureScript, Layer 6 inference kernels get compiled C fast paths.

The question is whether it's worth the complexity. MLX's lazy graph already moves the
real compute to GPU Metal kernels — the JS/ClojureScript layer mostly builds the graph,
not executing it. The `cc()` approach would matter most for inference algorithms with
many small graph operations per step (like MH with per-step accept/reject logic).

## JSC vs V8 for ClojureScript

Bun uses JavaScriptCore (JSC) instead of V8. Some relevant differences:

- **Megamorphic inline cache**: JSC uses a multi-level per-thread lookup cache with
  >80% hit rate for polymorphic sites. V8 degrades after >4 hidden classes. This
  benefits ClojureScript's protocol dispatch patterns.
- **Four-tier JIT**: JSC has LLInt -> Baseline -> DFG -> FTL (using B3 backend).
  More gradual warmup than V8's two-tier Ignition -> TurboFan.
- **Memory model**: JSC's "butterfly" layout with up to 6 inline property slots per
  object is efficient for ClojureScript's many small HAMT trie nodes.
- **Benchmarks** (Mike Fikes, Planck): JSC is ~6% faster for map key access, up to
  3.3x faster for some map operations, 2.6x faster for higher-order functions with
  `:static-fns`.

These advantages are real but modest for GenMLX, since most compute time is in MLX
GPU kernels, not JS object manipulation.

## Known Bun Limitation

Bun's N-API layer does not correctly pass `Slice` class instances from native addons
as `Index` type arguments. This is the only incompatibility found with `@frost-beta/mlx`.
The fix (using `take` with int32 indices instead) is on branch
`experiment/bun-compatibility` and produces bitwise-identical results on both runtimes.

---

## Prior Art: Gen.clj `gen-mlx` Branch

The [`gen-mlx` branch of Gen.clj](https://github.com/robert-johansson/Gen.clj/tree/gen-mlx)
took the "bare metal" approach: JVM Clojure calling `mlx-c` (Apple's official C API)
directly via Project Panama, with a custom C shim for hot-path fusion.

### Architecture

```
Clojure (gen macro, GFI protocols, distributions)
  ↓
gen.mlx.array — MLXArray deftype with java.lang.ref.Cleaner, IDeref/Seqable/IFn
  ↓
gen.mlx.ffi — coffi/Panama bindings to mlx-c (~1150 lines)
  ↓
gen_mlx_shim.c — C fusion layer, 4 functions (~280 lines)
  ↓
mlx-c (Apple's official C API) → MLX C++ core → Metal GPU kernels
```

Total code: ~3200 lines for FFI + shim + array + transforms + dynamic + distributions,
compared to GenMLX's ~800 lines for `mlx.cljs` + `mlx/random.cljs`.

### What was clever

**1. C shim fuses entire HMC trajectories into one native call.**
`gen_mlx_leapfrog()` takes a value-and-grad closure, position, momentum, eps, and L —
runs the complete leapfrog loop C-to-C with zero JVM crossings. L+1 gradient evaluations
happen entirely in native code. This is the single biggest optimization GenMLX lacks.

**2. Branchless MH accept/reject via `arr/where`.**
The entire accept/reject is `(arr/where accepted proposal current)` — pure graph, no
eval, no branching. The full HMC step (leapfrog + MH) is a single lazy MLX graph,
evaluated once.

**3. Vectorized score function batching.**
`build-compiled-score-fn` replays the model once to discover its distribution structure,
groups choices by distribution type (`:normal`, `:exponential`, etc.), then computes all
logpdfs for each type in one vectorized call. Reduces ~6N FFI calls to ~6D (D = number
of distinct distribution types).

**4. MLXArray as a first-class Clojure citizen.**
`@arr` forces evaluation (like `deref` on a delay), `(seq arr)` gives elements,
`(arr :shape)` returns metadata, `(arr 0)` indexes. `toString` never forces eval —
shows shape/dtype without materializing the computation graph.

**5. Data-dependency detection.**
`params-depend-on-choices?` replays the model twice with perturbed choices and checks
if distribution parameters changed. Determines whether the fast cached path or the
slower per-call replay path is needed.

**6. Full GFI integration.**
`MLXDynamicDSLFunction` implements `IGenerativeFunction`/`IGenerate`. `MLXTrace`
implements `ITrace` + `IChoiceGradients`. The same `gf/simulate`, `gf/generate` calls
work with both kixi and MLX backends. Bridge functions connect traces to HMC/NUTS/MAP.

### What was painful

- **~35us per scalar FFI call** vs ~30ns for pure JVM — 1000x overhead from
  Panama + mlx-c vector-array ceremony. Forces NUTS tree-building to use JVM doubles.
- **Panama upcall stubs** for callbacks — wrapping Clojure fns as `mlx_closure` requires
  manual `MethodHandle` adaptation with ABI workarounds for aarch64. ~50 lines of
  intricate code per closure type.
- **No public `mlx-c` vmap API** — only internal `mlx_detail_vmap_*` functions,
  blocking true vectorized particle inference.
- **`mlx_vector_array` ceremony** — every mlx-c operation requires constructing/
  deconstructing vector containers for arguments and results. The C shim exists
  solely to fuse this overhead away.
- **Setup complexity** — build mlx-c from source, compile C shim, ensure correct
  dylib paths, require JDK 22+. vs GenMLX's `npm install`.

### Comparison with GenMLX

| | Gen.clj gen-mlx | GenMLX |
|---|---|---|
| **FFI path** | Clojure -> Panama -> mlx-c (C API) -> MLX C++ | CLJS -> N-API -> MLX C++ |
| **Per-call overhead** | ~35us (Panama + mlx-c vector ceremony) | ~11ns (N-API direct) |
| **Hot-path fusion** | C shim: entire leapfrog in 1 FFI call | None — each op is separate |
| **Memory management** | `Cleaner` + `AtomicBoolean` double-free guard | N-API refs + JS GC |
| **Batched inference** | Parallel chains via `(N,D)` arrays + vmap | Shape-based `vsimulate`/`vgenerate`, 61-122x |
| **GFI integration** | Full (`MLXDynamicDSLFunction`, `MLXTrace`) | Full (handler system, DynamicGF) |
| **Setup** | Build mlx-c, compile C shim, JDK 22+ | `npm install` |
| **Code volume** | ~3200 lines | ~800 lines |

### Key lesson for GenMLX

Calling `mlx-c` directly is viable but painful — 4x more code than the N-API approach,
with ABI fragility and higher per-call overhead. The N-API path is genuinely simpler
and faster for individual calls.

But the **C shim fusion pattern for leapfrog is a real win** that doesn't depend on
the FFI choice. GenMLX could achieve the same benefit by:

1. Writing a small N-API addon that fuses the HMC inner loop — calling
   `@frost-beta/mlx`'s bundled MLX C++ directly.
2. Or using Bun's `cc()` to compile a C function at startup that calls `mlx-c`
   for the leapfrog loop.
3. Or using `mx.compile` on the gradient function (as the prob-cljs branch did)
   to cache the Metal program, combined with `mx.tidy` per leapfrog step.

Option 3 is the lowest-friction path and doesn't require any native code. The
prob-cljs branch demonstrated that `compile-fn` + `tidy` + pre-allocated constants
gives most of the benefit of the C shim approach.
