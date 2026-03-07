# Known Issues

## 1. Unfold combinator state must be plain JS values

**Severity:** Medium — causes crashes
**Location:** `src/genmlx/combinators.cljs` (unfold-combinator), `src/genmlx/mlx.cljs:410` (array?), `src/genmlx/inference/util.cljs` (collect-trace-arrays)
**Description:** When using `comb/unfold-combinator` with `smc-unfold`, the kernel return value (state) could crash if it contained MLX arrays inside Clojure maps/vectors. The `tidy-run` in `smc-unfold` would dispose arrays it didn't know about, and `array?` could behave unexpectedly on non-array values.
**Status:** Fixed with two changes:
1. `array?` now guards against collections (`not (coll? x)`) and wraps `.shape` access in try/catch
2. `collect-trace-arrays` now recursively walks the retval via `walk-value-arrays`, finding MLX arrays inside maps and vectors (e.g., `{:mu mlx-scalar :seg 3}`)
This ensures `tidy-run` preserves all arrays in nested state structures.

## 2. Metal resource exhaustion on sustained N-API workloads

**Severity:** High — kills process
**Location:** N-API weak callback deferral + Metal 499K allocation limit
**Description:** Sustained workloads that create and discard many MLX arrays eventually hit Metal's hard limit of 499,000 buffer allocations. This affects **both Node.js and Bun** — it is NOT runtime-specific. The root cause is that N-API Release/Release callbacks are deferred (run asynchronously after GC marks objects unreachable), so Metal buffers aren't freed immediately even when the JS wrapper is collected.
**Mitigation in node-mlx:** We added `napi_adjust_external_memory` reporting (1MB minimum per array) to signal GC about native memory pressure. This makes GC run earlier and extends the safe window, but cannot fully prevent the issue because finalizers remain async.
**Workaround:** Use GenMLX's explicit resource management:
- `mx/tidy-run` — tracks and disposes non-preserved arrays after each step
- `u/dispose-trace` — explicitly dispose trace arrays when done
- `mx/clear-cache!` — release MLX's internal buffer cache periodically
- Split long-running workloads into smaller batches

## 3. Metal buffer accumulation in MCMC loops

**Severity:** Medium — causes OOM/crashes on long runs
**Location:** All MCMC algorithms in `src/genmlx/inference/mcmc.cljs`
**Description:** Each MCMC iteration creates new MLX arrays for proposals, gradients, and log-probability evaluations. Over thousands of iterations, Metal buffer memory accumulates faster than JS GC can reclaim it.
**Status:** Fixed. All MCMC paths now have proper Metal resource management:
- `collect-samples` (kernel.cljs): wraps in `mx/with-resource-guard`, uses `tidy-step` per iteration, `clear-cache!` after each step
- `run-loop-compiled-mh`: wrapped in `mx/with-resource-guard`, burn-in loops call `clear-cache!`
- `run-loop-compiled-hmc`: wrapped in `mx/with-resource-guard`, burn-in loops call `clear-cache!`
- `run-loop-compiled-mala`: wrapped in `mx/with-resource-guard`, burn-in loops call `clear-cache!`
- `map-optimize`: wrapped in `mx/with-resource-guard`
- `dual-averaging-warmup`: uses `tidy-run` per step with `clear-cache!`

## 4. `dispose-trace` disposes shared observation arrays

**Severity:** High — causes crashes
**Location:** `src/genmlx/inference/util.cljs` (`dispose-trace`)
**Description:** `dispose-trace` walks all MLX arrays in a trace and disposes them. However, when a trace is created via `p/generate` with an observation choicemap, the observed values from the choicemap are incorporated into the trace's choices. Disposing these shared arrays causes "Error converting 'this' to mx.array" on subsequent `p/generate` calls that reuse the same observation choicemap.
**Status:** Fixed. `dispose-trace` now accepts an optional second argument `preserve`:
```clojure
;; Preserve observation arrays:
(u/dispose-trace trace observations)       ;; pass the observation choicemap
(u/dispose-trace traces observations)      ;; works on collections too
(u/dispose-trace trace (u/collect-choicemap-arrays observations))  ;; pass a JS Set
```
Arrays found in the `preserve` choicemap (or Set) are skipped during disposal. The original 1-arity call `(u/dispose-trace trace)` still works with the old behavior for backward compatibility.

## 5. HMC/NUTS don't support array-valued trace sites

**Severity:** Low — limited use case
**Location:** `src/genmlx/inference/mcmc.cljs` (HMC, NUTS, MALA, compiled-mh)
**Description:** HMC, NUTS, MALA, and compiled-mh used `(count addresses)` to determine the number of parameters, assuming each address holds a scalar.
**Status:** Fixed. All four functions now use `compute-param-layout` to compute total parameter dimensionality, and pass the layout to `make-score-fn` and `extract-params` for proper array flattening/unflattening. Array-valued trace sites (e.g., `[18]`-shaped parameter vectors) now work correctly.
**Note:** Vectorized MCMC paths (`vectorized-compiled-trajectory-mh`, `vectorized-hmc`, `vectorized-mala`) still assume scalar choices due to the batching approach.
