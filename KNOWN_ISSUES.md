# Known Issues

## Metal resource exhaustion on sustained N-API workloads

**Severity:** Medium (was High — now mitigated at the native layer)
**Location:** N-API weak callback deferral + Metal 499K allocation limit
**Description:** Sustained workloads that create and discard many MLX arrays can hit Metal's hard limit of 499,000 buffer allocations. The root cause is that N-API finalizer callbacks are deferred via `SetImmediate`, so Metal buffers aren't freed until the event loop yields — which never happens in synchronous inference loops.

**Fix (node-mlx):**
1. **`napi_adjust_external_memory` on all paths** — GC pressure signaling now covers arrays returned from operations (AllowPassByValue path), not just constructor-created arrays. This was a critical gap in the original mitigation.
2. **`sweepDeadArrays()`** — New native function that synchronously walks the wrapper registry, finds arrays whose JS objects have been GC'd but whose deferred finalizers haven't run, and immediately frees the Metal buffers. This bypasses the deferred finalizer bottleneck entirely.
3. **Double-free guard** — Deferred finalizers check if the native object was already freed by sweep before attempting cleanup.

**GenMLX integration:**
- `mx/force-gc!` now automatically calls `sweep-dead-arrays!` after triggering GC
- `mx/sweep-dead-arrays!` available for explicit use in custom loops
- Existing `mx/tidy-run` and `u/dispose-trace` continue to work as before

**Remaining limitation:** Code that creates many arrays in a tight synchronous loop without ANY cleanup (no `tidy-run`, no `force-gc!`, no event loop yields) will still accumulate. The recommended pattern for inference loops is to call `(mx/force-gc!)` periodically (every N iterations) or use `(mx/tidy-run ...)` per step.
