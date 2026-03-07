# Known Issues

## Metal resource exhaustion on sustained N-API workloads

**Severity:** High — kills process
**Location:** N-API weak callback deferral + Metal 499K allocation limit
**Description:** Sustained workloads that create and discard many MLX arrays eventually hit Metal's hard limit of 499,000 buffer allocations. This affects **both Node.js and Bun** — it is NOT runtime-specific. The root cause is that N-API Release/Release callbacks are deferred (run asynchronously after GC marks objects unreachable), so Metal buffers aren't freed immediately even when the JS wrapper is collected.
**Mitigation in node-mlx:** We added `napi_adjust_external_memory` reporting (1MB minimum per array) to signal GC about native memory pressure. This makes GC run earlier and extends the safe window, but cannot fully prevent the issue because finalizers remain async.
**Workaround:** Use GenMLX's explicit resource management:
- `mx/tidy-run` — tracks and disposes non-preserved arrays after each step
- `u/dispose-trace` — explicitly dispose trace arrays when done
- `mx/clear-cache!` — release MLX's internal buffer cache periodically
- Split long-running workloads into smaller batches
