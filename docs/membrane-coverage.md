# Compute-membrane coverage matrix (`@mlx-node/core` → `mlx.cljs`)

*Audit: `genmlx-0vwn` / `genmlx-e3jg`. Snapshot grounded 2026-06-15.*

The `@mlx-node/core` package is GenMLX's compute substrate (Layer C). The
**membrane** — `src/genmlx/mlx.cljs` + `src/genmlx/mlx/random.cljs` (Layer 0) —
binds it. This document is the coverage matrix: every export is either **wrapped**
in the membrane or **intentionally omitted** with a reason. There are no no-op
stubs; the membrane stays thin and honest.

> **This document is a snapshot. The authority is the test.**
> `test/genmlx/membrane_coverage_test.cljs` partitions the *live* export surface
> at runtime and fails if any export is neither wrapped nor on the omission
> allowlist (both directions — additions and deletions). When this doc and the
> test disagree, the test is right; regenerate this doc from the audit recipe
> below.

## Source of truth for the surface

The export surface is read from **`packages/core/index.d.cts`** (the `types`
target in `@mlx-node/core`'s `package.json`). The older `index.d.ts` (Apr 1) is
**stale** — it still lists `getParameters`/`applyGradients`, which no longer
exist; the training surface is now engine-class-only. Any audit that reads
`.d.ts` is wrong on day one.

## Summary

| | Count |
|---|---:|
| Function exports (`typeof === "function"`, incl. classes) | **212** |
| → Wrapped in the membrane | **164** |
| → Intentionally omitted | **48** |

`wrapped ⊎ omitted = 212` — the partition tiles the surface exactly (asserted by
`coverage-accounting-test`). Non-function exports (DType constants etc.) and
per-class *method* coverage (e.g. `MxArray.addmm` / `argpartition` /
`putAlongAxis`) are out of scope for this floor — see *Deferred* below.

## Wrapped (162)

Not enumerated here (a static list would drift). The membrane binds every export
not on the omission allowlist below — covering the full pure-math / reduction /
linalg / FFT-free array surface, RNG keys (`mlx/random.cljs`), autograd
(`value_and_grad` / `grad`), memory introspection + control, and the LLM
forward-pass primitives. Recently completed (this audit): the trig/hyperbolic
family (`arcsin`/`arctan`/`sinh`/`cosh`), `log-softmax`, `logical-and/or/not`,
`isfinite`, `cumprod`, `roll`, `pad`, the memory introspection ops
(`get-memory-limit`/`get-wired-limit`/`memory-stats`/`get-memory-snapshot`,
`synchronize!`), and `gpu-architecture-gen` (the sole native source of the GPU
architecture generation).

To list the wrapped set, run the audit recipe and take the complement of the
omissions.

## Intentionally omitted (50)

Each export below is deliberately **not** in the pure compute membrane. The
category records where the capability belongs instead.

### Functions (21)

| Export | Category | Reason |
|---|---|---|
| `broadcastTo` | `:broken` | Native op mis-fills size-1 dims. `mx/broadcast-to` is a custom reconstruction; pinned by `broadcast-to-omission-test`. |
| `getProfilingData`, `isProfilingEnabled`, `setProfilingEnabled`, `resetProfilingData` | `:profiling-gated-i0s4` | Profiling instrumentation; wire-on-demand behind the `genmlx-i0s4` cost meter. |
| `buildRewardOutputs`, `createRandomQwen3Checkpoint`, `createRandomQwen35Checkpoint`, `createRandomQwen35MoeCheckpoint` | `:training-orchestration` | GRPO/SFT training surface — bind at engine level in a future `@mlx-node/trl` Layer-6 ns with the mutable-state quarantine, never the pure membrane (`genmlx-706r`). |
| `convertForeignWeights`, `convertGgufToSafetensors`, `convertModel`, `convertParquetToJsonl` | `:model-conversion` | Offline weight/format conversion tooling — not graph ops. |
| `createPaddleocrVlConfig`, `createQianfanOcrConfig`, `documentToXlsx`, `formatDocument`, `saveToXlsx`, `parsePaddleResponse`, `parseToolCallsFromText`, `parseVlmOutput` | `:ocr-vlm-document` | OCR / vision-language / document pipelines — bind via `@mlx-node/lm` vision (`llm/vision.cljs`). |

### Classes (27)

A JS `class` is a function export, so it counts toward the runtime surface and
must be accounted for. None are wrapped as compute ops.

| Exports | Category | Reason |
|---|---|---|
| `GrpoTrainingEngine`, `SftTrainingEngine`, `NativeRewardRegistry`, `Gradients`, `OutputStore`, `ResponseStore` | `:training-orchestration` | Training engines + result/registry types — engine-level, mutable-state quarantine (future `@mlx-node/trl` Layer-6). |
| `Gemma4Model`, `HarrierModel`, `Lfm2Model`, `Qwen3Model`, `Qwen35Model`, `Qwen35MoeModel`, `Qwen3Tokenizer`, `BatchGenerationResult`, `GenerationResult`, `ChatStreamHandle` | `:llm-orchestration` | Loaded-model / tokenizer / generation classes — bound via `@mlx-node/lm` ChatSession (`msa.cljs` / `backend.cljs`). The per-token GFI path reaches the low-level forward, not these. |
| `DocLayoutModel`, `DocOrientationModel`, `DocUnwarpModel`, `PrivacyFilterModel`, `QianfanOCRModel`, `TextDetModel`, `TextRecModel`, `VLModel`, `VlmChatResult`, `VlmProcessedImage` | `:ocr-vlm-document` | OCR / vision-language pipeline classes — `@mlx-node/lm` vision. |
| `Tensor` | `:foreign-tensor-type` | A foreign tensor class distinct from `MxArray` (the membrane's value type). |

## Deferred (separate beans, not part of this floor)

- **`@mlx-node/trl` training engines** — bind `GrpoTrainingEngine` / `SftTrainingEngine`
  at engine level in a Layer-6 ns with the mutable-state quarantine. This is the
  real GRPO/SFT work and feeds the self-training-coder flywheel (`genmlx-706r`).
- **Profiling data** — gate on the `genmlx-i0s4` cost meter.
- **`MxArray` array-method gaps** — `addmm` / `argpartition` / `putAlongAxis` and
  the rest of the ~150-method class surface (category-b appendix).
- **Non-function exports** — DType constants and class instances (a separate spike).

## Re-audit recipe

The drift guard runs every time the membrane suite runs. To regenerate this
matrix by hand after an `@mlx-node/core` bump:

1. Enumerate the surface: `Object.keys(require("@mlx-node/core")).filter(k => typeof core[k] === "function")`.
2. Cross-reference against `(.-name c)` / `(.name c …)` references — bound off the
   `@mlx-node/core` object `c` — in `mlx.cljs` + `mlx/random.cljs`. The test's
   `referenced?` scopes to the `c` receiver so a same-named method on another
   object (`.vmap` on `M`, `.gcAndSweep` on `jsc`) is not counted as a wrap.
3. For each new export: wire it (if pure-math/introspection capability) or add an
   omission entry with a category + reason. For each deleted export: remove its
   omission entry.
4. Update the counts above and run `membrane_coverage_test.cljs`.

The test fails with an actionable message naming exactly which exports are
unaccounted, stale, or misclassified.
