# Compute-membrane coverage matrix (`@mlx-node/core` → `mlx.cljs`)

*Audit: `genmlx-0vwn` / `genmlx-e3jg`. Snapshot grounded 2026-06-20.*

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

The export surface is enumerated from the **live `@genmlx/core` runtime module**
(`Object.keys(require("@genmlx/core")).filter(k => typeof core[k] === "function")`) —
this is what `membrane_coverage_test` partitions, and it is the authority. The package
ships **no `types` field**; its `index.d.ts` is a regenerated snapshot that can lag the
runtime, so any audit that reads a `.d.ts` can be wrong. Concretely, the live surface
has **224** function exports (2026-06-21 W-B / genmlx-hg7q added the `valueAndGrad` /
`computeGradients` module free fns — they moved from MxArray methods to module exports;
2026-07-07 genmlx-lgbx added `conv2d` / `scatterAdd` / `putAlongAxis` — the first two
FFI-shimmed but never exported, the last newly shimmed via `scatter_add_axis`;
the `index.d.ts` is a regenerated snapshot that can lag, so re-derive its count from the
live module after a rebuild; 2026-07-10 genmlx-q5uq added `gatherQmm` / `dequantize` —
both FFI-shimmed since the MoE/quantize work but never exported; they unlock the owned
qwen3_5_moe forward's packed-expert path and odd-bit dequantization; same day genmlx-w3og
added `qwen35VlmPreprocess` — native image preprocessing for the owned vision tower;
same day genmlx-ps8a added `gatedDeltaScan` — the chunk-parallel GDN recurrence
(mlx-core's `gated_delta_chunked_ops`, BT=64 WY form), the fused-prefill primitive for
the owned qwen3.5 forward's linear-attention layers). One runtime-only name is `Gradients` (a real class export,
correctly on the `:training-orchestration` omission allowlist). When the doc and the live
test disagree, the test is right.

## Summary

| | Count |
|---|---:|
| Function exports (`typeof === "function"`, incl. classes) | **224** |
| → Wrapped in the membrane | **176** |
| → Intentionally omitted | **48** |

`wrapped ⊎ omitted = 224` — the partition tiles the surface exactly (asserted by
`coverage-accounting-test`). Non-function exports (DType constants etc.) and
per-class *method* coverage (e.g. `MxArray.addmm` / `argpartition`) are out of
scope for this floor — see *Deferred* below.

## Wrapped (176)

Not enumerated here (a static list would drift). The membrane binds every export
not on the omission allowlist below — covering the full pure-math / reduction /
linalg / FFT-free array surface, RNG keys (`mlx/random.cljs`), autograd
(`value_and_grad` / `grad`), memory introspection + control, and the LLM
forward-pass primitives. As of `genmlx-zftr` (Phase 0) the **training membrane
face** `world/train.cljs` also wraps `GrpoTrainingEngine` and
`createRandomQwen35Checkpoint` behind its mutable-state quarantine (the GRPO
engine is NOT in the pure compute membrane). Recently completed (this audit): the trig/hyperbolic
family (`arcsin`/`arctan`/`sinh`/`cosh`), `log-softmax`, `logical-and/or/not`,
`isfinite`, `cumprod`, `roll`, `pad`, the memory introspection ops
(`get-memory-limit`/`get-wired-limit`/`memory-stats`/`get-memory-snapshot`,
`synchronize!`), and `gpu-architecture-gen` (the sole native source of the GPU
architecture generation).

To list the wrapped set, run the audit recipe and take the complement of the
omissions.

## Intentionally omitted (47)

Each export below is deliberately **not** in the pure compute membrane. The
category records where the capability belongs instead.

### Functions (21)

| Export | Category | Reason |
|---|---|---|
| `broadcastTo` | `:broken` | Native op mis-fills size-1 dims. `mx/broadcast-to` is a custom reconstruction; pinned by `broadcast-to-omission-test`. |
| `getProfilingData`, `isProfilingEnabled`, `setProfilingEnabled`, `resetProfilingData` | `:profiling-gated-i0s4` | Profiling instrumentation; wire-on-demand behind the `genmlx-i0s4` cost meter. |
| `buildRewardOutputs`, `createRandomQwen3Checkpoint`, `createRandomQwen35MoeCheckpoint` | `:training-orchestration` | Remaining native training surface — bind in `world/train.cljs` as Phase 1-4 tap them, behind the mutable-state quarantine (`genmlx-zftr`/`genmlx-706r`). The GRPO engine + `createRandomQwen35Checkpoint` are already **wrapped** there. |
| `convertForeignWeights`, `convertGgufToSafetensors`, `convertModel`, `convertParquetToJsonl` | `:model-conversion` | Offline weight/format conversion tooling — not graph ops. |
| `createPaddleocrVlConfig`, `createQianfanOcrConfig`, `documentToXlsx`, `formatDocument`, `saveToXlsx`, `parsePaddleResponse`, `parseToolCallsFromText`, `parseVlmOutput` | `:ocr-vlm-document` | OCR / vision-language / document pipelines — bind via `@mlx-node/lm` vision (`llm/vision.cljs`). |
| `compileFn` | `:compile-strategy-bypass` | Native MLX graph-caching compile, deliberately bypassed — GenMLX compilation uses noise transforms + the expression compiler (Level 1), not MLX's compile. |

### Classes (26)

A JS `class` is a function export, so it counts toward the runtime surface and
must be accounted for. None are wrapped as compute ops.

| Exports | Category | Reason |
|---|---|---|
| `SftTrainingEngine`, `NativeRewardRegistry`, `Gradients`, `OutputStore`, `ResponseStore` | `:training-orchestration` | Remaining training engines + result/registry types — bind in `world/train.cljs` as Phase 1-4 tap them. `GrpoTrainingEngine` is already **wrapped** there (`genmlx-zftr`). |
| `Gemma4Model`, `HarrierModel`, `Lfm2Model`, `Qwen3Model`, `Qwen35Model`, `Qwen35MoeModel`, `Qwen3Tokenizer`, `BatchGenerationResult`, `GenerationResult`, `ChatStreamHandle` | `:llm-orchestration` | Loaded-model / tokenizer / generation classes — bound via `@mlx-node/lm` ChatSession (`msa.cljs` / `backend.cljs`). The per-token GFI path reaches the low-level forward, not these. |
| `DocLayoutModel`, `DocOrientationModel`, `DocUnwarpModel`, `PrivacyFilterModel`, `QianfanOCRModel`, `TextDetModel`, `TextRecModel`, `VLModel`, `VlmChatResult`, `VlmProcessedImage` | `:ocr-vlm-document` | OCR / vision-language pipeline classes — `@mlx-node/lm` vision. |
| `Tensor` | `:foreign-tensor-type` | A foreign tensor class distinct from `MxArray` (the membrane's value type). |

## Deferred (separate beans, not part of this floor)

- **Native training engines** — `GrpoTrainingEngine` is now wrapped in the
  `world/train.cljs` training face behind the mutable-state quarantine
  (`genmlx-zftr` Phase 0). `SftTrainingEngine` + the reward/result types bind there
  incrementally as Phase 1-4 tap them; this feeds the GFI-reward GRPO flywheel
  (`genmlx-ugkv` / `genmlx-706r`).
- **Profiling data** — gate on the `genmlx-i0s4` cost meter.
- **`MxArray` array-method gaps** — `addmm` / `argpartition` and the rest of the
  ~150-method class surface (category-b appendix). (`putAlongAxis` graduated to a
  wrapped module export in genmlx-lgbx, 2026-07-07.)
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
