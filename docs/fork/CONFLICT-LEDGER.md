# Conflict ledger — per-file resolution doctrine

> **Bean:** `genmlx-rjfm`. **Date:** 2026-07-25.
> **Scope of the measured section:** merging mlx-node `v0.0.8` (`0ebeaa57`) into our
> `08921022`. Merge-base `5602f12d`. Reproduce with
> `git -C mlx-node merge-tree --write-tree HEAD up/main` → tree `508b68da`.
> **Every line number below was read out of that merged tree**, not estimated.
> Re-derive them after any rebase or further commits — they will shift.

## How to use this file

1. Work the **silent hazards** in §1 *before* spending a build. They have no conflict marker.
2. Resolve the marked conflicts in §2 in the order given (cheap → expensive); the TS files only
   typecheck once `run-agent.ts` is decided.
3. Append a dated entry to §4 at the end of every sync.

**Rule for the whole file:** the merge commit contains **pure resolution and nothing else**.
Repairs, migrations and platform gates land as separate follow-up commits on `sync/up-vNEXT`.

---

## 1. Silent hazards — no conflict marker, roll call requires human sign-off

This is the dangerous half of the merge. Git auto-merged all of it. Two of the five classes
produce no compiler error either.

### 1a. Three type errors Rust *will* catch

Upstream's `vlm_prepare_vision_features` signature auto-merged to
`per_image_hashes: &[u64]` (merged `qwen3_5/model.rs:11917`), but three of our flat-vision call
sites still pass the scalar `image_cache_key: u64`. Verified in the merged tree — four other
call sites already pass `&per_image_hashes` and are fine:

| call site (merged tree) | offending arg | ours from |
|---|---|---|
| `qwen3_5/model.rs:4495` | `image_cache_key` at `:4497` | genmlx-9v44 dense flat vision |
| `qwen3_5_moe/model.rs:1182` | `image_cache_key` at `:1184` | genmlx-52mh MoE flat prefill |
| `qwen3_5_moe/model.rs:3436` | `image_cache_key` at `:3438` | genmlx-52mh continuation path |

**Resolution:** migrate all three to `engine::compute_image_cache_keys` (upstream, defined at
`crates/mlx-core/src/engine/cache.rs:89`, returns `(u64, Vec<u64>)`). Do **not** try to restore
the scalar signature — see §2's entry for `qwen3_5/model.rs`.

### 1b. Two markerless semantic auto-merges Rust will *not* catch

**`crates/mlx-core/src/models/qwen3_5/attention.rs` — the single highest-risk file in the merge.**
Upstream changed it +1190/−47; we changed it +62/−44 (CUDA flat attention + M-RoPE rope-delta,
genmlx-9v44/52mh). Git interleaved 1190 upstream lines into our 62 with **zero markers**. A
wrong-but-typechecking interleave shows up as garbled or repetitive generation, or as
nondeterministic garbage — the failure class that per `docs/` debug-method history is found only
by diffing against the Python oracle or by bisect. Budget a full read of the diff **plus**
`scripts/llm_forward_xval_mlxlm.py`.

**`crates/mlx-core/src/models/qwen3_5/quantized_linear.rs`.** Ours +267/−0 (genmlx-n32r frozen
packed experts, genmlx-x76x dequantize-for-training) into theirs +183/−5 (fp8_e4m3 / MXFP /
NVFP4 rework). GRPO on the quantized 0.8B and the 4-bit 35B both run through here. A bad
interleave is a silently-skipped or NaN training step — the genmlx-li1p class, which hid for
weeks. Re-verify the dequantize path explicitly; a clean compile proves nothing.

### 1c. Small-ours-into-huge-theirs — the hook may land in the wrong place

`qwen3_5/persistence.rs` (ours +12/−3 into theirs +640/−154), `qwen3_5_moe/persistence.rs`
(+16/−5 into +128/−27), `crates/mlx-sys/src/lib.rs` (ours +266 genmlx FFI decls into theirs
+110). These merge cleanly *by construction*, but the genmlx-x76x dequantize-at-InitTraining
hook can easily end up at the wrong point in the new load order.

### 1d. Generated file — our edits vanish with no marker

`packages/core/index.d.cts` carries **119 hand-added lines** of ours. It is a *generated* file:
upstream regenerates it wholesale in its `preversion` script (`oxnode ./build.ts && git add .`).
Our edits will be deleted silently at some future release.

**Resolution: delete them pre-merge.** Verified safe — the only TS declarations for
`branchCache` / `forwardBranch` / `vlmPrefillFlat` / `initCaches` that anything consumes live in
our own generated `packages/genmlx-core/index.d.ts`. Nothing in `packages/**` reads the
`index.d.cts` copies.

### 1e. Roll call

Before requesting a build, print and sign off on each:

```bash
T=$(git -C $MN merge-tree --write-tree HEAD up/main | head -1)
for f in crates/mlx-core/src/models/qwen3_5/attention.rs \
         crates/mlx-core/src/models/qwen3_5/quantized_linear.rs \
         crates/mlx-core/src/models/qwen3_5/persistence.rs \
         crates/mlx-core/src/models/qwen3_5_moe/persistence.rs \
         crates/mlx-sys/src/lib.rs; do
  echo "=== $f ==="; git -C $MN diff --stat HEAD..$T -- "$f"
done
```

---

## 2. Marked conflicts — 14 paths, 26 hunks

Ordered cheap → expensive.

### Trivial

**`crates/mlx-sys/mlx`** *(gitlink)* — a non-event. Our 15 MLX commits are CUDA/CPU/CMake;
mlx-node's 3 are Metal NAX. Disjoint: `merge-tree` inside the submodule exits 0 with zero
conflicts. Git simply refuses to auto-merge `160000` entries.
→ `git update-index --cacheinfo "160000,$MLXSHA,crates/mlx-sys/mlx"`. **Never** `checkout --ours/--theirs`.

**`crates/mlx-core/src/models/qwen3_5_moe/quantized_linear.rs`** — 1 hunk, 10 lines. Pure doc
comment on `QuantizedSwitchLinear`: ours describes the genmlx-n32r frozen-experts snapshot,
theirs the fp8_e4m3 dequantize-at-load path. → **Union both paragraphs.**

**`packages/cli/src/commands/agent/index.ts`** — 1 hunk, 5 lines. `runAgent({...})` option bag:
ours adds `genmlxModels`, theirs adds `traceLogFile`. → **Union.** Compiles only after
`run-agent.ts` is resolved.

### Mechanical

**`crates/mlx-core/src/models/qwen3_5/gated_delta_net.rs`** — 2 hunks, 34 lines. One doc comment
(union). One ours-vs-nothing: our `dequantize_to_standard()` (genmlx-x76x) against an empty
upstream side. → **Keep ours verbatim** — it is load-bearing for GRPO on quantized qwen3.5
checkpoints.

**`crates/mlx-sys/src/mlx_nn_ops.cpp`** + **`crates/mlx-core/src/array/data.rs`** — resolve as a
pair; they are the eval error channel. Ours calls `mlx_report_error` (thread-local slot read by
`mlx_take_last_error`, genmlx-uhtp/kfli); theirs calls `mlx_copy_error` into a bounded buffer
plus `mlx_trace_native_error`. Both symbols exist post-merge.
→ **Union in the C++** (do both calls, so the bounded buffer *and* the thread-local channel
work); → **take theirs in `data.rs`** (`eval_native` + structured `tracing::error!` that actually
uses the `context` param, which ours ignored). Strictly better, and it preserves the
NVRTC-compile-error detail surfacing genmlx-uhtp added.

**`packages/agent/__test__/run-agent.test.ts`** — 2 hunks, 51 lines. Encodes the seam change.
→ **Take theirs**, then re-add our genmlx-provider registration assertions. Nearly free once
`run-agent.ts` is decided.

### Semantic

**`packages/agent/__test__/convert-messages.test.ts`** — 2 hunks, 189 lines. The two competing
VLM-image designs written as assertions. You cannot keep both suites.
→ Follows whatever `convert-messages.ts` decides; port the surviving assertions.

**`crates/mlx-core/src/models/qwen3_5_moe/model.rs`** — the *marked* conflict is a trivial
import-list union (1 hunk, 10 lines: ours adds `vlm_prepare_vision_continuation`, theirs adds
`IMAGE_TOKEN_ID`, `Qwen3_5ContextLimits`, `constrain_paged_context_params`,
`qwen35_expanded_prompt_token_count`; all verified present in the merged tree).
→ **Union.** But the same file carries two of the three §1a type errors. Rewire them and
re-verify the M-RoPE rope-delta continuation — the genmlx-52mh failure mode is single-token
repetition.

### Severe

**`crates/mlx-core/src/models/qwen3_5/model.rs`** — the worst file. 1 hunk but **421 conflict
lines** at `:11925–12345` inside `vlm_prepare_vision_features`. Our side is 8 lines: a call to
`vision_features_cached(image_cache_key, …)`, our genmlx-9v44 helper, absent from the merge
base. Their side is a 411-line rewrite: rank-4 pixel validation, `clear_cache()` +
`probe_vision_memory()` headroom accounting, `plan_vision_image_requests` /
`lookup_vision_feature_cache` per-image keying, budgeted eviction with `protected_keys`, batched
miss handling, `concatenate_many` reassembly.

Three compounding problems, all verified in the merged tree:
1. the signature **already** auto-merged to upstream's `per_image_hashes: &[u64]` (`:11917`), so
   "take ours" cannot compile;
2. the §1a type error at `:4497`;
3. our `vision_features_cached` survives as a **live orphan** — still defined at `:12385`, still
   called at `:11927` and at `:12494` from `vlm_prepare_vision_continuation` (genmlx-lds5
   image-tolerant KV prefix reuse).

→ **Adopt theirs; migrate all three flat-vision call sites to `engine::compute_image_cache_keys`;
port the continuation path onto the new per-image cache.** This is a migration, not an
amputation — the capability is kept, ~50 lines of ours are deleted, the type errors die by
construction, and the worst file's future conflict surface shrinks permanently.

Do **not** keep both cache layers: that double-holds vision features in device memory on a box
with a documented global-OOM reboot bug (genmlx-h3p5).

**`packages/agent/src/run-agent.ts`** — 2 hunks, 65 lines. Upstream `#97` rewrote the function:
the test seam moved from `opts.mainImpl: RunAgentMain` to `opts.piImpl: RunAgentPi` (carrying
`main` + `ModelRegistry`); `MlxModelHost` construction moved *into* run-agent and is injected;
`PagedConfigOverrideManager` gained a try/finally lifecycle; three new extensions were added.

→ **Take theirs**, re-register `createGenmlxProviderExtension`, and **widen the registry
allowlist**. Verified at `up/main:packages/agent/src/run-agent.ts:110`:

```ts
const restoreModelRegistry = installMlxOnlyModelRegistryFilter(
  pi.ModelRegistry,
  opts.models.map((model) => model.discovered.name),
);
```

genmlx models are not in `opts.models`, so they are filtered out of Tab, `/models`, RPC
enumeration and session restore. **The failure is silent — models just vanish, no error.**

**`packages/agent/src/provider/model-host.ts`** — 3 hunks, 39 lines. Upstream reverted
`MlxModelHost` to a static `loadModel` + `new ChatSession(...)`. Ours deliberately routes through
`await loadNativeHost()` so the agent's import graph contains **no static native chain** — the
genmlx-djw6 native-owner latch, enforced by `__test__/native-import-graph.test.ts`. Taking
theirs silently defeats that design; its absence means registering both providers dlopens a
second MLX runtime.

**The killer:** `requirePagedCache: true` is set at `up/main:run-agent.ts:102` and enforced at
`model-host.ts:116`:

```ts
if (this.requirePagedCache && sessionModel.hasBlockPagedCache?.() !== true && !gemmaDraftActive) {
```

`has_block_paged_cache()` returns `self.paged_active`, and `crates/mlx-sys/src/mlx_paged_stubs_linux.cpp:11-13`
states outright that the Rust loaders gate paged attention on `mlx_metal_is_available()` —
**false on CUDA**. Note the optional chaining: an *absent* method also throws. A straight
"take theirs" makes **every `mlx agent` model load throw on Thor.**

→ Take theirs, re-apply the lazy `loadNativeHost()` latch, and gate `requirePagedCache` on Metal
availability. **Record this as a permanent divergence** — every future sync re-litigates it.

**`packages/agent/src/provider/stream-adapter.ts`** — 2 hunks, 36 lines; small marker footprint,
large blast radius. Upstream hard-coupled `makeMlxStreamSimple` to the concrete native
`ChatSession`: `session.supportsImages()` and `session.contextLimits()`. Meanwhile
`StreamSimpleHost.runWithResident` still hands out our duck-typed `StreamableSession`, which has
neither — **the merged file does not typecheck**. Upstream also dropped the third argument from
`startFromHistoryStream(config, signal)`.

Two of our features live inside the conflicted region and die under "take theirs": the
`MLX_AGENT_DUMP_SYSTEM` clean-room prompt dump (genmlx-qick) and the `sessionId` third arg that
keys engine state for the O(1) pi-session fork (genmlx-lin9).

→ Widen `StreamableSession` with `supportsImages()` / `contextLimits()`, implement them on
`GenmlxSession` / `GenmlxModelHost`, and either re-add `sessionId` or migrate to upstream's
`rootCacheOwnerId`. Genuine design work.

**`packages/agent/src/provider/convert-messages.ts`** — 5 hunks, 174 lines; the most fragmented
conflict. Both sides independently built the *same feature* (VLM image plumbing) with
incompatible architectures. Ours (genmlx-etfm/5aah): unconditional `splitParts` + a byte-stable
`TOOL_IMAGE_HOIST_TEXT` synthetic user message, explicitly designed so the replayed prefix never
varies and native KV reuse survives. Theirs: a `supportsImages` flag threaded everywhere,
placeholder rendering, stale-note stripping, and a changed return type
`ConvertedMessage { message, toolResultImages? }`.

Per-hunk cherry-picking is **impossible**: the `assistant` case between the hunks already
auto-merged to upstream's `{ message: converted }` shape, so keeping our `user`/`toolResult`
hunks yields a function with two incompatible return types.

→ **Adopt theirs wholesale.** Port forward only the typed
`IMAGE_CHANGE_REQUIRES_SESSION_RESTART` rejection, then **re-establish and re-measure the
KV-prefix-stability property ours existed to guarantee** — that property is the reason our
version was written, and adopting theirs does not preserve it for free.

---

## 3. Permanent divergences

Carried deliberately; every sync re-litigates them. Keep this list short.

| divergence | why | re-check each sync |
|---|---|---|
| `.gitmodules` → `robert-johansson/mlx` | our MLX fork | `git diff up/main...HEAD -- .gitmodules` is exactly the 1-line hunk |
| `requirePagedCache` gated on Metal availability | paged is Metal-only; ungated it bricks `mlx agent` on CUDA | grep the gate survives |
| lazy `loadNativeHost()` latch in `model-host.ts` | keeps the agent import graph free of static native chains | `__test__/native-import-graph.test.ts` |
| registry allowlist widened to include genmlx models | otherwise they silently vanish from the UI | assert in `llm/pi_provider_test.cljs` |

Each one is a candidate for deletion via upstreaming — see `README.md`'s shrink program.

---

## 4. Sync log

| date | upstream | conflicts | notes |
|---|---|---|---|
| *(pending)* | `v0.0.8` `0ebeaa57` | 14 paths / 26 hunks | first sync under this ledger; measured 2026-07-25, not yet executed |
