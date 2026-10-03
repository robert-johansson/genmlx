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

### 1d. Generated file — CORRECTED 2026-07-26, the original entry was wrong

`packages/core/index.d.cts` carries 119 lines covering our genmlx surface (`forward`,
`forwardWithCache`, `initCaches`, `branchCache`, `forwardBranch`, `vlmPrefillFlat`, …).

**The original entry called these "hand-added" and had them deleted pre-merge (commit
`f5dc863e`). That premise was wrong.** They are not hand edits — they are the *generated* output
of the `#[napi]` doc comments on our own forked Rust (`crates/mlx-core/src/models/qwen3_5*/model.rs`).
Running `yarn build:native` regenerated the file and restored all 119 lines verbatim, **plus**
upstream v0.0.8's new `supportsImages()` / `contextLimits()` declarations. The regenerated file is
therefore strictly more correct than either side of the merge, and is what gets committed.

The amputation did no harm (the generator undoes it) but it bought nothing, and it did **not**
remove 2 files from the both-touched set as claimed.

What survives from the original reasoning: **never hand-edit `index.d.cts`.** Edits not backed by
a Rust doc comment genuinely would vanish at the next regeneration. Our lines were backed, which
is exactly why they came back.

**Lesson for the next sync:** before "amputating" anything from a generated file, run the
generator and diff. A file being generated is an argument for regenerating it, not for deleting
content from it.

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
| `agentPagedCacheSupported()` gating `requirePagedCache` **and** the paged config overlay | paged is Metal-only; ungated it throws on every `mlx agent` model load on CUDA | `agentPagedCacheSupported` unit test in `run-agent.test.ts` |
| lazy `loadNativeHost()` latch in `model-host.ts`, plus `createPagedConfigOverrides()` replacing upstream's **value** import of `PagedConfigOverrideManager` | keeps the agent import graph free of static native chains | `__test__/native-import-graph.test.ts` |
| registry allowlist widened to the union of `opts.models` + `opts.genmlxModels`, and `model-registry-filter.ts` widened via `LOCAL_PROVIDER_BASE_URLS` | upstream's predicate hard-requires `provider === 'mlx' && baseUrl === 'mlx://local'`, so genmlx models silently vanish from Tab / `/models` / RPC enumeration / session restore | the allowlist test in `run-agent.test.ts` |
| `MLX_AGENT_DUMP_SYSTEM` prompt dump (genmlx-qick) | clean-room persona verification | grep it survives in `stream-adapter.ts` |
| precise f32 SwiGLU in `nn::RMSNormGated::forward` (mlx-ogvd) | upstream's `swiglu_compiled` runs in bf16; our CUDA parity oracle (`qwen35_moe_forward_parity_test`, `llm_forward_xval_mlxlm.py`) pins the mlx_lm `_precise_swiglu` numerics | the parity tests; the fn carries a comment naming this ledger |
| `StreamableSession` carries `dispose` + non-streaming `startFromHistory` (2026-10-03) | upstream's `shared-host.test.ts` exercises both through the host callback; `GenmlxSession` implements them (dispose = reset, startFromHistory drains the stream) so the genmlx provider stays a first-class host session | `yarn typecheck` + `shared-host.test.ts` |
| no `-DCMAKE_PROJECT_INCLUDE=` on the non-Metal path of `mlx-sys/build.rs` | CMake reads the empty value as a directory and every `project()` fails; upstream only builds on macOS where the overlay branch always runs | the Linux build itself; upstream-PR candidate |
| trailing optional `preserveThinking` on the napi `applyChatTemplate` (`tokenizer.rs`) | our fork pinned `preserve_thinking => true` inside the render; upstream's rewrite made it caller-controlled with HF-parity `false` as the napi default. GenMLX commits the rendered stream as a reusable KV prefix and re-renders the grown history each turn, so `false` strips the previous turn's think block and every turn cold-prefills (`cachedTokens` 0 — the three pi_* reds of the 2026-10-03 battery). The arg is additive (upstream's TS callers unchanged); GenMLX passes `true` through `llm/render-chat-template`. Upstream-PR candidate. | `pi_provider_test` / `pi_fork_test` / `pi_assess_test` (delta-prefill assertions) |
| mlx patch `e947e13d8`: typed `CompileOptions::Data{}` in `cuda/custom_kernel.cpp` | mlx-node's `83818b9c7` hand-edited the CUDA call sites with a bare `{}`, which cannot deduce through `std::make_shared`; never compiled upstream | the Linux build; upstream-PR candidate against mlx-node/mlx |
| native-free `@mlx-node/lm/*` subpath imports in the agent package (`chat-config.ts` → `family-data`; `events.ts` → a new `./tool-call-buffer` export in `packages/lm/package.json`) and `native-import-graph.test.ts` allowing the five native-free subpaths plus the dynamic-only native owner `provider/shared-host.js` (2026-10-03) | upstream reintroduced nine static imports; the `@mlx-node/lm` barrel dlopens `@mlx-node/core`, the subpaths are `node:*` + type-only imports (verified by reading them, not assumed). The gate now also asserts native-owner files are never statically imported | `native-import-graph.test.ts` — needs a fresh `yarn build:ts` dist |
| `GenmlxSession` mirrors `defaultConfig` + `activeTools` (2026-10-03) | upstream's warm-reuse helper gained the pair (tools are conversation state in `ChatSession`); the drift test asserts every touched field exists on the genmlx session | `genmlx-session.test.ts` "warm-reuse-touched fields exist" |
| `installMlxOnlyModelRegistryFilter` restore recomposes every runtime the choke touched (2026-10-03) | pi's `registerProvider` → `refresh` → `rebuildProviders` runs under the choke and strips cloud providers from the INSTANCE's model map; un-patching the prototype cannot give them back | `run-agent.test.ts` "allowlists genmlx models too" |
| `qwen3_next` pinned into upstream's two family-list expectations (`chat-config.test.ts`, `paged-config-override.test.ts`) | consequence of the `qwen3_next` family-data row | both tests |
| `@mlx-node/core-darwin-arm64` optionalDependency RESTORED to upstream's line (retires genmlx-w837 option 3, 2026-10-03) | upstream's `publish-targets.test.ts` requires the workspace edge; the stale-prebuilt trap w837 closed stays covered by the 91b3 runtime guard (the template is binary-less, `require` throws loudly) | `publish-targets.test.ts` |
| `packages/core/build.ts`: off macOS, the post-generation declaration guard KEEPS the committed `index.d.cts` after a subset check | napi typings generated on Linux are lossy — `cfg(target_os = "macos")` members (`Qwen3AsrCapture`, `Qwen3AsrModel`) vanish and `tsc -b` breaks in `packages/asr`. The committed copies are macOS-authoritative plus our cross-platform additions (23 lines on 2026-10-03); a host surface that GREW fails the build loudly | `yarn build:native` exits 0 on Linux; `yarn build:ts` green |

Each one is a candidate for deletion via upstreaming — see `README.md`'s shrink program.
`agentPagedCacheSupported` and the `LOCAL_PROVIDER_BASE_URLS` widening are both good upstream PRs:
neither is genmlx-specific (any CUDA/Linux user hits the first; any second local provider hits
the second).

### Divergences RETIRED by the v0.0.8 sync

Deleting a divergence is the point of the exercise — record them so they are not re-introduced.

| retired | how |
|---|---|
| the `sessionId` third arg to `startFromHistoryStream` | migrated onto upstream's sibling field: `buildChatConfig` sets `config.cacheOwnerId = options.sessionId`, byte-identical to what our third argument carried. **Not** `rootCacheOwnerId` — that carries the *root* session id, which would collapse every subagent session onto the root's engine session and misalign its delta prefill. `StreamableSession` is back to upstream's 2-parameter shape. |
| our 119 hand-edited `packages/core/index.d.cts` lines | deleted pre-merge; the file is generated |
| our `vision_features_cached` second vision cache | continuation path migrated onto upstream's per-image cache, inheriting its budget and eviction policy |
| our `splitParts` / `TOOL_IMAGE_HOIST_TEXT` rendering | adopted upstream — but the property it guaranteed is **weakened**, tracked in `genmlx-8hod` |

### Divergences RETIRED by the 2026-10-03 sync (`up/main` f6db56b1)

| retired | how |
|---|---|
| two-model draft speculative decoding (genmlx-orsr: `qwen3_5/draft.rs`, `Qwen35DraftStepper`, `hasDraftModel`, the dense `load(path, paged_override, options)` shape) | upstream's DFlash2 companion now owns `draftModelPath` for Qwen3.5 dense and implements the engine's `DsparkBackend` for `Qwen35Inner` itself; re-hosting ours meant re-porting the whole flat decode loop. Nothing in GenMLX or the SCI scripts referenced it. Recoverable from `pin/mlx-node/2026-08-08`. |
| one `generation_stream: Stream` per model (genmlx-d3yn) | upstream's `Stream::generation()` (one persistent stream per model thread, #171) is the same fix; the field is gone from `Qwen35Inner`/`Qwen35MoeInner`, kept only in `Qwen3Inner` where upstream left it |
| `tools => tools_value.unwrap_or_default()` in the chat-template render | upstream now OMITS the key when no tools are given; Qwen3-Coder-Next's template guards with `{% if not tools is defined %}{% set tools = [] %}`, exactly the Nemotron-H case their change was written for |
| `preserve_thinking => true` pinned inside the same render block | NOT silently retired: taking upstream's block dropped it, the battery's three pi_* reds found it (cachedTokens 0 every turn), and it came back as the explicit `preserveThinking` napi argument above — the lesson for the next sync is that a conflict block can carry TWO of our behaviors, and resolving the one you examined is not resolving the block |
| `mlx_gated_delta_chunked` (chunked GDN prefill C++ kernel) | no caller left on either side after upstream's GDN rework; dropped with the conflict |
| `detectModelTypePure` + `RAW_MODEL_TYPE_ALIASES` in `packages/agent/src/provider/models.ts` | upstream's `@mlx-node/lm/model-discovery` is native-free (fs + family-data only) — the property our mirror existed to guarantee |
| `packages/server/src/presets.ts` `qwen3_next` launch preset | moved with upstream's #132 into a `MODEL_FAMILY_DATA` row in `packages/lm/src/family-data.ts` (kind `trainable`, traits `reasoning: false`, Qwen3-Coder sampling) |

**Deferred, not retired — needs its own bean:** the whole-turn FLAT VLM chat path on CUDA (genmlx-9v44 dense / genmlx-52mh MoE — `mlx agent` image turns without the paged cache). It lived in the `ChatBackend` impl regions upstream deleted when it split `model.rs`, and upstream's vision turns are paged-only. The GenMLX-facing flat prefill (`vlmPrefillFlat` + `forwardWithCache`/`forwardBranch` rope-shifted continuation) IS re-ported in `qwen3_5_moe/model/genmlx_surface.rs`.

---

## 4. Sync log

| date | upstream | conflicts | notes |
|---|---|---|---|
| 2026-07-26 | `v0.0.8` `0ebeaa57` | 14 paths / 26 hunks | first sync under this ledger; measured 2026-07-25, executed 2026-07-26 (`pin/mlx-node/2026-07-26`); §1d corrected from its outcome |
| 2026-08-08 | `up/main` `2d1fe60e` (v0.0.10-4: #112 ASR, #114 layer-kind cache, #115 MTP gate) | 2 paths / 4 hunks, **no gitlink conflict** | Cheapest sync yet. All 4 hunks were pure additive struct/initializer UNION in `qwen3_5/model.rs` + `qwen3_5_moe/model.rs`: ours `generation_stream: Stream` (genmlx-d3yn) vs theirs `layer_kinds` (#114). The gitlink did **not** conflict — upstream's mlx pin is unchanged at `fef8890f5`, so only our side had moved and git kept ours; `update-index --cacheinfo` was unnecessary (the first sync where that was true). **mlx untouched** at `2324cb204`; superset invariant re-verified (`git cherry` returns no `+`). Zero §2-class conflicts and zero §3 re-litigation: upstream never touched `packages/agent` this range. One bracketing hazard checked by hand — our `MLX_GUARD_VOID` wraps `mlx_qwen3_forward_step` in `mlx_advanced_ops.cpp` while upstream edited 6 hunks *inside* it; guard still opens after the failure prologue and closes before the function brace. No spec-decoding collision: ours is the new file `qwen3_5/draft.rs`, theirs is `mtp_turn.rs`/`mtp_decode.rs` — different mechanisms (two-model draft vs MTP self-speculation), disjoint files, but **whether both can be enabled at once is untested**. Surface drift +6 exports / −0 (ASR five + the `Qwen3AsrCaptureSource` enum), re-pinned 240→245 fns / 55→60 omissions under a new `:asr-speech` category. Battery DEFERRED at user request — see the `pin/mlx-node/2026-08-08` tag message. Bean `genmlx-bcyp`. |
| 2026-10-03 | `up/main` `f6db56b1` (v0.0.15+13: Qwen3.8 line #129/#130/#143/#154/#171, `qwen4_exp` Flash-Next, shared model traits #138, family-data rows #132, deps #133) | 30 paths / ~80 hunks, **gitlink conflict** (their pin → `053e43fe`) | The hardest sync so far, and the cost was STRUCTURAL, not the hunk count: upstream split `qwen3_5/model.rs` (ours 722 KB) into `model/{forward,flat_turn,paged_turn,training,lifecycle,...}.rs`, so git showed our whole surface as 10k/9.5k-line "ours" blocks that were really upstream's own deleted code with our additions interleaved. Resolution: take theirs everywhere the file was split, then RE-PORT from the saved pre-merge tree: dense + MoE `model/genmlx_surface.rs` (forward/forwardWithCache/initCaches, branch*, vlmPrefillFlat/vlmVisionFeatures, rope-shifted continuation), command variants + arms in both `model/commands.rs`, Qwen3 dense arms, training dequantize (genmlx-x76x) into the shared `models/quantized_linear.rs` + both `model/training.rs` (with the genmlx-at2q seed hook), GDN `dequantize_to_standard` extended to upstream's split projections, precise SwiGLU into `nn/normalization.rs`, `qwen3_next` into a family-data row + `FAMILY_WRAPPERS`. Merge-time unions elsewhere: `gated_delta_net.rs` (our de-interleave inside their split-projection forward), both `persistence.rs` (`merged_qkvz` threaded through their `per_layer_quant` signature), `mlx-sys/src/lib.rs` (53 + 52 disjoint FFI decls), `mlx_common.h` (their `copy_to_buffer_as` template under our `MLX_GUARD_BOOL`). Silent-hazard roll call: 0 of our added lines missing across all 22 auto-merged both-touched files (incl. `attention.rs` +1812/−390 theirs over ours +62/−44, `convert.rs` +9828/−4016). mlx side: their pin = 25 nax commits on an ml-explore base OLDER than ours; the runbook §1 rebase replayed 25 + our 27 with zero conflicts (range-diff 27× `=`) — `merge-tree` had predicted 12, which is the wrong procedure for the gitlink and should be ignored. Two fork repairs upstream cannot see (macOS-only CI): the empty `CMAKE_PROJECT_INCLUDE` define and the CUDA `CustomKernel` ctor call. Retired: see §3. Deferred: flat VLM chat on CUDA (own bean). Surface drift +11 / −0 (K2Horizon/MuseGlimmer/NemotronH/Qwen4Exp model classes, ChatSessionCall, the GGUF catalog four, the `mlx eval` pair — all omitted; membrane re-pinned 245→256 fns / 60→71 omissions). Battery RTX sm_120, scale 6, 4/4-way, 101 min: **435/440 passed, 2 skipped** (gemma4/qwen2_coder fixtures); heavy 4/4. The 5 misses: 3 were ONE sync regression — the `preserve_thinking` pin lost with the tokenizer block (every pi turn cold-prefilled, `cachedTokens` 0) — fixed by the additive `preserveThinking` napi arg + `llm/render-chat-template` and re-passing 48/0, 15/0, 20/0; 1 one-off (`fused_mh_api`, 0/40 contended re-runs, cause now surfaced by the wrappers); 1 pre-existing (`control_metareasoner`, genmlx-6qz4). Contract guards + membrane green on the final addon. Hand-run: world_train 34/0, world_train_reward 39/0, owned_branch 23/0, branched 8/0, vlm_flat_branch 6/0; the two VLM e2e suites SKIPPED (no test image). `llm_forward_xval_mlxlm.py` NOT run (venv drift, unrelated). Also fixed en route: the genmlx-a9kf `var` typing (hard build failure under @napi-rs/cli 3.8). Bean `genmlx-9fvg`. **Same-day follow-up (`822274c5`, `e40c3b6c`; the tag was re-cut on `e40c3b6c`):** the battery validates `@genmlx/core` only — rebuilding the sibling `@mlx-node/core` (`yarn build:native`) and running mlx-node's own TS suite took it from 107 failing files (stale addon) to 23, of which 6 were real sync ripples (import-graph purity after the family-data move, warm-reuse fields, registry-filter restore, two family-list pins, the retired `./presets` export) — all fixed, ledger §3 rows added — and the rest are missing fixtures (12: tokenizer/model/gsm8k), macOS-only desktop packaging (3), and `examples/` deps (1). Declarations regenerated on Linux are lossy (`cfg(macos)` napi members vanish): committed copies = upstream's macOS text + our 23 lines, `build.ts` guard adapted. |
| 2026-07-27 | K-quants `b89b84c` (PR #101) | 3 paths / 6 hunks (+ gitlink) | metallib-select.ts+test → THEIRS (upstream independently wrote our genmlx-lr9c fix — divergence erased); build.rs → their switch-exhaustiveness comment + our `110a;120a;121a` arch default (their hunk reverted to the `121a` arch-locked-cubin bug). MLX side: 14 theirs + 17 ours replayed, zero conflicts, range-diff all `=`. Follow-up commit: `__fp16`→`_Float16` portability fix in vendored ggml (x86_64-GCC build break; upstream-PR candidate). Validated RTX sm_120: battery 431/431. Bean `genmlx-an7d`. |
