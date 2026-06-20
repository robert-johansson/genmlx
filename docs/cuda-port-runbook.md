# GenMLX → Linux/CUDA Port: Execution Runbook

> **Tracking:** bean `genmlx-ste5` (Linux/CUDA port). **Audience:** the session running ON the CUDA machine.
> **Provenance:** produced 2026-06-20 by a 3-analyst research workflow, every file:line verified against the live tree.
> **Companion:** `docs/fork-to-zero-plan.md` (the stock-mlx-node architecture this builds on).


**Audience:** the next session, running ON the Linux/CUDA machine. Follow in order. File:line refs are relative to `/Users/robert/code/genmlx` unless noted. All three research reports (A1 build, A2 membrane, A3 MLX-fork) were re-verified against the live tree before writing this.

---

## 1. Feasibility verdict

**GenMLX can run on Linux/CUDA. Confidence: MEDIUM-HIGH for core inference, LOW for the LLM/paged-attn path.**

The pieces line up:
- The vendored MLX fork (v0.32.0) **already contains the full CUDA backend** (`mlx-node/crates/mlx-sys/mlx/mlx/backend/cuda/`, 66 entries) and a working `MLX_BUILD_CUDA` CMake path (`mlx/CMakeLists.txt:38,155-163`).
- **Our four special-function CUDA kernels are fully wired** — `lgamma.cu, digamma.cu, bessel_i0e.cu, bessel_i1e.cu` are listed in `mlx/backend/cuda/unary/CMakeLists.txt:17-20` with real device functors (verified). These compile and run on CUDA.
- **Cholesky/Inverse JVP-VJP autograd patches are graph-level** (`mlx/primitives.cpp`), backend-agnostic, no kernel — work unchanged.
- Upstream mlx-node **already wrote the Linux/CUDA `build.rs` branch** we need to port (it ships an experimental aarch64-Linux/CUDA-13 preview).

**The three real blockers (all verified present):**
1. **`SearchSorted::eval_gpu` is undefined on CUDA** — it has CPU + Metal impls only (`mlx/backend/{cpu,metal}/searchsorted.cpp`, no CUDA file), and is **NOT** in the CUDA `NO_GPU` list (`mlx/backend/cuda/primitives.cpp` has `Inverse`/`Cholesky`/`SVD`/… but no `SearchSorted`). This is a **hard link failure** of the whole `.node` addon, not a runtime fallback.
2. **`get_num_resources()`/`get_resource_limit()` defined only in `backend/metal/allocator.{h,cpp}:267,270`** but declared in shared `mlx/memory.h:69,73` and **called unconditionally** by the FFI (`mlx-sys/src/mlx_stream.cpp:252,271`). On a CUDA build → **undefined-symbol link failure**.
3. **`mlx-sys/build.rs` has no CUDA branch at all** — it is hardcoded to Metal/`xcrun` (verified: panics without metal toolchain, always links `framework=Metal/...`, only knob is `MLX_BUILD_METAL`). `MLX_DISABLE_METAL=1` today yields **CPU-only**, not CUDA.

**Biggest unknowns / risks (ranked):**
- **A. x86_64 vs aarch64.** Upstream mlx-node validated CUDA *only* on aarch64 (GB10/DGX Spark). If the arriving machine is x86_64+NVIDIA, you are on a less-trodden path (MLX C++ supports x86_64 CUDA, but mlx-node's surface crate has not been validated there). **Confirm the machine's arch first** — it changes risk materially.
- **B. CUDA toolkit version.** The fork's CMake **hard-rejects CUDA 13.1** (`mlx/CMakeLists.txt:159-162 FATAL_ERROR`) and needs cuDNN mandatory. Target CUDA 12.4–12.9 or ≥13.2.
- **C. searchsorted perf, not just correctness.** The cheap fix (add `NO_GPU(SearchSorted)`) unblocks the link but makes categorical sampling + systematic SMC resampling throw on a GPU stream → must route to CPU stream → SMC hot-loop slowdown. The proper fix (port the ~22-line Metal kernel to `.cu`) keeps it on GPU.
- **D. LLM/paged-attn path has no CUDA kernels** (`mlx-paged-attn` is `.metal`-only). Core GFI/distributions/inference do **not** need it; only `@mlx-node/lm` does. **Scope LLM out of v1 on CUDA.**

---

## 2. Prerequisites (verify on the box before building)

| Item | Requirement | Why / source |
|---|---|---|
| **GPU compute capability** | ≥ 7.0 (Volta); auto-detected. SM 90+ (Hopper) gets `a`-suffix kernels | `mlx/backend/cuda/CMakeLists.txt:152-175` |
| **CUDA Toolkit** | **12.4–12.9 or ≥13.2. NOT 13.1** (hard FATAL_ERROR) | `mlx/CMakeLists.txt:159-162`; ≥12.8 needed for `--compress-mode=size` |
| **cuDNN** | **Mandatory** (`find_package(CUDNN REQUIRED)`) — install cuDNN 9 | `mlx/CMakeLists.txt:155-158` |
| **NVIDIA driver** | Driver newer than the CUDA 12.4 release (for compress-mode) | `mlx/backend/cuda/CMakeLists.txt:145-150` |
| **System libs** | `liblapacke-dev` + BLAS (CPU-stream LAPACK for Cholesky/Inverse/SVD), build-essential, cmake ≥3.25, ninja | MLX CUDA build links LAPACK for CPU-stream linalg |
| **Rust** | stable (match repo's toolchain; `rustup` default) | NAPI build |
| **Node / Bun** | Node 18+ and/or Bun (GenMLX runs under `bun run --bun nbb`) | CLAUDE.md |
| **nbb** | pinned `1.4.208` (via the `nbb` script in `package.json`) | CLAUDE.md |
| **Network at configure time** | CCCL v3.1.3 + CUTLASS fetched by CMake | `mlx/backend/cuda/CMakeLists.txt:223-224` |
| **glibc** | ≥ 2.35 | MLX install docs |

Confirm arch + toolkit first:
```bash
uname -m                          # expect aarch64 (validated) or x86_64 (higher risk)
nvidia-smi                        # driver + GPU
nvcc --version                    # toolkit version — must NOT be 13.1
ls /usr/lib/*/libcudnn*           # cuDNN present?
dpkg -l | grep -E 'lapacke|liblapack|libopenblas'
```

---

## 3. Ordered build steps

### 3a. Patch the MLX fork for a clean CUDA link (do this FIRST — two link blockers)

These two edits make `libmlx` link on CUDA. Without them the `.node` addon never loads.

**Blocker 1 — searchsorted.** Choose ONE:
- **Fast (unblock link, CPU-stream the op):** add to `mlx-node/crates/mlx-sys/mlx/mlx/backend/cuda/primitives.cpp`, next to the existing `NO_GPU(Inverse)`/`NO_GPU(Cholesky)` block (verified at lines 30-31):
  ```cpp
  NO_GPU(SearchSorted)
  ```
  This makes searchsorted throw on a GPU stream; GenMLX must then route it to a CPU stream (see §3d / §4 note). Acceptable for first-boot.
- **Proper (keep on GPU):** port `mlx/backend/metal/kernels/searchsorted.metal` (~22 lines, one thread/value) to `mlx/backend/cuda/searchsorted.cu`, define `SearchSorted::eval_gpu`, and add the file to the CUDA CMake source list. Do this *after* a green first boot.

**Blocker 2 — resource-count symbols.** `get_num_resources`/`get_resource_limit` are Metal-only (verified `backend/metal/allocator.{h,cpp}:267,270`) but FFI-called unconditionally. Add CUDA/no-Metal definitions returning 0:
- Cleanest: add a non-Metal TU (e.g. compiled when `MLX_BUILD_METAL=OFF`) providing:
  ```cpp
  namespace mlx::core {
  size_t get_num_resources() { return 0; }
  size_t get_resource_limit() { return 0; }
  }
  ```
- Returning 0 is the intended degraded-host behavior — the membrane's `buffer-count-limit` already falls back via `(if (pos? l) l 499000)` (verified `mlx.cljs:929-931`), and `buffer-count-pressure?` with count 0 never trips → the genmlx-5ucd count-sweep becomes a harmless no-op on CUDA. **No CLJS change required for correctness.**

> Note: Cholesky/Inverse staying `NO_GPU` on CUDA is **not** a regression — they are CPU-stream-only on Metal too (MLX `check_cpu_stream` throws on any GPU stream). GenMLX already runs them on a CPU stream today.

### 3b. Add the CUDA branch to `mlx-sys/build.rs`

The current `build.rs` is Metal-only (verified: `metal_toolchain_available()` panic path, `compile_paged_attn_metallib` via `xcrun`, no `MLX_BUILD_CUDA`). Port upstream mlx-node's Linux/CUDA branch. Required changes:

1. **Gate Metal toolchain panic + `compile_paged_attn_metallib`** behind `target_os == "macos"`.
2. **Add a `target_os == "linux"` branch** to the cmake config:
   ```rust
   cfg.define("MLX_BUILD_CUDA", "ON")
      .define("MLX_BUILD_METAL", "OFF");
   if let Ok(arch) = env::var("MLX_CUDA_ARCHITECTURES") {
       cfg.define("CMAKE_CUDA_ARCHITECTURES", &arch);
   }
   ```
3. **Replace the unconditional Apple-framework links** with CUDA libs on Linux: `cudart, cublas, cublasLt, cufft, nvrtc, cuda, cudnn`, plus `lapack`/`blas` (and `lapacke`).
4. **`#[cfg]`-gate the C++ FFI bridge** Metal calls and any `mlx_paged_dispatch.cpp` Metal references off on Linux.

Reference: upstream mlx-node `crates/mlx-sys/build.rs` already has exactly this branch (`MLX_BUILD_CUDA=ON`/`MLX_BUILD_METAL=OFF`, links the libs above, reads `MLX_CUDA_ARCHITECTURES`). Rebase our patches onto stock/latest mlx-sys, **keeping our vendored MLX fork**, rather than hand-writing it.

> **CUDA is not cross-compilable from macOS.** Build natively on the box.

### 3c. Build the native addon (genmlx-core → `.node`)

```bash
# from mlx-node/
export MLX_CUDA_ARCHITECTURES=90a      # match nvidia-smi: 90a Hopper/GB10, 89 Ada, 80 Ampere
# first cmake configure fetches CCCL/CUTLASS — needs network
yarn build:native                      # canonical native build (cargo build alone does NOT make the .node)
```
Expected first-run failure modes and where they point:
- `undefined symbol ...SearchSorted::eval_gpu` → §3a blocker 1 not applied.
- `undefined symbol ...get_num_resources` → §3a blocker 2 not applied.
- `CUDA Toolkit 13.1 is not supported` → wrong toolkit (§2).
- Metal/`xcrun`/`framework=Metal` errors → §3b not applied (build.rs still Metal-only).

Also handle the napi packaging single-target hardcode:
- `packages/core/package.json:43-44` declares only `aarch64-apple-darwin` + `@mlx-node/core-darwin-arm64`. Add the Linux triple (`aarch64-unknown-linux-gnu` or `x86_64-unknown-linux-gnu`) to napi `targets`, add a `linux-*` npm sub-package (`"os":["linux"],"cpu":[...]`), and a matching optionalDependency.
- `packages/core/build.ts:72,145` hardcodes `mlx-core.darwin-arm64.node` and **requires** `mlx.metallib`/`paged_attn.metallib` (throws if absent). **Gate `copyMetallibs()` to darwin** and make the `.node` name target-derived. There is no metallib on Linux.

### 3d. Wire @genmlx/core and run GenMLX

```bash
# from /Users/robert/code/genmlx/
npm install                            # picks the linux optionalDependency for @genmlx/core
node -e "const m=require('@genmlx/core'); console.log(Object.keys(m).length)"   # addon loads?
bun run --bun nbb -e '(require "@genmlx/core")'   # loads under nbb
```
If §3a used the `NO_GPU(SearchSorted)` fast path, expect runtime throws when categorical sampling / systematic SMC resampling hit a GPU stream. The pragmatic fix is CPU-stream routing for `searchsorted`, `cholesky`, `cholesky-inv`, `linalg-inv` (and any SVD/eig) — these NAPI exports currently take **no** stream arg (`mlx-node/crates/genmlx-core/src/genmlx.rs:277,447,471,481`) and `mlx.cljs:1372-1375 set-default-device!` is a **no-op**. So routing needs a stream/device param plumbed through those NAPI signatures + a real device selector in `mlx.cljs`. **This is the main GenMLX-layer work item** — without it the GPU path crashes even after a green compile.

---

## 4. Membrane changes (do + test on Mac NOW, before the machine arrives)

Goal: make the membrane CUDA-neutral while staying byte-identical on Metal (everything behind `(metal-is-available?)`, which is `true` on Metal). These are safe to land and verify on the Mac today.

**(Optional, recommended) Make the count-heuristic explicitly backend-aware** instead of relying on the count-returns-0 coincidence. In `src/genmlx/mlx.cljs`:
- Add a backend probe near `metal-is-available?` (`mlx.cljs:964`). **Ordering caveat:** `buffer-count-limit` is defined at `:929`, *before* `metal-is-available?` at `:964` — either move the probe up or inline the `(.metalIsAvailable c)` check. Define something like `count-tracking-available? = (and metal-available? (pos? buffer-count-limit))`.
- Short-circuit `buffer-count-pressure?` (`mlx.cljs:1035-1051`) to return `false` when `count-tracking-available?` is false — skips the `get-num-resources` FFI hop on CUDA and documents intent.

**(Optional, low priority) Honest device introspection.** `metal-device-info`/`gpu-architecture-gen` (`mlx.cljs:965-989`) and `default-device`/`set-default-device!` (`mlx.cljs:1367-1376`) hardcode Apple-GPU assumptions. On CUDA, `metal-is-available?` is false though a GPU exists. Branch `metal-device-info` on `(metal-is-available?)` and read `gpu::device_info()` (returns a real `sm_XX` name) when false. Diagnostic-only.

**Do NOT touch** the `set-default-device!` no-op as a "fix" in isolation — making CPU-stream routing actually work requires the NAPI stream-param plumbing in §3d, which is a coordinated Rust+CLJS change, not a membrane one-liner.

**Mac-green guarantee:** all edits are gated on `(metal-is-available?)` = true on Metal, so the count-sweep keeps firing exactly as today. Verify on the Mac with the unchanged native-contract suites: `clip_contract_test`, `membrane_coverage_test` (the two new C++ symbols add **no** `@mlx-node/core` exports, so the coverage matrix / `intentional-omissions` pin at `membrane_coverage_test.cljs:205` is unaffected), `exact_test`, `level0_certification_test` (68/68).

---

## 5. Validation on the CUDA box (expected results)

Run in this order; each gates the next. (`bun run --bun nbb test/genmlx/<f>.cljs`)

1. **Addon loads** (§3d node/nbb require) — no link/symbol error. *If this fails, you have a build problem, stop and fix §3a/§3b.*
2. **`membrane_coverage_test`** — surface-drift guard; should be **green/unchanged** (no new exports).
3. **`exact_test`** — exercises cholesky/inverse via the CPU stream. *This is the first place a missing CPU-stream-routing fix will throw* (`has no CUDA implementation`). Green = linalg routing is correct.
4. **`gradient_fd_test`** + **`score_gradient_test`** — **these exercise the CUDA special-function kernels** (gradients of gamma/beta/student-t/etc. flow through `lgamma`/`digamma`/`bessel_i0e`/`bessel_i1e`, which are the four `.cu` kernels we wired in `unary/CMakeLists.txt:17-20`). Green here = **our CUDA kernels are correct on real hardware** — the single most important functional signal of the port.
5. **`clip_contract_test`** — `Either<&MxArray,f64>` bounds contract still holds.
6. **`level0_certification_test`** — must pass **68/68**. Broad smoke over simulate/generate/update/regenerate.
7. **Vectorized + SMC** (`vectorized_test`, anything touching `compiled_smc`) — exercises `searchsorted`. *This is where the searchsorted decision shows:* throws (if `NO_GPU` fast-path + no CPU routing), runs on CPU (routed), or runs on GPU (`.cu` ported).

Expected end state for a v1 (core, no LLM): items 1-6 green; item 7 green once searchsorted is either CPU-routed or ported. LLM/paged-attn (`@mlx-node/lm`) is **out of scope** on CUDA (no `.cu` kernels).

---

## 6. Mac-now vs machine-required

**Can be done on the Mac NOW (before the machine arrives):**
- **All §3a MLX-fork source patches** — add `NO_GPU(SearchSorted)` (or write `searchsorted.cu`) and the `get_num_resources`/`get_resource_limit` non-Metal definitions. Pure source edits; commit them. (You can't *CUDA-compile* them on Mac, but you can write + review them, and confirm the Metal build still links.)
- **All §3b `build.rs` CUDA-branch code** — write the Linux branch + Mac `#[cfg]` gating. Verify the **macOS path still builds** (`yarn build:native` on Mac stays green — the Linux branch is `cfg`-dead on Mac).
- **§3c packaging edits** — napi Linux triple in `package.json`, gate `copyMetallibs()` to darwin in `build.ts`, make `.node` name target-derived. Verify Mac build unaffected.
- **All §4 membrane edits** — and run the full §5 contract suite on the Mac (Metal) to prove zero regression.
- **Pre-write the CPU-stream-routing change** (§3d main work item): the NAPI stream-param signatures in `genmlx-core/src/genmlx.rs` + the `mlx.cljs` device selector. Reviewable on Mac; only *exercisable* on the box.

**Requires the machine (cannot be done until it arrives):**
- CUDA compilation of MLX + the four special-fn kernels (`nvcc`, CCCL/CUTLASS fetch).
- The native `yarn build:native` CUDA link (proves §3a blockers are resolved).
- Producing the Linux `.node` (NAPI CUDA is **not cross-compilable from macOS** — native build only).
- Every §5 validation item (real GPU execution).
- searchsorted GPU-vs-CPU perf decision (measure on the box).

---

## 7. Risks + fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| **searchsorted link failure** | Certain without §3a-1 | Fast: `NO_GPU(SearchSorted)` + CPU-stream route (slower SMC). Proper: port 22-line Metal kernel → `searchsorted.cu`. |
| **resource-symbol link failure** | Certain without §3a-2 | Add non-Metal `get_num_resources`/`get_resource_limit` → 0. Count-sweep degrades to no-op automatically (membrane fallback already present). |
| **build.rs has no CUDA path** | Certain | Port upstream mlx-node's Linux branch (it already exists upstream); don't hand-write from scratch. |
| **Cholesky/Inverse/SVD throw on GPU stream** | Certain on GPU path | Route to CPU stream (they're CPU-only on Metal too — not a regression). Needs NAPI stream-param plumbing (§3d). |
| **CUDA 13.1 hard-rejected** | If box ships 13.1 | Install 12.4–12.9 or ≥13.2; cuDNN 9 mandatory. |
| **x86_64 not validated upstream** | If box is x86_64 | MLX C++ supports it; expect more debugging. If aarch64, you match upstream's validated GB10/DGX-Spark path. |
| **napi cross-build from Mac** | Certain limitation | Build natively on the Linux box. Mac can prep source + Metal-side verification only. |
| **paged-attn / LLM has no CUDA kernels** | Certain | Scope `@mlx-node/lm` out of v1. Core GFI/distributions/inference don't need it. |
| **genmlx-5ucd buffer-count safety net absent on CUDA** | By design | CUDA allocator has no ~499000 buffer-count wall; byte-pressure heuristics (`auto-cleanup!` 512MB, `gfi-cleanup!` 128MB) still work on CUDA's real `get-active-memory`. Memory behavior on CUDA is **unvalidated** — watch it during §5. |

**Bottom line:** Build mechanics are tractable (the CUDA backend + our kernels already exist; the missing wiring is small and upstream-precedented). The real work is (a) the two one-to-few-line MLX link fixes, (b) porting the `build.rs` Linux branch + packaging, and (c) CPU-stream routing for searchsorted/Cholesky/Inverse so the GPU path doesn't crash post-compile. Items (a) and (b) source and all §4 membrane edits can be written and Metal-verified on the Mac **now**; (c) can be written now and exercised only on the box. Target **aarch64 Linux, CUDA 12.x, inference/core-first, LLM deferred**.

Key verified file:line anchors: `mlx-sys/build.rs:5-69` (Metal/xcrun-only, no CUDA); `mlx/backend/cuda/primitives.cpp:30-31` (`NO_GPU(Inverse)`/`NO_GPU(Cholesky)`, no `SearchSorted`); `mlx/backend/{cpu,metal}/searchsorted.cpp` (CPU+Metal only, no CUDA); `mlx/backend/metal/allocator.cpp:267,270` + `mlx/memory.h:69,73` (resource symbols Metal-only-defined); `mlx/backend/cuda/unary/CMakeLists.txt:17-20` (our four `.cu` kernels wired); `src/genmlx/mlx.cljs:911-932` (`get-num-resources`/`buffer-count-limit` with `(if (pos? l) l 499000)` fallback), `mlx.cljs:1372-1375` (`set-default-device!` no-op); `mlx/CMakeLists.txt:38,155-163` (`MLX_BUILD_CUDA` + cuDNN/CUDA-13.1 reject). Sources: [mlx-node](https://github.com/mlx-node/mlx-node), [MLX CUDA discussion #2422](https://github.com/ml-explore/mlx/discussions/2422), [MLX issue #847](https://github.com/ml-explore/mlx/issues/847), [MLX build docs](https://ml-explore.github.io/mlx/build/html/install.html).