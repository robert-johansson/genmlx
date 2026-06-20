# GenMLX → Linux/CUDA Port: Execution Runbook (v2)

> **Tracking:** bean `genmlx-ste5`. **Audience:** the session running ON the Linux/CUDA machine.
> **Provenance:** v1 produced 2026-06-20 by a 3-analyst research workflow; **v2 (2026-06-20)** re-verified by a 5-analyst workflow (build.rs / C++ link / packaging / membrane / adversarial blocker-audit), every file:line checked against the live tree + the `upstream` mlx-node remote.
> **Companion:** `docs/fork-to-zero-plan.md` (the stock-mlx-node architecture this builds on).

File:line refs are relative to `/Users/robert/code/genmlx` unless noted. Follow the steps in order.

---

## 0. The correction that changes everything (READ FIRST)

**v1 of this runbook said the "main GenMLX work item" was CPU-stream routing — plumbing a stream/device param through the cholesky/inverse/svd/eig NAPI exports + a real device selector in `mlx.cljs`. THAT IS WRONG AND UNNECESSARY. Do not do it.**

Every linalg op is **already** pinned to `Device::cpu` one layer below the NAPI, in the mlx-sys C++ shim (`mlx-node/crates/mlx-sys/src/mlx_linalg.cpp:6-9`):

```cpp
// Linalg ops are CPU-only in MLX. Use a CPU stream explicitly.
static mlx::core::Stream cpu_stream() { return default_stream(Device::cpu); }
...
linalg::cholesky(*arr, upper, cpu_stream());   // + inv, tri_inv, cholesky_inv,
                                               //   solve, solve_triangular, svd, eigh, eigvalsh, qr
```

So cholesky / inverse / svd / eigh / qr / solve **run correctly on CUDA with zero code change** — they are `NO_GPU` on CUDA *and* on Metal, and both already route to the CPU stream. No NAPI signature change, no `set-default-device!` plumbing.

**The adversarial audit (HIGH confidence) confirmed: `searchsorted` is the ONLY GenMLX-reachable op that fails on a CUDA build — and it fails as a LINK error (undefined symbol `SearchSorted::eval_gpu`), not a runtime throw.** Evidence:

- All **10** linalg shims are CPU-pinned (`mlx_linalg.cpp` lines 20/29/40/48/56/64/78/91/104/113). `norm` (lines 119/130) runs on GPU but has a CUDA `eval_gpu`, so it is fine.
- The CUDA `NO_GPU` list (`mlx/backend/cuda/primitives.cpp:27-37`) is `LUF, QRF, SVD, Inverse, Cholesky, Eig, Eigh, Send, Recv`. The 7 linalg entries are unreachable on a GPU stream (CPU-pinned). `Send/Recv` (distributed) are **not** exposed by GenMLX (zero distributed exports in `genmlx-core/src/genmlx.rs`).
- All **118** declared primitives were diffed against CUDA `eval_gpu` coverage. The only primitive GenMLX reaches that has **no** CUDA impl **and** is **not** `NO_GPU`'d is `SearchSorted` (`eval_gpu` exists only in `mlx/backend/metal/searchsorted.cpp:10`).
- The 4 GenMLX special-function `.cu` kernels are wired with **real math** (not stubs): `cuda/unary/CMakeLists.txt:17-20` lists `lgamma.cu / digamma.cu / bessel_i0e.cu / bessel_i1e.cu`; the device functors in `cuda/device/unary_ops.cuh:134-253` use `::lgamma`, a digamma asymptotic series, and Cephes Chebyshev coefficients for the bessel functions.
- Fork-patch audit (the 9 add-only MLX commits): they added exactly 5 primitive classes — `BesselI0e, BesselI1e, LogGamma, Digamma` (all got CUDA paths) and `SearchSorted` (commit `d9d8bb9e` touched only `backend/cpu` + `backend/metal`, **zero** `backend/cuda`). No other fork op lacks a CUDA path.

`searchsorted` is reached on the default (GPU) stream from three CLJS hot-paths: `mlx.cljs:382`, `vectorized.cljs:31` (systematic resampling), `inference/compiled_smc.cljs:50` (the **default** SMC resample). So it must be handled or the addon will not link.

**Net: the entire port is (a) one `build.rs` Linux branch + one companion stub file, (b) three small C++ edits for searchsorted + the resource symbols, (c) a committed build script for the `genmlx-core` addon, and (d) the validation run. The membrane edits are already done + Metal-verified.**

---

## 1. Feasibility verdict

**GenMLX can run on Linux/CUDA. Confidence: MEDIUM-HIGH for core inference, LOW for the LLM/paged-attn path (out of v1 scope).**

The vendored MLX fork (v0.32.0) already contains the full CUDA backend (`mlx-node/crates/mlx-sys/mlx/mlx/backend/cuda/`) and a working `MLX_BUILD_CUDA` CMake path. Cholesky/Inverse JVP-VJP autograd patches are graph-level (`mlx/primitives.cpp`), backend-agnostic. Our 4 special-fn kernels compile and run on CUDA. **Upstream mlx-node already wrote and tested the `build.rs` Linux/CUDA branch** (commit `87a59fc`, "aarch64-unknown-linux-gnu (NVIDIA GB10 / DGX Spark) CUDA backend support, #71") — we are exactly one feature commit behind on that file.

**Risks (ranked):**
- **A. x86_64 vs aarch64.** Upstream validated CUDA *only* on aarch64 (GB10/DGX Spark). x86_64+NVIDIA is a less-trodden path (MLX C++ supports it; the surface crate is unvalidated there). **Confirm `uname -m` first.**
- **B. CUDA toolkit version.** The fork's CMake hard-rejects CUDA 13.1 (`mlx/CMakeLists.txt:159-162 FATAL_ERROR`); cuDNN mandatory. Target CUDA 12.4–12.9 or ≥13.2, cuDNN 9.
- **C. searchsorted perf.** The fast fix (CPU-route) keeps correctness but a host round-trip in the SMC resample loop. The proper fix (port the kernel to `.cu`) keeps it on GPU. Do fast first, proper after a green boot.
- **D. LLM/paged-attn has no CUDA kernels** (`.metal`-only). Core GFI/distributions/inference do not need it. **Scope `@mlx-node/lm` out of v1.**

---

## 2. Prerequisites (verify on the box before building)

| Item | Requirement | Source |
|---|---|---|
| **GPU compute capability** | ≥ 7.0 (Volta); SM 90+ gets `a`-suffix kernels | `mlx/backend/cuda/CMakeLists.txt:152-175` |
| **CUDA Toolkit** | **12.4–12.9 or ≥13.2. NOT 13.1** (hard FATAL_ERROR) | `mlx/CMakeLists.txt:159-162` |
| **cuDNN** | **Mandatory** — install cuDNN 9 | `mlx/CMakeLists.txt:155-158` |
| **System libs** | `liblapacke-dev` + BLAS/LAPACK (`libblas.so` = CBLAS, `liblapack.so`), build-essential, cmake ≥3.25, ninja | MLX links LAPACK for CPU-stream linalg |
| **Rust / Node / Bun / nbb** | stable rust; Node 18+ and/or Bun; nbb pinned `1.4.208` | CLAUDE.md |
| **Network at configure time** | CCCL v3.1.3 + CUTLASS fetched by CMake | `mlx/backend/cuda/CMakeLists.txt:223-224` |
| **glibc** | ≥ 2.35 | MLX install docs |

```bash
uname -m                          # aarch64 (validated GB10/DGX) or x86_64 (higher risk)
nvidia-smi                        # driver + GPU; note compute capability (90a Hopper/GB10, 89 Ada, 80 Ampere)
nvcc --version                    # toolkit — must NOT be 13.1
ls /usr/lib/*/libcudnn*           # cuDNN present?
dpkg -l | grep -E 'lapacke|liblapack|libopenblas|libblas'
```

> **`MLX_CUDA_ARCHITECTURES`**: the `build.rs` Linux branch defaults to `121a` (GB10/sm_121). On any other GPU you **must** set it: `90a`=H100, `89`=L40/4090, `80`=A100. Export it before building.

---

## 3. The build, step by step

### 3a. Patch `build.rs` for the Linux/CUDA branch + add the companion stub

We are one commit behind upstream on this file. **Two ways to apply — pick one:**

#### Option B — take upstream's tested file verbatim (fastest)
```bash
# from mlx-node/  (our fork has `upstream` = mlx-node/mlx-node configured)
git fetch upstream
git checkout 87a59fc -- crates/mlx-sys/build.rs crates/mlx-sys/src/mlx_paged_stubs_linux.cpp
```
**Caveat:** upstream's `87a59fc` `build.rs` also adds incidental macOS-path churn (`resolve_build_tool`/`xcrun_find` + `CMAKE_AR`/`CMAKE_RANLIB`/`CMAKE_*_COMPILER_AR` defines). That is benign on a standard Xcode Mac but is extra surface vs our current file. **Do NOT blanket-sync the rest of `crates/mlx-sys/src/`** — upstream's src/ diverged further between #55 and #71 (`mlx_linalg.cpp`/`mlx_transforms.cpp`/`mlx_random.cpp` refactored, MTP/int8 files added); only `build.rs` + the one stub file are wanted.

#### Option A — surgical edits (keeps the macOS path byte-identical)
Seven edits to `crates/mlx-sys/build.rs` + one new file. Each is provably equivalent on macOS (the Mac `yarn build:native` is unaffected). Anchors are current line numbers.

1. **Hoist `target_os`/`target_arch`; add `is_macos`/`build_metal`; gate the Metal panic + metallib compile** (current lines 171-192). Read OS/arch up front; `let is_macos = target_os == "macos"; let build_metal = is_macos && !metal_disabled;`. Replace `if !metal_disabled && !metal_toolchain_available()` → `if build_metal && !metal_toolchain_available()`, and the metallib `if !metal_disabled {…}` → `if build_metal {…}`. *(On Mac `build_metal == !metal_disabled`, so identical.)*
2. **Gate `CMAKE_OSX_ARCHITECTURES` behind macOS; flip `MLX_BUILD_METAL` to `build_metal`** (lines 194-208). Drop `CMAKE_OSX_ARCHITECTURES` out of the unconditional `cfg.define` chain into an `if is_macos { … }`; set `.define("MLX_BUILD_METAL", if build_metal {"ON"} else {"OFF"})`.
3. **Add the Linux/CUDA cmake branch** (after the `if target_os == "macos" {…}` block, before `let dst = cfg.build();`):
   ```rust
   } else if target_os == "linux" {
       let cuda_archs = env::var("MLX_CUDA_ARCHITECTURES").unwrap_or_else(|_| "121a".into());
       cfg.define("MLX_BUILD_CUDA", "ON")
          .define("MLX_BUILD_METAL", "OFF")
          .define("MLX_BUILD_CPU", "ON")
          .define("MLX_CUDA_ARCHITECTURES", &cuda_archs)
          .define("CMAKE_BUILD_TYPE", "Release");
   }
   ```
4. **Gate the Apple-framework links; add the Linux CUDA + LAPACK/BLAS link block** (lines 305-313). Wrap the `framework=Metal/QuartzCore/Foundation/Accelerate/c++` links in `if is_macos { … }` and add:
   ```rust
   } else if target_os == "linux" {
       let cuda_path = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME"))
           .unwrap_or_else(|_| "/usr/local/cuda".into());
       println!("cargo:rustc-link-search=native={cuda_path}/lib64");
       println!("cargo:rustc-link-search=native={cuda_path}/lib64/stubs");
       for l in ["cudart","cublas","cublasLt","cufft","nvrtc","cuda"] {
           println!("cargo:rustc-link-lib=dylib={l}"); }
       println!("cargo:rustc-link-lib=dylib=cudnn");      // see note on split cuDNN
       for l in ["lapack","blas"] { println!("cargo:rustc-link-lib=dylib={l}"); }
       for l in ["stdc++","dl","pthread"] { println!("cargo:rustc-link-lib=dylib={l}"); }
   }
   ```
   *Validated against `mlx/backend/cuda/CMakeLists.txt` (CUDA::cublasLt/cufft/nvrtc/cuda_driver + CUDNN::cudnn_all) and `mlx/CMakeLists.txt` `find_package(LAPACK REQUIRED)`. Upstream links only `lapack`+`blas` — add a bare `lapacke` **only** if the box link reports unresolved `LAPACKE_*` symbols.*
5. **C++ standard split** (lines 318-343). Remove `.std("c++17")` from the unconditional chain; inside `if is_macos { bridge.std("c++17"); bridge.compiler("clang++"); } else { bridge.std("c++20"); }`. *(MLX is built `CMAKE_CXX_STANDARD 20`; its public headers use C++20 `operator==` defaults that GCC enforces strictly.)* Also gate the `metal_cpp` include behind `is_macos`.
6. **Exclude the 3 Metal-only TUs on non-macOS** (the `src/*.cpp` compile loop, lines 347-355). Before the loop, `const METAL_ONLY_TUS: &[&str] = &["mlx_paged_dispatch.cpp","mlx_paged_ops.cpp","mlx_paged_profile.cpp"];` and `if !is_macos && METAL_ONLY_TUS.contains(&file_name.as_str()) { continue; }`. *(These use raw `MTL::` types / `#include mlx/backend/metal/device.h` — they don't compile without Metal.)*
7. **Add the companion stub** `crates/mlx-sys/src/mlx_paged_stubs_linux.cpp` (NEW; entire body under `#if !defined(__APPLE__)`, so empty on Mac). It defines `mlx::core::fast::paged_kv_write / paged_attention / paged_attention_varlen` as runtime-throwing stubs so the Linux link resolves — the excluded TUs (#6) defined them. They are never reached at runtime because compiled-forward registration is gated on `mlx_metal_is_available()` (false on CUDA). Take it verbatim from `87a59fc`:
   ```bash
   git checkout 87a59fc -- crates/mlx-sys/src/mlx_paged_stubs_linux.cpp
   ```
   > **Verify before the first Linux link:** `paged_attention_varlen`'s signature in our `crates/mlx-sys/src/mlx_paged_ops.h` (8 arrays + 2 floats + 5 ints + `KvDtype` + `StreamOrDevice`) matches the stub arg list.

### 3b. The two C++ link blockers + searchsorted routing (fast path)

These make `libmlx` + the addon link and keep searchsorted correct. All three are cfg-dead or Metal-byte-identical on Mac.

**Blocker 1 — `SearchSorted::eval_gpu` undefined on CUDA.** Add to `mlx-node/crates/mlx-sys/mlx/mlx/backend/cuda/primitives.cpp` after `NO_GPU(Cholesky)` (line 31):
```cpp
NO_GPU(SearchSorted)
```
> **VERIFY MACRO ON BOX:** the Metal impl (`backend/metal/searchsorted.cpp:10`) uses the **multi-output** `eval_gpu(const std::vector<array>&, std::vector<array>& outputs)`. `NO_GPU(func)` expands to the *single*-output form; `NO_GPU_MULTI(func)` to the multi-output form. `grep "class SearchSorted" mlx/primitives.h` — if it overrides `std::vector<array>& outputs`, use **`NO_GPU_MULTI(SearchSorted)`**. (Likely `NO_GPU_MULTI`.) This file is in the CUDA backend dir → compiled only when `MLX_BUILD_CUDA=ON` → cfg-dead on Mac.

**Blocker 2 — `get_num_resources` / `get_resource_limit` undefined on CUDA.** These GenMLX-fork getters are declared in `mlx/memory.h:69,73` but defined only in `backend/metal/allocator.cpp:267,270` (excluded when `MLX_BUILD_METAL=OFF`), while `mlx-sys/src/mlx_stream.cpp:254,273` calls them unconditionally. Add a new TU `mlx-node/crates/mlx-sys/src/mlx_resource_stub.cpp` (auto-picked-up by the `build.rs` `src/*.cpp` glob — no build.rs edit needed):
```cpp
// CUDA/CPU resource-count stub. Real defs live in backend/metal/allocator.cpp,
// excluded on a non-Metal build. Guarded so on macOS this TU is empty (no ODR clash).
#ifndef __APPLE__
#include <cstddef>
#include "mlx/memory.h"
namespace mlx::core {
size_t get_num_resources() { return 0; }
size_t get_resource_limit() { return 0; }
}  // namespace mlx::core
#endif  // !__APPLE__
```
Returning 0 is the intended degraded behavior — `buffer-count-limit` (`mlx.cljs:929-931`) already falls back to `499000`, and the membrane's `count-tracking-available?` probe (§4) makes the count-sweep a no-op on CUDA. **No CLJS change required for correctness.**

**Blocker 3 — searchsorted runs on the default (GPU) stream.** With `NO_GPU(SearchSorted)` it would *throw* on CUDA at runtime. Route it to the CPU stream on non-Metal builds. Edit `mlx-node/crates/mlx-sys/src/mlx_array_ops.cpp:936`:
```cpp
#if defined(__APPLE__)
  array result = searchsorted(*sorted, *values, right);                          // GPU on Metal
#else
  array result = searchsorted(*sorted, *values, right, default_stream(Device::cpu));  // CPU on CUDA
#endif
```
> **`__APPLE__` is the correct discriminator** — `MLX_BUILD_METAL` is a CMake option scoped to the `mlx` target (NOT a C++ preprocessor define for the cc-compiled `src/*.cpp`), and `_METAL_AVAILABLE` does not exist in this tree. `__APPLE__` is the established `src/` convention (used 6× in `mlx_paged_profile.cpp`). The 4-arg overload is confirmed (`mlx/ops.h:806`); `default_stream`/`Device` are in scope via `mlx_common.h:283-284`. The `#if defined(__APPLE__)` branch is the original line verbatim → Metal-identical.

**Proper follow-up (do on the box, after a green first boot — keeps searchsorted on GPU):** port `mlx/backend/metal/kernels/searchsorted.h` (two trivial templated binary searches, one thread per value) to `mlx/backend/cuda/searchsorted.cu` implementing `SearchSorted::eval_gpu`, add it to `mlx/backend/cuda/CMakeLists.txt` (model on `scan.cu`/`arange.cu`), and **remove** the `NO_GPU(SearchSorted)` line (3b-1) + revert the `#if/#else` routing (3b-3) to the single line. Kernel sketch:
```cuda
template <typename T>
__global__ void searchsorted_kernel(const T* sorted, const T* values, int32_t* out,
                                    int n, size_t nvals, bool right) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nvals) return;
  T val = values[tid]; int lo = 0, hi = n;
  while (lo < hi) { int mid = (lo + hi) / 2;
    bool go_right = right ? !(val < sorted[mid]) : (sorted[mid] < val);
    if (go_right) lo = mid + 1; else hi = mid; }
  out[tid] = lo;
}
```
Instantiate over the Metal dtype set (float32/16, bfloat16, int/uint 32/16/8); early-return on `values.size()==0`.

### 3c. Build the addon GenMLX actually loads — `genmlx-core` (NOT `@mlx-node/core`)

**Critical:** GenMLX loads `mlx-node/packages/genmlx-core/index.node` (symlinked as `@genmlx/core`; `mlx.cljs:27`). It is **not** produced by `yarn build:native` (that builds the sibling `@mlx-node/core`). It is built by a **bare** napi call (no `--platform`, so the name stays `index.node`), then metallibs are colocated. There is currently no build script — historically (commit `5f3c8ee`) it was built ad-hoc:
```bash
# from mlx-node/
node_modules/.bin/napi build --release \
  --manifest-path crates/genmlx-core/Cargo.toml -o packages/genmlx-core
# then (darwin only) colocate mlx.metallib + paged_attn.metallib into packages/genmlx-core/
```
**Mac-now deliverable (commit it):** add `packages/genmlx-core/build.mjs` that runs the bare napi build and gates the metallib copy to `process.platform === 'darwin'` (no metallib on Linux/CUDA), plus a `"scripts": { "build": "node ./build.mjs" }` in `packages/genmlx-core/package.json`. Then on the box:
```bash
yarn workspace @genmlx/core build      # runs build.mjs; on Linux skips metallib colocation
```
> On Linux, confirm the bare build emits `index.node` (not `genmlx-core.node`); if napi derives the base name from the crate lib, set `napi.binaryName: "index"` or rename post-build.

**Optional (parity for the sibling `@mlx-node/core`, not GenMLX's runtime path):** add the Linux napi triple to `packages/core/package.json` (`aarch64-unknown-linux-gnu` / `x86_64-unknown-linux-gnu` + linux npm sub-packages + optionalDependencies), gate `copyMetallibs()` to darwin in `packages/core/build.ts:65`, and make the `.node` name target-derived (`packages/core/build.ts:72-83` hardcodes `mlx-core.darwin-arm64.node`).

### 3d. Wire + run

`@genmlx/core` is an **in-tree symlinked workspace package** (`node_modules/@genmlx/core → packages/genmlx-core`). There is **no** npm platform optionalDependency to fetch — for the dev flow you just rebuild `index.node` in place (§3c) and it is picked up. (v1's "`npm install` picks the linux optionalDependency for `@genmlx/core`" was misleading.)
```bash
# from /Users/robert/code/genmlx/
node -e "console.log(Object.keys(require('@genmlx/core')).length)"   # addon loads? (link OK)
bun run --bun nbb -e '(require "@genmlx/core")'                      # loads under nbb
```

---

## 4. Membrane (Mac-now — DONE + Metal-verified)

Two edits already applied to `src/genmlx/mlx.cljs` and verified green on this Mac (`clip_contract_test`, `membrane_honesty_test`, `membrane_coverage_test` — coverage pin **212 exports / 48 omissions** unchanged; both edits add only one private CLJS def, zero `@mlx-node/core` exports):

1. **`count-tracking-available?` probe** (after `metal-is-available?`, line 964): `(and (try (metal-is-available?) (catch :default _ false)) (pos? buffer-count-limit))`. True on Metal (so every count-aware path is byte-identical), false on CUDA.
2. **`buffer-count-pressure?` short-circuit** (`if-not count-tracking-available?` instead of `if-not (metal-is-available?)`): skips the `get-num-resources` FFI hop on CUDA. On Metal both predicates were already false → identical else-branch.

These make the genmlx-5ucd count-sweep CUDA-neutral without a behavior change on Metal. *(Optional, deferred: honest `metal-device-info` introspection for non-Metal — diagnostic only, could perturb a key-set assertion, so left as a sketch in the bean.)*

---

## 5. Validation on the box (corrected chain — run IN ORDER, each gates the next)

`bun run --bun nbb test/genmlx/<f>.cljs`. Expect `0 failures, 0 errors` (cljs.test) / `N passed, 0 failed` (println files) unless a pin is given.

1. **Addon loads** (§3d). Proves the CUDA `.node` linked (no undefined `SearchSorted::eval_gpu`) and the membrane requires cleanly. On CUDA expect `metal-is-available? => false`, `get-num-resources => 0` (→ `count-tracking-available?` false → count-sweep no-op). *If this fails, it's a build problem — fix §3a/§3b. This is THE gate.*
2. **`membrane_coverage_test`** — surface-drift guard; should be green/unchanged (the two new C++ symbols add **no** JS exports). Pin: 212 exports / 48 omissions.
3. **`exact_test`** — first GFI routing gate: exercises **categorical sampling (native keyCategorical) + the enumerate/exact engine**. *(Corrected: this does NOT exercise cholesky/inverse — see step 6 for the real linalg gate.)*
4. **`gradient_fd_test`** then **`score_gradient_test`** — **THE most important correctness signal.** They differentiate through beta/gamma/inv-gamma log-probs whose gradients require `lgamma`/`digamma` — i.e. our CUDA unary `.cu` kernels. FD-vs-analytic agreement here = **our `.cu` kernels are numerically correct on real NVIDIA HW.**
5. **`clip_contract_test`** — `Either<&MxArray,f64>` bounds contract on the new binary (6 tests / 7 assertions).
6. **`level0_certification_test`** — MUST be **68/68**. Then **`l3_5_multivariate_test`** (and/or `genjax_compat_test`) — the **real cholesky/inverse/MVN gate** (multivariate-normal conjugacy → cholesky/cholesky-inv/linalg-inv on the CPU stream). Green = linalg CPU-pinning works on CUDA.
7. **`vectorized_test`** then **`compiled_smc_test`** — the **searchsorted canary**: `vectorized.cljs` + `compiled_smc.cljs` systematic resampling use `mx/searchsorted` on a cumsum. With the fast path (3b) they run on the CPU stream; if the `.cu` kernel was ported they stay on GPU.

End state for v1 (core, no LLM): 1-6 green; 7 green once searchsorted is CPU-routed or ported. `@mlx-node/lm` (paged-attn) is **out of scope** on CUDA.

---

## 6. Mac-now vs box-only

**Done / doable on the Mac NOW (this session):**
- ✅ **The membrane edits** (§4) — applied + Metal-verified green.
- ✅ **This runbook** — every edit verified against the live tree + `upstream`.
- The §3a/§3b/§3c **source edits** can be *written* on the Mac, but they are `cfg`/`#ifdef`-dead on Metal (Rust *removes* `#[cfg(target_os="linux")]` and the CUDA TU before type-check), so a Mac rebuild does **not** validate the Linux code. They are documented here rather than committed untested. (`build.rs` Option A's macOS-path edits are provably equivalent and *can* be Mac-rebuild-verified if desired.)

**Requires the machine:**
- CUDA compilation of MLX + the 4 special-fn kernels (`nvcc`, CCCL/CUTLASS fetch).
- The native CUDA link (proves §3a/§3b resolve — incl. whether more fork-added Metal-only symbols surface beyond the two resource fns; do a full `nm`/link pass and extend `mlx_resource_stub.cpp` if so).
- The `NO_GPU` vs `NO_GPU_MULTI` choice for `SearchSorted` (grep `class SearchSorted` in `primitives.h`).
- Producing the Linux `.node` (napi CUDA is **not cross-compilable from macOS**).
- Every §5 validation item (real GPU execution) + the searchsorted GPU-vs-CPU perf decision.

---

## 7. Risks + fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| **searchsorted LINK failure** | Certain without §3b-1 | `NO_GPU(SearchSorted)`/`NO_GPU_MULTI` + CPU-route (§3b-3). Proper: port `searchsorted.cu`. |
| **`NO_GPU` vs `NO_GPU_MULTI` mismatch** | Possible | grep `class SearchSorted` in `primitives.h`; metal impl is multi-output → likely `NO_GPU_MULTI`. |
| **resource-symbol LINK failure** | Certain without §3b-2 | `mlx_resource_stub.cpp` → 0. Count-sweep degrades to no-op automatically. |
| **More fork-only Metal symbols undefined on Linux** | Possible | `nm`/link pass on the box; extend the stub TU. |
| **build.rs has no CUDA path** | Certain | `git checkout 87a59fc -- build.rs mlx_paged_stubs_linux.cpp` (Option B) or the 7 surgical edits (Option A). |
| **GCC rejects MLX C++20 headers via the cc bridge** | Possible | C++17(mac)/C++20(linux) split (§3a-5). |
| **CUDA 13.1 hard-rejected** | If box ships 13.1 | Install 12.4–12.9 or ≥13.2; cuDNN 9 mandatory. |
| **`MLX_CUDA_ARCHITECTURES=121a` wrong for the GPU** | If not GB10 | Export the right arch (90a/89/80) before building. |
| **split-cuDNN: `-lcudnn` unresolved** | Possible | Add `cudnn_graph`/`cudnn_ops`/`cudnn_engines_*` per the cmake link line. |
| **x86_64 not upstream-validated** | If box is x86_64 | MLX C++ supports it; expect more debugging. |
| **paged-attn / LLM has no CUDA kernels** | Certain | Scope `@mlx-node/lm` out of v1. Stubs throw if reached (they are not — gated on `mlx_metal_is_available()`). |
| **genmlx-5ucd buffer-count net absent on CUDA** | By design | No ~499000 wall on CUDA; byte-pressure heuristics still run on real `get-active-memory`. Watch memory during §5. |

**Bottom line:** the port is small and upstream-precedented. The real work is (a) the `build.rs` Linux branch + companion stub (§3a — upstream-tested as `87a59fc`), (b) three small C++ edits for searchsorted + the resource symbols (§3b), (c) a committed `genmlx-core` build script (§3c). Linalg "CPU-stream routing" is **already done** in the shim — ignore v1's framing. Target **aarch64 Linux, CUDA 12.x, core-inference-first, LLM deferred**.

---

## 8. Key verified file:line anchors

`mlx-sys/src/mlx_linalg.cpp:6-9,20,29,40,48,56,64,78,91,104,113` (all linalg CPU-pinned) · `mlx-sys/src/mlx_array_ops.cpp:930-939` (searchsorted, default GPU stream) · `mlx/backend/cuda/primitives.cpp:27-37` (`NO_GPU` list, no SearchSorted) · `mlx/backend/metal/searchsorted.cpp:10` (only `eval_gpu`, multi-output) · `mlx/memory.h:69,73` + `mlx/backend/metal/allocator.cpp:267,270` (resource symbols Metal-only) · `mlx-sys/src/mlx_stream.cpp:254,273` (unconditional FFI calls) · `mlx/backend/cuda/unary/CMakeLists.txt:17-20` + `cuda/device/unary_ops.cuh:134-253` (4 `.cu` special-fn kernels, real math) · `mlx-sys/build.rs:171-355` (Metal/xcrun-only; cc bridge at 318-355, cmake at 194-225; upstream port = `87a59fc`) · `mlx/ops.h:806` (4-arg searchsorted) · `mlx_common.h:283-284` (`default_stream`/`Device` using-decls) · `packages/genmlx-core/{package.json,index.node}` (the addon GenMLX loads; bare-napi build, commit `5f3c8ee`) · `packages/core/build.ts:65,72-83` (darwin-arm64 hardcode + metallib throw) · `src/genmlx/mlx.cljs:27,382,929-931,964,1061` (require, searchsorted, buffer-count fallback, membrane probes) · `src/genmlx/{vectorized.cljs:31,inference/compiled_smc.cljs:50}` (searchsorted hot-paths). Sources: [mlx-node](https://github.com/mlx-node/mlx-node) commit `87a59fc`, [MLX CUDA #2422](https://github.com/ml-explore/mlx/discussions/2422), [MLX build docs](https://ml-explore.github.io/mlx/build/html/install.html).
