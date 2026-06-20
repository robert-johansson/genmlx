# Taking the mlx-node fork to zero — tiered plan

> **Bean:** `genmlx-nldo`. Builds on the merged genmlx-core migration (PR #143).
> **Goal (refined 2026-06-20):** make mlx-node's Rust/TS source **stock upstream above the
> mlx-sys substrate**. We deliberately KEEP `mlx-sys` as ours — it vendors our MLX fork
> (submodule pointer) and hosts the 3 C++ shims. So the target is: **`mlx-core` + all packages
> stock; `mlx-sys` = our thin substrate; `genmlx-core` = our crate in the GenMLX repo; MLX = our
> maintained fork.** This DROPS Tier C (C++-shim relocation) — the hardest/riskiest tier.
> **Date:** 2026-06-20. Every file:line below verified against source.

## Where we are now

After PR #143 the mlx-node fork's genmlx footprint is:
- **mlx-core**: 2 add-only `pub` lines (`take_last_native_error`), `pub mod genmlx;` disabled, **plus still-present** `transforms.rs`, `memory_napi.rs`, and the keyed-PRNG block in `array/random.rs`.
- **mlx-sys**: 3 C++ shims (`mlx_random.cpp`, `mlx_linalg.cpp`, `mlx_transforms.cpp`) + their `extern "C"` FFI decls.
- **crates/genmlx-core**: the GenMLX surface crate (lives *inside* mlx-node).
- (Plus the original 34-commit divergence — mostly pruning of LFM2/experiment code. "Going stock" means tracking upstream as-is and dropping the pruning.)

## The four tiers

### Tier A — relocate `transforms.rs` + `memory_napi.rs` → genmlx-core  ·  MECHANICAL
Same pattern as the genmlx.rs port. Both are free-function `#[napi]` modules.
- **`memory_napi.rs`**: calls `crate::array::memory::*` — all already `pub` (re-exported in `array/mod.rs:27`). Rewrite `crate::` → `mlx_core::`, `a.handle.0` → `a.as_raw_ptr()` (`:131`). No new pub lines.
- **`transforms.rs`**: constructs `MxArray` via `MxArray::from_handle` (`pub(crate)`, `array/mod.rs:94`) at `:38,:170`, and reads `handle.0`/`handle.clone()` at `:41,:89,:92,:94,:134,:144`. Needs **`from_handle` → `pub`** (1 add-only line); rewrite `handle.clone()`→`clone()`, `handle.0`→`as_raw_ptr()`, `crate::`→`mlx_core::`.
- Delete both files from mlx-core; remove `pub mod transforms; pub mod memory_napi;` from `lib.rs:31,33`. Removal-safe: nothing in mlx-core references `crate::transforms`/`crate::memory_napi` (verified).
- **Build + guard-validate** (the proven loop). Risk: low.

### Tier B — keyed-PRNG: inline the FFI into genmlx-core's free fns  ·  MODERATE
`array/random.rs` holds the keyed-PRNG as **`#[napi]` methods on `MxArray`** (`:71–275`: `random_key`, `random_split`, `random_split_n`, `key_uniform`, `key_normal`, `key_bernoulli`, `key_categorical`, `key_randint`, `key_gumbel`, `key_laplace`, `key_truncated_normal`, `key_multivariate_normal`). genmlx-core's free fns (`genmlx.rs:560–640`) currently **delegate** to those methods (`key.key_uniform(...)`). The methods are thin wrappers over `sys::mlx_random_*_key` + `from_handle`.
- **Rewrite** each genmlx-core free fn to call `sys::mlx_random_*_key` **directly** (copy the method bodies from `array/random.rs:71–275`), constructing results via `MxArray::from_handle` (already `pub` from Tier A). genmlx-core already depends on `mlx-sys`.
- **Delete** the keyed-PRNG methods (`:71–275`) from `array/random.rs`, restoring it to stock (only the upstream `random_uniform`/`random_normal`/`random_bernoulli`/`randint`/`categorical` methods remain, `:8–59`).
- Removal-safe: nothing outside genmlx.rs calls the `.key_*()` methods (the membrane uses the free fns — `mlx/random.cljs:23,111`).
- **Build + guard-validate** — keyed-PRNG round-trip + `gradient_fd` exercise this. Risk: moderate (FFI signature transcription; the compiler + guards catch it).

**After A+B: mlx-core is stock** except the 2 `pub` lines (`take_last_native_error` + `from_handle`) — both add-only, upstreamable, or avoidable (reimplement `take_last_native_error` via the mlx-sys error FFI directly in genmlx-core).

### Tier C — relocate the mlx-sys C++ shims → genmlx-core  ·  DROPPED (by the refined target)

> **Not pursued.** The refined target keeps `mlx-sys` as our owned substrate (it vendors our MLX
> fork anyway), so the C++ shims stay where they are. This removes the hardest tier entirely.
> Documented below only as the path *if* a fully-stock mlx-sys were ever wanted.
The keyed-PRNG/linalg/transform FFI is implemented in `mlx-sys/src/mlx_{random,linalg,transforms}.cpp` (+ `mlx_common.h`), compiled into `libmlx_ffi.a` by `mlx-sys/build.rs:318–357` (c++17, `-DMLX_STATIC`, includes: `mlx_dir`, `mlx_dir/mlx`, the **generated** include dir with metal_cpp, `src/`). For mlx-sys to be stock, these must move to genmlx-core with its **own** `cc::Build`.
- **The hard part:** genmlx-core's `build.rs` must compile C++ that `#include "mlx/random.h"` (and transitively metal_cpp on macOS) — which needs MLX's header tree, **including the cmake-generated include dir that only exists after the MLX build**. mlx-sys does not export these (no `links` key, no `cargo:include`).
- **Two options:**
  - **C1 (recommended): a ~3-line mlx-sys change** — add `links = "mlx_sys"` + `println!("cargo:include={...}")` for the MLX header dirs + generated include. genmlx-core's build.rs reads `DEP_MLX_SYS_INCLUDE`. *This is still a (tiny) mlx-sys patch* — so true mlx-sys-zero needs it **upstreamed**, or:
  - **C2:** genmlx-core references the vendored MLX headers by relative path (`../mlx-node/crates/mlx-sys/mlx/mlx` + the build-OUT_DIR generated include) — no mlx-sys change, but fragile (hardcodes submodule layout + the post-cmake generated path).
- **Spike before committing:** a throwaway crate that compiles ONE shim (`mlx_random.cpp`) downstream against the located headers and links it. If the metal_cpp/generated-include resolution works, Tier C is feasible; if it hits header hell, keep the shims in mlx-sys (Option A from the original plan — a small, upstreamable residual).
- Risk: **high**. This is the tier most likely to stall.

### Tier D — lift genmlx-core out of mlx-node → the GenMLX repo  ·  RISKY (napi pinning)
Move `crates/genmlx-core` → `/Users/robert/code/genmlx/genmlx-core` (path-dep on `../mlx-node/crates/{mlx-core,mlx-sys}`). Its own `Cargo.lock` reintroduces **napi-version-skew risk** (a patch-bump → two NAPI registries in one addon → "Failed to recover MxArray"). Mitigation: `=`-pin napi/napi-derive to mlx-core's exact resolved versions + a CI assert that the resolved napi == mlx-core's. Add a `build.ts`/`napi build` step + the `@genmlx/core` `file:` dep (already in `package.json`).
- Risk: medium — the napi pin must be airtight.

## End state & sequencing

| After | state under the refined target |
|---|---|
| A+B | **mlx-core stock** (− 1 upstreamable pub line: `from_handle`; `take_last_native_error` dropped via mlx-sys FFI) |
| +D  | genmlx-core **in the GenMLX repo** → mlx-node tree is **stock above mlx-sys** |
| (steady state) | ours = MLX fork + mlx-sys substrate (MLX pointer + 3 C++ shims) + genmlx-core (our repo); everything else stock |

**Recommended order:** A → B (validate mlx-core stock via the guards) → D. All proven/low-moderate risk; Tier C is dropped.

**Note:** even at fork-zero, mlx-node's vendored **MLX submodule still points to our patched MLX** (the 9 commits). Zeroing *that* is the separate `docs/mlx-math-reimplementation-spec.md`. So "stock mlx-node + stock MLX" = (this plan) + (the MLX-math spec).
