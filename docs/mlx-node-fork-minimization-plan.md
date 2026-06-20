# mlx-node Fork-Minimization Plan (validated)

> **Bean:** `genmlx-nldo` (spike `genmlx-53lu` — GREEN). Part of milestone `genmlx-nv1t`.
> **Status:** architecture **empirically validated**; the remaining port is mechanical.
> **Date:** 2026-06-20.

## Goal

Shrink the mlx-node fork toward **zero** so GenMLX rides on stock mlx-node, while keeping the
PPL-specific native surface in a **GenMLX-owned crate**. The end-state is: stock MLX + stock
mlx-node + one `@genmlx/core` superset `.node` addon that GenMLX owns.

## What the spike proved (genmlx-53lu)

A separate crate (`crate-type = ["cdylib"]`) depending on **stock `mlx-core` (rlib)** compiles into
**one superset addon** (220 exports). Decisive results:

- **mlx-core's `#[napi]` registration ctors survive the rlib→cdylib link with NO `-force_load`**
  (DCE survival = true): `MxArray`, `memoryStats`, `setCacheLimit`, `randomKey`, `zeros` all present
  alongside the new `genmlxSpike*` fns.
- **Round-trip OK:** an `MxArray` minted by an mlx-core op flowed through a *new* genmlx `#[napi]`
  fn in the same addon — one NAPI type registry, one MLX runtime (no "Failed to recover MxArray").
- **Two separate addons remain DEFINITIVELY BROKEN** (proven from source): per-addon NAPI type-tag
  registries + per-dylib MLX Meyers-singleton statics (two Metal `Device`/`Allocator`). So the
  superset addon is the *only* viable shape — which is exactly what ships today (genmlx.rs is
  already compiled *into* mlx-core via `pub mod genmlx;` at `mlx-core/src/lib.rs:32`).

## The export-surface contract (validated against source)

What the genmlx NAPI surface touches in mlx-core / mlx-sys, and what each needs:

### Tier 1 — mlx-core Rust items
| Item | Current visibility | Action |
|---|---|---|
| `MxArray::scalar_float` (`array/creation.rs:109`) | `pub` | none ✓ |
| `MxArray::as_raw_ptr` (`array/mod.rs:108`) | `pub` | use instead of `handle.0` ✓ |
| `MxArray` (`Clone`, `array/mod.rs:113`) | `pub` | use `arr.clone()` instead of `arr.handle.clone()` ✓ |
| `nn::Activations::silu` (`nn/activations.rs:16`) | `pub` | none ✓ |
| `utils::functional::rms_norm_functional` (`utils/functional.rs:77`) | `pub` | none ✓ |
| `array::scaled_dot_product_attention` (`array/attention.rs:25`) | `pub` | none ✓ |
| `utils::safetensors::load_safetensors_lazy` (`utils/safetensors.rs:685`) | `pub` | none ✓ |
| `array::take_last_native_error` (`array/handle.rs:9`) | **`pub(crate)`** | **→ `pub`** |
| `MxArray::scalar_float_like` (`array/creation.rs:116`) | **`pub(crate)`** | **→ `pub`** (if used) |

**Net Tier-1 fork change: ~2 add-only `pub` lines.** Cleanly upstreamable as a "let downstream
crates build on `MxArray`" PR. The `MxArray.handle` field (`pub(crate)`) does **not** need to leak
— `as_raw_ptr()` + `MxArray: Clone` cover the two uses (`genmlx.rs:21`, `:515`).

### Tier 2/3 — mlx-sys C++ shims + FFI
The keyed-PRNG / linalg / transform shims live in **separate, movable files**
(`mlx-sys/src/mlx_random.cpp`, `mlx_linalg.cpp`, `mlx_transforms.cpp`) and use the **MLX C++ API**
(`#include "mlx/random.h"`, `mlx::core::random::`, plus `mlx_common.h` and — transitively on macOS —
metal_cpp headers). Their `extern "C"` symbols (`mlx_random_key`, `mlx_random_normal_key`, …) are
declared in `mlx-sys/src/lib.rs:587-620` and compiled into `libmlx_ffi.a` by `mlx-sys/build.rs:318-357`.

mlx-sys does **not** export `cargo:include` (no `links` key), so a downstream crate cannot replicate
the MLX C++ include environment to compile its own shims. Two options:

- **Option A (RECOMMENDED): keep the 3 `.cpp` shims + their FFI decls in mlx-sys.** genmlx-core stays
  **pure Rust** (calls the mlx-sys FFI through the normal rlib dependency — the same linkage the spike
  already exercised when `randomKey` surfaced and was callable). Residual mlx-sys fork = 3 C++ files +
  ~20 `extern "C"` decls — small, self-contained, and upstreamable to mlx-node as "extended FFI for
  downstream PPL/array consumers."
- **Option B: move the C++ shims to genmlx-core** with its own `cc::Build`. Requires mlx-sys to add a
  `links` key + emit `cargo:include` for the MLX header dirs (incl. the *generated* metal_cpp include
  that only exists post-cmake). ~3 lines in mlx-sys, but the downstream C++ build must mirror
  `mlx-sys/build.rs:318-357` (c++17, `-DMLX_STATIC`, `mlx_dir`/`mlx_dir/mlx`/generated includes).
  More moving parts; defer unless Option A's C++ residual is rejected upstream.

**Pick Option A.** It keeps genmlx-core pure Rust and needs no downstream C++ toolchain replication.

## Target fork footprint after migration

| Layer | Before (current fork) | After (Option A) |
|---|---|---|
| **mlx-core** | `genmlx.rs` (697) + `transforms.rs` (361) + `memory_napi.rs` + keyed-PRNG `array/random.rs` (+213) | **2 `pub` lines** |
| **mlx-sys** | 3 C++ shims + FFI decls | 3 C++ shims + FFI decls (unchanged) |
| **GenMLX repo (new)** | — | `genmlx-core` crate: all the Rust NAPI + keyed-PRNG wrappers |

The ~1271 lines of genmlx Rust **leave the mlx-node tree entirely.** If the two residual PRs (Tier-1
`pub` + Tier-2/3 C++ FFI) are accepted upstream, **the mlx-node fork goes to zero** and genmlx-core
consumes fully stock mlx-node.

## Mechanical port steps (the real work, post-approval)

1. **mlx-core:** make `take_last_native_error` + `scalar_float_like` `pub` (2 lines). [PR to mlx-node]
2. **genmlx-core crate in the GenMLX repo** (`crate-type = ["cdylib"]`), depends on stock `mlx-core` +
   `mlx-sys` + `napi`/`napi-derive` pinned to the **exact** mlx-core napi version (a version skew
   silently reintroduces the two-registry failure *inside one addon*). Port `genmlx.rs`,
   `transforms.rs`, `memory_napi.rs`, and the keyed-PRNG Rust wrappers; rewrite `handle.0`→`as_raw_ptr()`,
   `handle.clone()`→`clone()`.
3. **Build** the superset `.node` via `@napi-rs/cli` (manifest = genmlx-core), copy `mlx.metallib`
   colocated. Validate the full surface (≈ the 212 wrapped exports + genmlx surface) loads.
4. **Repoint the membrane:** `src/genmlx/mlx.cljs` `(js/require "@genmlx/core")` instead of
   `"@mlx-node/core"`. Run the contract guards (`exact_test`, `gradient_fd_test`, `clip_contract_test`,
   `membrane_coverage_test`, `level0_certification_test`) — they exist to catch exactly this.
5. **Update `package.json`** to build/depend on the genmlx-core addon.

## Upstream PRs that zero the fork

- **PR 1 (mlx-core):** `pub` the 2 accessors — "expose `MxArray` error-drain + scalar-like for
  downstream crates." Tiny, add-only.
- **PR 2 (mlx-sys):** keep the keyed-PRNG/linalg/transform C++ shims + FFI as a documented
  "extended array FFI" — or, if Option B, the `links` + `cargo:include` export.

## Status

- ✅ Superset addon mechanism — **empirically proven** (spike, no `-force_load` needed).
- ✅ Two-addon failure — **proven from source**.
- ✅ Tier-1 Rust contract — **mapped to file:line** (~2 pub lines).
- ✅ Tier-2/3 C++/FFI linkage from a downstream Rust crate — **already exercised** by the spike
  (`randomKey` callable downstream).
- ▶ Remaining: the mechanical port + membrane repoint + contract-guard run (a reviewable
  implementation task; spec-first not required — each step is empirically gated by the build + guards).
