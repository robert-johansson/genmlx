# Fork-to-zero — Horizon 1 milestone (grounded 2026-06-21)

> **Status:** SPEC for discussion. Supersedes the status section of
> `docs/fork-to-zero-plan.md` (which is stale — see below). Bean: `genmlx-cigu`.
> **Goal of Horizon 1:** shrink the mlx-node fork so the only GenMLX-owned thing
> left *inside the mlx-node tree* is the `mlx-sys` substrate (the MLX-fork pointer
> + 3 C++ shims). Everything else stock; `genmlx-core` in the GenMLX repo.
> **Out of scope (Horizon 2):** upstreaming the MLX C++ patches to dissolve the
> fork entirely.

## What grounding found (vs the stale plan)

Verified against live source (merge-base `35559218`), 2026-06-21:

- **Tiers A + B are already DONE** (commit `f86a344`, 2026-06-20): `transforms.rs`
  + `memory_napi.rs` moved `mlx-core → genmlx-core`; keyed-PRNG **inlined** as
  `genmlx-core/src/genmlx.rs` free fns calling `mlx_sys::mlx_random_*_key`
  directly (uses `dt as i32`, so `DType::code()` did **not** need to go public);
  `array/random.rs` is back to 64 lines / stock methods only. PR #143 already
  moved `genmlx.rs`. The plan's "Where we are now" describes the pre-A+B state.
- **The plan's "After A+B: mlx-core is stock" is FALSE.** Beyond the relocated
  free-fn modules, mlx-core still carries **~33 GenMLX `#[napi]` methods defined
  as `impl MxArray`** — which the **Rust orphan rule forbids moving** to
  genmlx-core *as methods* (you can't add inherent methods to a foreign type).
  The plan never names this; it is the real remaining blocker for a stock
  mlx-core crate.

## Current residual GenMLX delta in `mlx-core` (vs merge-base)

1. **2 pub promotions** — `from_handle` (`array/mod.rs:97`, was `pub(crate)`) and
   `take_last_native_error` (`array/mod.rs:24` re-export + new `pub fn` in
   `handle.rs:9`, the genmlx-5ucd error surfacing). Add-only, upstreamable.
2. **~33 orphan-rule-bound `#[napi] impl MxArray` methods** (all "GenMLX
   consolidation", all new vs merge-base):
   - `autograd.rs:518+` — `value_and_grad` / `valueAndGrad` / `computeGradients`
   - `array/reduction.rs:208+` — `all` / `any` / `topk` / `logcumsumexp`
   - `array/ops.rs:390+` — ~27 ops: `sigmoid`, `softmax`, `erfinv`, `lgamma`,
     `digamma`, `expm1`, `bessel_i0e/i1e`, `logaddexp`, `nan_to_num`, `flatten`,
     `inner`, `outer`, `diag`, `einsum`, `trace`, `cholesky`, `linalg_solve`,
     `solve_triangular`, `linalg_inv`, `tri_inv`, `cholesky_inv`, `qr`, `svd`,
     `eigh`, `eigvalsh`, `linalg_norm`.
   These export through `@genmlx/core` via the rlib link (spike genmlx-53lu).
3. `genmlx-core` still lives **inside** `mlx-node/crates/` (Tier D not done).
4. `mlx-sys` substrate (MLX-fork submodule pointer `robert-johansson/mlx` @
   `49503b65` + 3 C++ shims) — **stays** (the irreducible Horizon-2 knot).

## The two real workstreams

### WS1 — Tier D: lift `genmlx-core` out of `mlx-node` → the GenMLX repo

The decision that defines this WS is **how** to relocate, because napi-skew
safety today is *implicit* (shared workspace `Cargo.lock` → one `napi` 3.8.4 /
`napi-derive` 3.5.3; verified one `[[package]] napi` in the lock):

- **Option D1 — out-of-tree workspace member (recommended).** Move the crate dir
  to `genmlx/genmlx-core` but keep it a member of the mlx-node workspace
  (`members = ["crates/*", "../genmlx-core"]` or equivalent). Keeps the **single
  shared lock** → **napi-skew cannot occur**, and re-uses the workspace
  `[profile.release]` (`lto=true`, `codegen-units=1` — load-bearing for the
  single-registry rlib link). Achieves the Horizon-1 goal (genmlx-core is no
  longer *in the mlx-node tree*) at near-zero risk.
- **Option D2 — fully standalone crate (its own lock).** The plan's original
  framing. Reintroduces the skew risk: a `cargo update` in either repo can land a
  different `napi` patch → two NAPI registries → "Failed to recover MxArray." The
  named mitigation (`=`-pin + CI lock-diff) is **not yet built**, and an `=`-pin
  on `napi`/`napi-derive` alone misses `napi-sys` / `napi-derive-backend` — the
  assert must compare the full `napi*` set or vendor a shared lock. Also must
  re-declare `[profile.release]`.

Mechanical items either way:
- **Layout:** mlx-node is **nested** (`genmlx/mlx-node`), not a sibling. So
  path-deps from `genmlx/genmlx-core` are `./mlx-node/crates/{mlx-core,mlx-sys}`
  — **the plan's `../mlx-node/...` is wrong** for this layout.
- `packages/genmlx-core/build.mjs`: re-point the `--manifest-path` + the metallib
  colocation walk (mlx-sys's metallibs land under `mlx-node/target`).
- `@genmlx/core` `file:` dep already in `package.json`; the npm package
  (`packages/genmlx-core`, package.json + build.mjs) can stay in mlx-node as a
  thin builder or also relocate — decide.

**Done means:** `genmlx-core` source lives in the GenMLX repo; `@genmlx/core`
builds via the (re-pointed) `build.mjs`; exactly one resolved `napi*` set; all
gates green (below).

### WS2 — the orphan-rule residual: make `mlx-core` *actually* stock?

This is the plan's blind spot and the real scope question. The ~33 `impl MxArray`
methods block a stock mlx-core. Options:

- **Option W-A — accept the residual (smaller).** Leave the ~33 methods in
  mlx-core. Horizon-1 then means "mlx-node tree is stock above `mlx-sys` *plus a
  GenMLX consolidation layer in mlx-core*," with genmlx-core lifted out (WS1).
  Honest, low-risk; mlx-core is "stock + ~33 add-only methods + 2 pub lines."
- **Option W-B — convert the ~33 methods to `genmlx-core` free fns (bigger).**
  Rewrite each `impl MxArray { #[napi] fn op(&self,..) }` as a `genmlx-core`
  `#[napi] fn op(a:&MxArray,..)` (the keyed-PRNG pattern), delete the methods from
  mlx-core → **truly stock mlx-core** (minus the 2 pub lines). Cost: the rewrite +
  updating every membrane call site in `mlx.cljs` (method `(.cholesky a)` → free
  fn) + re-pinning the `membrane_coverage` 214-export surface. Feasibility varies:
  the linalg/special-fn ops mostly call `mlx_sys`/`mlx-core` ops on `&MxArray`
  (mechanical); `value_and_grad`/`computeGradients` take closures and are the
  hard part.
- **Option W-C — upstream them (Horizon 2).** They're GenMLX-specific math; real
  upstreaming targets `ml-explore/mlx`, not mlx-node. Defer.

### WS3 — the 2 pub promotions + doc/bean hygiene (small)

- `from_handle` / `take_last_native_error`: upstream, or avoid (`take_last_native_error`
  is reimplementable via the mlx-sys error FFI inside genmlx-core).
- Update `docs/fork-to-zero-plan.md` status (A+B done; add the orphan-rule
  residual), the `membrane-coverage` matrix if the surface shifts, and bean
  `genmlx-cigu`.

## Gates (run after every native change; CLAUDE.md post-rebuild)

Rebuild: `node packages/genmlx-core/build.mjs` (NOT `yarn build:native`). Then:
membrane contract guards (`exact` / `gradient_fd` / `score_gradient` /
`clip_contract` / `membrane_coverage` — the 214-export pin) + `level0_certification`
+ `prng_key` / `prng_hygiene` (keyed-PRNG) + `vmap` / `vmap_property` /
`compile_fn` (relocated transforms).

## Risks

- **napi-skew** (D2 only) — the central risk; D1 sidesteps it.
- **Nested-vs-sibling path-deps** — get the relative path right (`./mlx-node/...`).
- **`[profile.release]` / LTO** — must carry over (D2) or be inherited (D1); the
  cross-rlib LTO is part of why one NAPI registry works.
- **WS2-B membrane churn** — converting methods → free fns touches `mlx.cljs` and
  the 214-export surface pin broadly.
- **`mlx-sys` is irreducible** — Horizon 1 cannot remove it; only Horizon-2
  upstreaming can.

## Sequencing

`WS1 (Tier D, Option D1)` → `WS3 (hygiene)` first — they deliver "genmlx-core out
of the mlx-node tree" at low risk. `WS2` is the larger, optional push to a truly
stock mlx-core crate; decide whether it's in this milestone or its own.

## GROUNDED FLOOR (2026-06-21 feasibility study — supersedes the estimates above)

A 4-investigator pass over live source classified every mlx-node delta
(merge-base `35559218..HEAD`, 41 commits, +6790/−1590, 54 files). Result:

**The single irreducible mlx-core pub patch is `from_handle` (1 line).** No public
ctor builds an `MxArray` from a raw FFI handle; genmlx-core needs it at 20+ sites.
Everything else genmlx-core needs is already stock-public (`as_raw_ptr`, the
generic `value_and_grad`/`compute_gradients` free fns, …) or reachable via pub
`mlx-sys` FFI. `take_last_native_error` pub is **avoidable** (inline 12 lines).

**The ~35 `impl MxArray` `#[napi]` methods are extractable as FREE FNS** (W-B):
bodies are mechanically `sys::mlx_array_<op>(arr.as_raw_ptr())` + `from_handle`;
no mlx-core private internals. The orphan rule only forbids them *as methods* — as
genmlx-core free fns they move out cleanly (cost: re-bind call sites in `mlx.cljs`
+ re-pin the 214-export surface). The `value_and_grad`/`computeGradients` JS pair
is the one non-trivial conversion, but genmlx-core's `transforms.rs` already proves
the exact JS-closure→MxArray recipe.

**THE REAL OBSTACLE (correction to the plan): mlx-sys is NOT "3 shims + a
pointer."** Its dominant GenMLX delta is **cross-cutting `MLX_GUARD`
exception-hardening woven IN-PLACE through ~144 existing upstream C++ functions**
(+~600 lines) + ~24 generic ops interleaved into those files. **In-place edits to
upstream functions cannot be extracted to a separate library at all.** They are
irreducible-as-fork-patch *unless upstreamed*. The 3 named shims (keyed-PRNG,
vmap/compile-apply) *are* extractable, but only behind a minimal generic `build.rs`
change (C1: export the MLX include dirs) and are coupled to the guard macros in
`mlx_common.h`.

**Standalone genmlx-core needs ZERO mlx-node workspace edit** (root `Cargo.toml`
is byte-identical to upstream; `members = ["crates/*"]` is a stock glob — physically
moving the crate dir removes it automatically). The single-NAPI-registry guarantee
(today free via the shared lock) is re-established by: commit genmlx-core's own
`Cargo.lock` + `=`-pin the full ABI set (`napi`, `napi-sys`, `napi-derive`,
`napi-derive-backend`, `napi-build`) + a CI `cargo tree` diff vs mlx-core; plus
re-declare `[profile.release]` (lto) and point `build.mjs` at `mlx-node/target`.

### The two levers (this is the strategic shape)

- **EXTRACT** (fast, fully in our control): move genmlx-core out + W-B free-fn
  conversion → **mlx-core drops to 1 patch line (`from_handle`)** and the entire
  NAPI addon lives in the GenMLX repo.
- **UPSTREAM** (slower, review-gated, but the only way to shrink the substrate):
  PR the `MLX_GUARD` hardening, the standard linalg/special-fn ops, the sharded
  loader, the GRPO reference-model KL, and C1 into mlx-node. As they land, the big
  in-place mlx-sys C++ patch dissolves AND a chunk of the orphan-bound ops become
  stock mlx-core.

### Realistic floors

| State | Residual mlx-node fork |
|---|---|
| **After EXTRACT only** | `from_handle` (1 line) in mlx-core · the full `mlx-sys` substrate (the ~144-fn `MLX_GUARD` in-place hardening + 3 shims + the MLX submodule pointer) |
| **After EXTRACT + UPSTREAM the generics** | ≈ just the MLX submodule pointer (+ maybe `from_handle` if not upstreamed) |
| **+ Horizon 2 (upstream the 9 MLX commits)** | zero — fork dissolves |

**Bottom line:** zero is unreachable by extraction alone — the `mlx-sys` guard
hardening and the MLX pointer are in-place/substrate. But EXTRACT (fast) gets
mlx-core to ~1 line with the addon fully in our repo, and UPSTREAM (slower) is the
lever that takes the substrate toward just-the-MLX-pointer.

## EXTRACT MILESTONE — concrete execution scope (2026-06-21)

**Goal:** `genmlx-core` lives in the GenMLX repo as a standalone library; `mlx-core`
reverts to stock **minus one line** (`from_handle` pub). After this, the mlx-node
fork delta *above the substrate* is essentially that one line — the `mlx-sys`
substrate (guards + MLX pointer) is a separate, slower upstreaming track.

**Grounding that shapes it (verified against live source):** the ~33 ops are
ALREADY `genmlx-core/genmlx.rs` free fns that *delegate* to mlx-core impl-methods
(`fn cholesky(a){ a.cholesky() }`, `fn einsum(..){ MxArray::einsum(..) }`, a unary
macro for sigmoid/lgamma/…). So W-B is **Rust-internal** — repoint the delegators
at the FFI and delete the methods — with **no `mlx.cljs` change except the
`value_and_grad`/`computeGradients` pair** (the only ops `mlx.cljs` calls as static
`MxArray` methods: `(.computeGradients M …)` / `(.valueAndGrad M …)`).

### Part W-B — make `mlx-core` stock (do FIRST, in-place in the workspace)

1. **`genmlx.rs`** — rewrite the delegating free-fn bodies to inline
   `mlx_sys::mlx_array_<op>(a.as_raw_ptr(), …)` + `MxArray::from_handle(...)`,
   instead of `a.op()` / `MxArray::op()`. Covers the unary macro (sigmoid, erfinv,
   lgamma, digamma, expm1, bessel_i0e/i1e, flatten) + softmax/logaddexp/nan_to_num/
   inner/outer/diag/trace + linalg (cholesky, linalg_solve, solve_triangular,
   linalg_inv, tri_inv, cholesky_inv, qr, svd, eigh, eigvalsh, linalg_norm, einsum)
   + reduction (all, any, topk, logcumsumexp, searchsorted) + data (shapeArray).
   Copy `validate_axes` (reduction.rs:10) in as a private helper.
2. **`genmlx.rs`** — add free fns `value_and_grad` + `compute_gradients` wrapping
   the stock-pub generic `mlx_core::autograd::{value_and_grad, compute_gradients}`,
   using the JS-closure→MxArray recipe already proven in `genmlx-core/transforms.rs`.
3. **`mlx.cljs`** (only membrane change) — repoint the autograd bindings from the
   `MxArray` class `M` to the core object `c`: `(.computeGradients M …)` →
   `(.computeGradients c …)`, `(.valueAndGrad M …)` → `(.valueAndGrad c …)` (≈4
   call sites, lines ~728/736/749/756).
4. **`genmlx.rs`** — inline `take_last_native_error` (12 lines over `mlx_sys::
   mlx_take_last_error`); revert the mlx-core `pub` re-export to `pub(crate)`.
5. **Delete** the ~35 GenMLX `#[napi] impl MxArray` methods from mlx-core
   (`array/ops.rs`, `array/reduction.rs`, `array/data.rs`, `autograd.rs`
   value_and_grad_js/compute_gradients_js) → those files revert to stock. **Keep
   `from_handle` pub** (the one irreducible line). FIRST grep mlx-core for any
   internal callers of the to-delete methods (expected: none — they're
   GenMLX-added; delegators were the only consumers).
6. **Rebuild** (`node packages/genmlx-core/build.mjs`) + gates; **re-pin
   `membrane_coverage`** (the MxArray-method exports drop; the module free-fn
   exports stay — adjust the surface count + matrix in `docs/membrane-coverage.md`).

**W-B done means:** `mlx-core` diff vs merge-base = `from_handle` pub (1 line); all
gates green; the @genmlx/core JS surface unchanged except `valueAndGrad`/
`computeGradients` moving from MxArray-class to module exports.

### Part Tier-D — relocate `genmlx-core` out (do SECOND, mechanical)

1. **Move** `mlx-node/crates/genmlx-core` → `genmlx/genmlx-core` (its own
   single-crate workspace). Path-deps become `../mlx-node/crates/{mlx-core,mlx-sys}`
   (genmlx-core and mlx-node are siblings *inside* the genmlx repo). NO mlx-node
   workspace edit needed (`members = ["crates/*"]` glob drops it automatically).
2. **One NAPI registry** — commit genmlx-core's own `Cargo.lock` + `=`-pin the full
   ABI set (`napi=3.8.4`, `napi-sys=3.2.1`, `napi-derive=3.5.3`,
   `napi-derive-backend=5.0.2`, `napi-build=2.3.1`) + a CI assert diffing
   `cargo tree` napi* vs mlx-core. Re-declare `[profile.release]` (`lto=true`,
   `codegen-units=1`).
3. **`build.mjs` + npm package** — move `packages/genmlx-core` (package.json +
   build.mjs) to `genmlx/genmlx-core` too (so @genmlx/core is fully self-contained
   in the genmlx repo); the root `package.json` dep becomes `file:./genmlx-core`.
   Re-point `--manifest-path`; set `CARGO_TARGET_DIR=mlx-node/target` so the
   mlx-sys/paged-attn substrate builds once and the metallib colocation still finds
   `mlx.metallib`/`paged_attn.metallib`.
4. **Rebuild from the new location** + full gate suite.

**Tier-D done means:** `genmlx-core` source + npm package live in the GenMLX repo;
`mlx-node/crates` and `mlx-node/packages` carry NO genmlx-core; `@genmlx/core`
builds + loads with exactly one resolved napi* set; all gates green.

### Gates (after each rebuild)

`node packages/genmlx-core/build.mjs` → membrane contract guards (`exact`,
`gradient_fd`, `score_gradient`, `clip_contract`, `membrane_coverage`) +
`level0_certification` + `prng_key`/`prng_hygiene` + `vmap`/`vmap_property`/
`compile_fn`, serial (Metal-wedge).

### Risks

- **napi-skew** (Tier-D own lock) — mitigated by committed lock + `=`-pin full set
  + CI assert.
- **`build.mjs` target/metallib orchestration** (`CARGO_TARGET_DIR`) — the most
  error-prone step.
- **W-B FFI transcription** (~35 ops) — each `mlx_sys::mlx_array_<op>` signature
  must match; the compiler + guard suite catch errors.
- **`membrane_coverage` re-pin** — MxArray methods removed from the surface.
- **Internal callers of the deleted methods** — grep first (step W-B.5).

### Sequencing & PRs

W-B first (in-place, shared workspace, validate `mlx-core` stock), then Tier-D
(move + standalone build). Each spans both repos (mlx-node submodule + genmlx
main), so each is a two-repo PR pair like genmlx-65d5.

### Bean structure (proposed)

Update milestone `genmlx-cigu` (A+B done, C dropped) → children: **(W-B)** "make
mlx-core stock: methods → genmlx-core free fns" and **(Tier-D)** "relocate
genmlx-core to the GenMLX repo (standalone + napi pin)."

## Decisions for discussion

1. **Tier D mode** — D1 (out-of-tree workspace member, shared lock, recommended)
   vs D2 (standalone + `=`-pin + CI lock-diff).
2. **mlx-core target** — W-A (lift genmlx-core out, accept the ~33 orphan-bound
   methods) vs W-B (convert them to free fns for a truly stock mlx-core crate).
   This is the real scope decision: "stock *tree*" vs "stock *mlx-core crate*."
3. **Milestone boundary** — is Horizon 1 = WS1+WS3 (smaller, ~stock tree), with
   WS2 split into its own milestone? Or all three?
