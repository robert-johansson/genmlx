# Handoff: bring GenMLX up on macOS / Apple Silicon (Metal)

> **Written:** 2026-07-26 from the Jetson AGX Thor box (sm_110, aarch64 CUDA).
> **Audience:** the agent on the Mac. Everything below is on GitHub; you need no access to
> the Thor or RTX boxes.
> **Companions:** [`README.md`](README.md) (patch-tree design), [`SYNC-RUNBOOK.md`](SYNC-RUNBOOK.md),
> [`RTX-PRO-6000-HANDOFF.md`](RTX-PRO-6000-HANDOFF.md) (the sibling sm_120 port).
> **Tracking bean:** `genmlx-lr9c`.

## 0. Read this first: macOS is not a port

GenMLX was **developed on macOS/Apple Silicon**. Metal is the original backend; CUDA came later.
So this is re-validation after a large upstream sync, not a port.

Three things make it structurally easy, all verified rather than assumed:

- **Our 15 MLX patches are Metal-inert.** Of 11 touched files, 6 sit under `mlx/backend/cuda/`
  and are never compiled when `MLX_BUILD_CUDA=OFF`. The rest are backend-agnostic:
  `mlx/backend/cpu/inverse.cpp`, `mlx/primitives.{cpp,h}` (Cholesky/Inverse VJPs), plus two
  CMake files.
- **`crates/mlx-sys/build.rs` already branches on `target_os`** — macOS → `MLX_BUILD_METAL=ON`,
  Linux → `MLX_BUILD_CUDA=ON`.
- **Upstream mlx-node is macOS-first.** All four of its build/test/lint CI jobs are `macos-26`.
  The v0.0.8 code in this tree is *better* tested on Metal than on either of our Linux boxes.

`genmlx-lr9c` already got a **Mac M4 native build of `@genmlx/core` to succeed** (exit 0:
`index.node` 60 MB + `mlx.metallib` 162 MB + `paged_attn.metallib` 19 MB) from this same source.
That was before the v0.0.8 sync, so it needs redoing — but the architecture is proven.

## 1. Which machine

**Use the M4 Mac mini (32 GB) for the core work.** It is sufficient, and `genmlx-lr9c`'s
successful Metal build was on that machine.

Apple Silicon memory is **unified** — the GPU shares the same 32 GB as the OS and the build. That
governs which suites you can run:

| workload | resident | 32 GB M4 mini | 64 GB M2 Max |
|---|---|---|---|
| Native build (MLX C++ + Rust) | peak ~8-16 GB with capped `-j` | ✅ cap parallelism (§3) | ✅ |
| Core battery (fast/medium/slow, non-LLM) | < 4 GB | ✅ | ✅ |
| `qwen3.5-0.8b-bf16` LLM/VLM suites | ~1.7 GB | ✅ | ✅ |
| `qwen3-0.6b` | ~1.2 GB | ✅ | ✅ |
| 4-bit 35B MoE (`qwen3.6-35b-a3b-4bit`) | ~20 GB on disk, more resident | ⚠️ very tight — leaves ~10 GB for OS+GPU | ✅ comfortable |
| 8-bit 35B (`ornith-1.0-35b-8bit`) | ~36 GB | ❌ | ⚠️ tight |
| 4-bit 80B Qwen3-Coder-Next | ~42 GB | ❌ | ❌ |

**Recommendation:** do the whole port validation on the **M4 mini**. Everything that proves
"GenMLX works on macOS" — build, membrane, L0/GFI certification, the CLJS inference suite, and the
LLM/VLM suites that matter — runs on the 0.8B and 0.6B checkpoints and fits comfortably.

**Switch to the M2 Max (64 GB) only** if you want to additionally validate the 35B-class paths
(MoE forward, token-SMC branching, GRPO). Those are not required to call macOS working, and the
80B does not fit on either machine — do not attempt it.

If you only have one of them free, the M4 mini is the right choice.

## 2. What to clone

```bash
git clone --recursive https://github.com/robert-johansson/genmlx.git
cd genmlx
```

| repo | path | branch | notes |
|---|---|---|---|
| genmlx | `.` | `main` | |
| mlx-node | `mlx-node` | `genmlx/integration` | upstream v0.0.8 merged into our fork |
| mlx | `mlx-node/crates/mlx-sys/mlx` | `thor/stack-mlx-latest` | ml-explore `973e27f82` + 4 mlx-node NAX commits + our 15 |

Submodules land **detached** — correct and expected; the gitlink is the source of truth.
Immortal tags if you need to name the state: `pin/mlx-node/2026-07-26`, `pin/mlx/2026-07-26`,
`archive/pre-resync-2026-07-25`.

> Branch names are historical: `thor/stack-mlx-latest` is the shared MLX stack, not
> Thor-specific. Renaming to platform-neutral names is an open cosmetic item in `genmlx-lr9c`.

### Metal-relevant commits you are getting

- `de7e34290` — "Allow building NAX kernels below the macOS 26.2 deployment target"
  (**mlx-node's own Metal build fix — matters if your Xcode/SDK predates 26.2**)
- three NAX D256 SDPA commits — Metal-only fused full-attention for `head_dim 256`
- `51ad6fc14` — **ours**: adds `MLX_BUILD_JACCL` and `build.rs` sets it `OFF`. This exists
  because of a Mac-only blocker `genmlx-lr9c` hit: MLX's Thunderbolt-RDMA `jaccl` backend
  (upstream PRs #2808/#3094/#3174/#3412/#3459) tried to **install into `/usr/local`** during the
  build. If you see anything reaching outside the build tree, that is where to look.

## 3. Build

```bash
cd mlx-node
yarn install
yarn build:ts
yarn build:native                       # @mlx-node/core (needed by mlx-node's own suite)
node packages/genmlx-core/build.mjs     # @genmlx/core   <-- REQUIRED, easy to miss
```

**Cap build parallelism on the 32 GB mini.** MLX's C++ template instantiation is memory-hungry
and an unbounded `-j` will thrash or OOM:

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=6     # 4 if you see pressure
export CARGO_BUILD_JOBS=6
```

Expect a **full MLX build from scratch** (hours on first run).

### macOS-specific: the metallibs

Unlike CUDA (which JITs kernels), Metal needs precompiled `.metallib` files sitting **next to
`index.node`** — MLX and paged-attn locate them via `dladdr`. `build.mjs` copies them for you:

- `mlx.metallib` from `target/<arch>/release/build/mlx-sys-*/out/lib/`
- `paged_attn.metallib` from the same dir or `mlx-paged-attn-*/out/`

If `build.mjs` reports either as not found, the native build did not produce them — that is a
build failure, not a packaging one. On Linux this whole step is skipped (`build.mjs:51`), so it
is **untested by our two Linux boxes** and is the most likely place for a Metal-only surprise.

### Verify the addon before trusting any test

```bash
realpath node_modules/@genmlx/core       # -> mlx-node/packages/genmlx-core
ls -l mlx-node/packages/genmlx-core/     # index.node + mlx.metallib + paged_attn.metallib
node -e "const m=require('./mlx-node/packages/genmlx-core');const k=Object.keys(m);
         console.log('functions:',k.filter(x=>typeof m[x]==='function').length,
                     'objects:',k.filter(x=>typeof m[x]!=='function').length)"
# expect: functions: 227  objects: 6
```

> `Object.keys(...).length` is **233**, not 227 — the matrix pins *function* exports; the other 6
> are `__internal__` plus the enums. Comparing raw key count against 227 is a guaranteed false
> alarm.

## 4. Running tests

```bash
bunx --bun nbb@1.4.208 test/genmlx/<file>.cljs
test/run.sh all                          # NOTE: no TEST_TIME_SCALE — see below
```

**`TEST_TIME_SCALE` should be 1 (i.e. unset) on Apple Silicon.** It is a host-speed knob, and
**Apple Silicon is the calibration baseline** — the tier caps and the absolute-ms perf assertions
in `fused_mcmc_test` (200 ms / 500 ms budgets) were tuned there. Thor uses 8; the RTX box settled
on 6. If those perf assertions fail on your Mac at scale 1, that is a *real* performance signal,
not a harness artifact — report it.

`GLIBC_TUNABLES` is a glibc/Linux thing — **not needed on macOS**.

### Expected results

| gate | expected |
|---|---|
| `membrane_coverage_test` | 227 functions / 49 omissions, 0 failures |
| `level0_certification_test` | **68/68** |
| `genjax_compat_test` | **73/73** |
| `gen_clj_compat_test` | 356/356 |
| `exact_test` | 120 passed |
| `gradient_fd_test`, `score_gradient_test`, `clip_contract_test` | 0 failures |
| full `run.sh all` | **428/430** was the last Linux figure; expect Metal to differ (§5) |
| `cd mlx-node && yarn test --run packages/agent` | 302/302 |

## 5. What will legitimately differ from Linux — do not "fix" these

- **Paged attention is Metal-only.** On CUDA it is force-disabled, so every paged code path in
  this tree is *less* exercised on our Linux boxes than on yours. Expect paged suites that skip
  on Linux to actually run for you. `mlx-node/packages/agent` sets `requirePagedCache: true`
  gated by `agentPagedCacheSupported()`, which returns `platform === 'darwin'` — so **macOS keeps
  upstream's behaviour and only CUDA diverges**. That gate is correct for you; leave it.
- **~15 Metal-only or Metal-calibrated tests** exist that our Linux boxes skip. The RTX agent
  flagged filing a platform-gating bean for these when someone triages on Metal — **that is you**.
  Six test files currently gate on `mx/metal-is-available?`; make the gating symmetric so
  Thor-only suites skip cleanly for you and vice versa (a `genmlx-lr9c` done-means item).
- **MCMC chain trajectories will differ.** Several sampler budgets were re-measured on CUDA on
  2026-07-26. Seeded chains are *not* expected to reproduce across Metal and CUDA kernels — the
  tests assert convergence to the true posterior, not a specific trajectory, and the fixes moved
  them from "depends where the chain started" to "converged with several times the tolerance in
  margin". They should pass, but if a band fails, **re-measure across seeds before touching it**
  (see §6).
- **`cargo test`** — a `gr51` SIGABRT at process exit is a known Thor observation and does **not**
  reproduce on sm_120. Whether it fires on Metal is genuinely unknown and worth reporting.

## 6. House rule: never weaken a test to make it green

This tree has a standing rule, and the last two sessions were run under it. If a stochastic
assertion fails:

1. Inventory every entropy source it depends on (`dyn/auto-key` is the usual hidden one).
2. Measure the pass rate across **≥10 seeds at the current budget**.
3. If nearly all pass → seeding alone is honest. If a real fraction fail → the *budget* is
   under-powered and that is the fix.

Never widen a tolerance, and never seed without the sweep — a seed that happens to sit near the
answer hides a real defect. That trap was hit and caught: `compiled_optimizer` looked fixed by
seeding until a sweep showed 3 of 15 seeds landing short.

## 7. Known-red / do-not-chase

| item | bean | note |
|---|---|---|
| `sbc_test` excluded from the battery | `genmlx-ec1c` | ~6 h SBC workload; use `test/run_sbc.sh`. Not a failure. |
| KV-prefix byte-stability weakened | `genmlx-8hod` | accepted regression from adopting upstream `convert-messages`; not platform-related |
| ~20 `auto-key` sites in `mcmc.cljs` | — | `mala`, `nuts` etc. still discard the caller's `:key` for their initial trace. Five sites were fixed 2026-07-26; the tail is an open, deliberately unswept change |

**Five flaky tests were fixed on 2026-07-26** (`mcmc_diagnostics`, `fused_mcmc` ×2, `kernel_dsl`,
`pmcmc`, plus `provider-live`). Nearly all were **one bug**: samplers auto-keying their initial
trace, so `:key` pinned the chain's steps but not where it started. If any reappear on Metal,
that is **new information** — report, do not re-fix.

**Checkpoints lie about their dtype.** Directories named `qwen3.5-0.8b-mlx-bf16` are **4-bit**
(`quantization: {bits: 4}` in their own `config.json`). The genuine one is
`mlx-community/Qwen3.5-0.8B-bf16` in the HF hub cache. `provider-live` was permanently red for
months because of this, misread as a provider defect. **Always read `config.json`** before
concluding anything about numerics or model capability.

## 8. What to report back

1. Whether the native build completed, how long, and whether the two `.metallib` files landed.
2. Whether the JACCL `/usr/local` escape recurred.
3. `functions: 227 / objects: 6`.
4. `run.sh all` totals at `TEST_TIME_SCALE` unset, and whether the `fused_mcmc` perf assertions
   pass at scale 1.
5. Which suites are Metal-only vs Linux-only — the input for symmetric gating (`genmlx-lr9c`).
6. Whether any of the five fixed flaky tests reappear.
7. Whether `cargo test` shows the `gr51` exit-time SIGABRT.
8. Peak memory during build and during the LLM suites, so we know how much headroom 32 GB leaves.
