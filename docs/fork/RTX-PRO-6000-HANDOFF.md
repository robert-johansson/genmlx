# Handoff: bring GenMLX up on the RTX PRO 6000 Blackwell (sm_120)

> **Written:** 2026-07-26, from the Jetson AGX Thor box (sm_110, aarch64, 128 GB unified).
> **Audience:** the agent/engineer working on the RTX PRO 6000 (sm_120, discrete, 96 GB VRAM,
> x86_64). **You do not need access to the Thor box** — everything below is on GitHub.
> **Companions:** [`README.md`](README.md) (patch-tree design), [`SYNC-RUNBOOK.md`](SYNC-RUNBOOK.md),
> [`CONFLICT-LEDGER.md`](CONFLICT-LEDGER.md). Deeper port plan + risk register:
> bean `genmlx-1aea` and `~/code/mlx/RTX-PRO-6000-PORT.md` on the Thor box (not in git).

## 1. What to clone

One command gets everything. **Use `main` on genmlx; the submodule pins carry the rest.**

```bash
git clone --recursive https://github.com/robert-johansson/genmlx.git
cd genmlx
```

Verify you landed on the intended tree:

| repo | path | expected SHA | branch it lives on |
|---|---|---|---|
| genmlx | `.` | `5d9fba6e0c4f6e4a74213d0c162bb264d111bf81` | `main` |
| mlx-node | `mlx-node` | `7c8db892cda1dcdae9763adf2232fc0c1d9b0588` | `genmlx/integration` |
| mlx | `mlx-node/crates/mlx-sys/mlx` | `fd21ea68cca6137adde25c8362ec54cb3f0a0522` | `thor/stack-mlx-latest` |

```bash
git rev-parse HEAD
git -C mlx-node rev-parse HEAD
git -C mlx-node/crates/mlx-sys/mlx rev-parse HEAD
```

Submodules land **detached** at those SHAs — that is correct and expected. Don't "fix" it by
checking out a branch; the gitlink is the source of truth.

### Immortal pin tags (use these if you need to name the state)

```
mlx-node   pin/mlx-node/2026-07-26      = 4dc64130 (the v0.0.8 merge point)
mlx        pin/mlx/2026-07-26           = fd21ea68c
mlx-node   archive/pre-resync-2026-07-25 = the pre-sync tip, never deleted
mlx        archive/pre-resync-2026-07-25 = ditto
```

> The gitlink at genmlx `main` currently points **one commit past** `pin/mlx-node/2026-07-26`
> (`7c8db892`, a test-only change). Trust the gitlink, not the tag, for "what to build".

### Branches you can ignore

`thor/*` (except `thor/stack-mlx-latest`, which the gitlink uses), `sync/*`, `mirror/*`,
`nax-on-ml-explore`, `backup/*`, `pre-rebase-*`. They are history and staging. Nothing on the
RTX box should need them.

## 2. What this tree is

- **mlx-node** = upstream `mlx-node/mlx-node` **v0.0.8** (`0ebeaa57`) merged into our fork,
  plus ~78 of our commits. Merged 2026-07-25; 14 conflicted paths resolved (see the ledger).
- **mlx** = `ml-explore/mlx` `973e27f82` + mlx-node's 4 NAX commits + **our 15 CUDA/CPU patches**.

Our 15 MLX patches, newest first — these are what must keep working on sm_120:

```
fd21ea68c  cap LRU cache auto-grow at 4x configured capacity        (genmlx-pnaw)
1f968c398  graph-construction failure -> graph-less fallback        (genmlx-5wrl)
66c75f01e  drain cp_async before the qmm_sm80 epilogue reuses smem  (genmlx-mdet)  <-- see below
a5314e915  never throw from ~CaptureContext                         (genmlx-kfli)
0bd439c8d  lazily register per-thread stream encoders (PARTIAL)     (genmlx-isws)
51ad6fc14  MLX_BUILD_JACCL build option
b324c54b6  synchronize device->host copy in move_to_unified_memory
a5877e761  CPU Inverse on singular matrix -> NaN instead of abort
5beb92685  add toolkit CCCL include dir to NVRTC compilation
0a880eeca  Inverse::vjp        (backend-agnostic autograd)
5edece428  Cholesky::vjp       (backend-agnostic autograd)
13116358c  Thor sm_110: auto-grow CUDA graph cache
dc6583310  Thor sm_110: persistent arch/source-keyed NVRTC cache
01d81bf19  Thor sm_110: integrated-GPU full-pool allocator + graph limits
a3f9fe190  Thor sm_110: honor MLX_CUDA_DISABLE_MEMPOOL
```

**`66c75f01e` matters more on your box than on ours.** `supports_qmm_sm80` gates on `cc >= 8`, so
sm_120 takes that path; without the `cp_async_wait<0>` the qmm_sm80 epilogue reuses the smem union
mid-flight and you get silent, nondeterministic quantized-matmul corruption.

### The four Thor-labelled patches are NOT a port cost

Verified by reading, not assumed:

1. **The integrated-GPU allocator is runtime-gated**, not arch-hardcoded:
   `cudaDeviceGetAttribute(&integrated_, cudaDevAttrIntegrated, device_)` plus
   `integrated_ == 1 && concurrent_managed_access_ == 1`. On a **discrete** card that is false and
   the code falls through to stock MLX's async mempool. Written *for* Thor, only *active* on Thor.
2. **`get_graph_limits` already has `case 1200: // Consumer Blackwell`**, sharing Thor's
   `ops=100 / mb=1000`. No arch-table entry needed.
3. **`crates/mlx-sys/build.rs` defaults `MLX_CUDA_ARCHITECTURES` to `110a;120a;121a`** — sm_120 is
   named. Override with that env var if you want a faster single-arch build (`120a`).

If you see the pool-less path taken on your box, something is wrong with the integrated
detection — that is a bug, not expected behaviour.

## 3. Build

**`yarn build` alone is not enough.** GenMLX loads `@genmlx/core`, which only `build.mjs` produces.

```bash
cd mlx-node
yarn install
yarn build:ts                              # tsc -b
yarn build:native                          # produces @mlx-node/core (needed by mlx-node's own tests)
node packages/genmlx-core/build.mjs        # produces @genmlx/core  <-- REQUIRED, easy to miss
```

`build.mjs` also colocates the CUDA JIT headers into `packages/include/`. Skip it and you get
cold-cache NVRTC "cannot open cute/..." errors that surface, confusingly, as
`item: array must have size 1`.

Expect a **full MLX build from scratch** (hours). On Thor an *incremental* MLX rebuild is ~7-9 min;
your first one will not be.

### Environment

The Thor block is below. **Check each line rather than copying blindly** — some is Tegra-specific:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu   # aarch64 -> x86_64
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192                # see note
export PATH="$HOME/.local/node/bin:$HOME/.bun/bin:$PATH"
```

`GLIBC_TUNABLES`: on Thor, omitting it makes every suite that loads `@genmlx/core` die instantly
with *"cannot allocate memory in static TLS block"* — the runner reports ~all files FAILing in 1s,
which looks catastrophic and is pure harness error. Root cause is mimalloc's TLS block.
**x86_64 DOES need it — answered 2026-08-01.** A bare `node -e "require('@genmlx/core')"` on the
RTX box dies without it; with `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192` it loads and
reports its exports. So set it on every run here too, exactly as on Thor. Use `8192`, not a
larger value — 262144 causes `pthread_create` EINVAL.

### Verify the addon before trusting any test result

```bash
realpath node_modules/@genmlx/core        # must point into mlx-node/packages/genmlx-core

# Use BUN — it is the runtime the battery actually runs under, so it is the
# surface that matters. (Verified 2026-08-01: bun and node expose a
# byte-identical function list here, 237/6 both, so this particular check is
# runtime-independent. Do not assume that for anything else — the two have
# different NAPI implementations.)
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192 bun -e \
  "const m=require('./mlx-node/packages/genmlx-core');
   const k=Object.keys(m);
   console.log('functions:',k.filter(x=>typeof m[x]==='function').length,
               'objects:',k.filter(x=>typeof m[x]!=='function').length)"
# expect: functions: 237  objects: 6      <- 237, measured 2026-08-01 (was 227 when this
#                                            doc was written; the surface grew). The membrane
#                                            COVERAGE MATRIX is the authority, not this number:
#                                            membrane_coverage_test partitions every export into
#                                            wrapped + intentional-omissions, so it names exactly
#                                            what moved. See docs/membrane-coverage.md.
```

> **`Object.keys(...).length` is 233, not 227.** The membrane matrix pins *function* exports;
> the extra 6 are `__internal__` plus the enums `BuiltinRewardType`, `DType`, `ChatRole`,
> `ElementType`, `OutputFormat`. Comparing raw key count against 227 is a guaranteed false alarm —
> that mistake was made and corrected during the v0.0.8 sync.

## 4. Running tests

```bash
bunx --bun nbb@1.4.208 test/genmlx/<file>.cljs        # single suite
TEST_TIME_SCALE=8 test/run.sh all                     # tiered battery
```

- **`--bun` matters.** Without it you drop to Node.
- **`ps` lies about the runtime.** It shows the worker as `node .../nbb`, but
  `readlink -f /proc/<pid>/exe` resolves to `.../bin/bun` — `bun run --bun` preserves node's argv.
  Check `/proc/<pid>/exe`, never the command line. (A stale note claiming `run.sh` runs on Node
  came from exactly this misreading.)
- **`TEST_TIME_SCALE` is a host-speed knob** that multiplies every tier cap and the absolute-ms
  perf assertions. `8` is the measured **Thor** value. Your box is likely faster — **re-tune it**.
  Running a suite standalone leaves it unset (= 1), which makes perf assertions false-red:
  `fused_mcmc_test` has 200ms/500ms budgets that Thor misses at scale 1.

### Expected results (as of this handoff, on Thor)

| gate | expected |
|---|---|
| `membrane_coverage_test` | 227 functions / 49 omissions, 0 failures |
| `level0_certification_test` | **68/68** |
| `genjax_compat_test` | **73/73** |
| `exact_test` | 120 passed |
| `gradient_fd_test` / `score_gradient_test` / `clip_contract_test` | 0 failures |
| `qmm_determinism_test` | ALL PASS (guards `66c75f01e` — **watch this one on sm_120**) |
| `gather_qmm_oracle_test` | 13 PASS 0 FAIL |
| full `run.sh all` | 428/430 at the last Thor run |

`mlx-node` has its own suite — **it is not part of `run.sh`**, and was missed for a long time
because of that:

```bash
cd mlx-node && yarn test --run packages/agent      # expect 302/302 on Thor
```

## 5. Known-red / do-not-chase list

Do not spend time on these; they are tracked and not sm_120 problems.

| item | bean | note |
|---|---|---|
| `sbc_test` excluded from the battery | `genmlx-ec1c` | ~6h SBC workload; its own header forbids monolithic runs. Use `test/run_sbc.sh`. Not a failure. |
| `cargo test` SIGABRT at process exit | `genmlx-gr51` | upstream MLX driver-shutdown teardown. On Thor `cargo test` can never be green. **Check whether this reproduces on sm_120** — that is genuinely useful data. |
| Thor global-OOM cascade | `genmlx-h3p5` | Thor-specific; should not apply to a discrete card. If you see it, that is a new finding. |
| KV-prefix byte-stability weakened | `genmlx-8hod` | accepted regression from adopting upstream's `convert-messages`; not platform-related. |
| ~20 `auto-key` sites in `mcmc.cljs` | — | `mala`, `nuts` etc. still discard the caller's `:key` for their initial trace, so they are not reproducible. `hmc` + the 3 fused samplers were fixed 2026-07-26; the rest is an open, deliberately unswept change. |

**Four flaky tests were fixed on 2026-07-26** (`mcmc_diagnostics`, `fused_mcmc` ×2, `kernel_dsl`,
`provider-live`). Three were one real bug: samplers auto-keyed their initial trace, so a seeded
chain still started somewhere different every run. If any of those go red on your box, it is
**new information**, not a known flake — please report rather than re-fix.

**Model checkpoints lie about their dtype.** Directories named `qwen3.5-0.8b-mlx-bf16` under
`~/.mlx-node/models` and `~/.cache/models` are **4-bit** (`quantization: {bits: 4}` in their own
`config.json`). The genuine one is
`~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-bf16/snapshots/<sha>/`. Always read
`config.json` before concluding anything about numerics or model capability — a whole class of
"provider defect" turned out to be this.

## 6. The real open question on your box: capacity

Everything above is code. The substantive unknown is **96 GB discrete VRAM vs 128 GB unified**.

The 80B 4-bit Qwen3-Coder-Next, the 4-bit 35B MoE, and the branchable-KV / token-SMC / Route-B
work were all sized against a pool where **host RAM *is* GPU memory**. On a discrete card it is
not. Also note a measured Thor finding that may not transfer: model load is expert-lazy (~7-8 GB),
but any forward makes packed experts device-resident, so steady-state owned memory exceeds native.

Worth establishing early, before building anything on top:
- does the 4-bit 35B (~18.6 GB resident) fit and run comfortably?
- does the 4-bit 80B fit in 96 GB at all?
- how much branching headroom is left for token-SMC?

## 7. What to report back

1. Whether the build completed, and how long the first MLX build took.
2. Whether `GLIBC_TUNABLES` was needed on x86_64.
3. The `functions: 227 / objects: 6` export check.
4. `qmm_determinism_test` and `gather_qmm_oracle_test` results — the sm_80+ kernel path.
5. A re-tuned `TEST_TIME_SCALE` for your hardware.
6. `run.sh all` totals, and **specifically whether any of the four fixed flaky tests reappear**.
7. Whether `cargo test` SIGABRT (`gr51`) reproduces.
8. The capacity answers from §6.
