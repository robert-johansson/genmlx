# Metal test triage ledger (Mac M4)

> **Tracking:** bean `genmlx-lr9c` (Metal bring-up after the 2026-07-25/26
> resync). **Status ledger of every file that does not pass `test/run.sh all`
> on the macOS/Metal host, with its cause** — the Mac counterpart of
> `docs/cuda-test-triage.md`. Update this file whenever the residual set
> changes.
>
> **Pins for the numbers below:** genmlx `d5aff7b` + the genmlx-lr9c Metal
> bring-up commits → mlx-node `806ade9` (`genmlx/integration`) → mlx
> `135c5945e` (`genmlx/metal-bringup`), 2026-07-27, M4 mini 32GB,
> macOS 26.5.2, `TEST_TIME_SCALE` unset (=1, the calibration baseline).

## Current residual (2026-07-27)

| file | cause | status |
|---|---|---|
| `llm/gemma4_test` | v0.0.8 upstream dropped `forward`/`forwardWithCache` from the native `Gemma4Model` (chat-session-only surface); genmlx token-level GF needs the seam | RED — real regression, bean `genmlx-lge5` (high). Only reproducible on hosts with the gemma4 checkpoint (both Linux boxes skip on ENOENT). |
| `fused_mcmc_test` | `cached vectorized < 500ms` perf budget: measured 618–637 ms across 5 runs at scale 1 (~24% over the pre-resync-tuned budget). Real perf regression of the resynced pin, consistent with the fused-eval buffer-retention change | RED (1 assert) — reported per the bring-up handoff §4; root cause tracked in bean `genmlx-rsgr`. Do NOT bump the budget; fix upstream or re-tune deliberately. |

## Resolved during the 2026-07-27 bring-up (for the record)

- **pi_assess D margin** (was in the residual): RESOLVED by Thor's slab-graph
  assess fix (`f37d2b1`, genmlx-3n7b) — the D law is tight again and the
  step-vs-slab drift is reclassified to D2. Validated 20/20 on Metal
  2026-07-27. Bean `genmlx-9y1c` superseded.
- **forward_golden cross-backend pins**: now BACKEND-CONDITIONAL
  (`9595918`, genmlx-z4hy) — Metal + CUDA golden maps, capture-don't-reuse
  skips. Metal branch validated 26/26.

- **GRPO training SIGTRAP** (`world_train_test` CRASH@0s): upstream made
  Metal command encoders thread_local; cross-thread eval threw and the
  unguarded conversion shim escalated to `std::terminate`. Fixed by the mlx
  fork's Metal lazy per-thread encoder registration (sibling of the CUDA
  `genmlx-isws` patch) + `MLX_GUARD_BOOL` on the `copy_to_buffer` family.
- **499k buffer-wall cluster** (medium tier 4-way: `gfi_laws`,
  `gfi_combinator_invariants`, `steppable`, `trace_translator`,
  `genjax_compat`, + serial-tier fallout): `getNumResources`/
  `getResourceLimit` were hardcoded-0 stubs on every backend, silently
  disabling the membrane's proactive count sweep. Fixed by real fork
  accessors + guarded shims.
- **Fused single-graph buffer wall** (`fused_mcmc` solo): the resynced pin
  retains every per-step temporary live for the whole fused eval
  (~166–205 buffers/step, linear; wall ≈ 2400 steps — the old
  `:native 80000` limit was 25× past it). `fused-ops-limits` recalibrated
  from measurement; white-box 3000-step chains resized; upstream fix
  tracked in `genmlx-rsgr`.
- **Paged-vs-flat seam** (`forward_golden`, `cljs_forward_qwen35`): v0.0.8
  defaults VLM checkpoints to the block-paged KV adapter on Metal, refusing
  the flat Tier-1 seam; CUDA force-disables paging and never sees it. Fixed
  via the native `load(path, paged_override)` param + `:paged? false` in
  the flat-seam suites (load policy design: `genmlx-eacv`).
- **Golden logprob re-pin** (`forward_golden`): the 973e27f82 kernel bump
  shifted bf16 logprobs 0.02–0.11 with ALL token ids/rankings unchanged;
  goldens re-captured after parity + oracle verification (the pin's
  documented lifecycle).
- **Checkpoint layout** (`llm/bytes_test`): the bare `qwen3.5-0.8b` dir on
  this host is an unconverted original export (`model.language_model.*` +
  `mtp.*`) the loader cannot read; test aligned with its 24 sibling suites
  on `-mlx-bf16`. Loader-side named error: bean `genmlx-76j5`.
- **f32-floor assertion** (`lanczos_test`): 1e-5 absolute on
  log-gamma(50)=144.57 demands sub-eps relative accuracy; new kernel is
  1.6 ulp off (correctly rounded). Bound now scales with f32 ulp.
- **Mis-tiered** (`llm/token_mcmc_test`): 113.7s solo vs the 45s fast cap
  at scale 1 — retagged `medium`.
- **Vacuous-green legacy** (`qwen2_coder_test`): no qwen2 class exists in
  `@genmlx/core` (and never did post-fork-min); now an explicit capability
  skip, scrap-or-support decision in `genmlx-r70f`.
- **metallib floor-gate parser** (build-time): u16 minor read folded a
  nonzero macOS patch version into "26.517" and rejected a correct build;
  min-OS minor/patch are separate bytes (mlx-node `94ba1e2`).

## Known process-level warts

- The bunx launcher race (parallel cold-cache `bunx nbb` collisions) can
  still cost 1-4 files per battery run (`FAIL(1)` at 0-1s with no
  cljs.test output; alphabetically-first files in a tier; they pass solo).
  The runner's one-retry heuristic does not catch every flavor.
- `inference_gradient_test` `adev-quadratic-cost-gradient-matches-analytical`
  is a low-rate statistical flake on Metal (1 battery failure vs 10/10 solo
  passes, 2026-07-27) — bucket F of the CUDA ledger; the sound fix
  (across-seed measurement + seed-pin + measured band) is specced in
  `genmlx-5hhd`.
- `genjax_compat_test` `s7-diagnostics` flakes under 4-way GPU contention
  (2 of 3 batteries, 2026-07-27; 73/73 solo) — a statistical diagnostics
  section, same bucket-F class.
- `sandbox_test` hang-detection asserts are timing-sensitive under 4-way
  CPU load (1 of 3 batteries; 31/31 solo).
- `mcmc_stationary_test` `multi-algorithm-agreement` fails ~2-of-5 runs
  (measured 2026-07-27; 1 assert, serial tier, contention-free) — the
  handoff's known-red `auto-key` class: this file's samplers are in the
  deliberately-unswept tail of the ~20 `auto-key` sites in `mcmc.cljs`,
  so every run draws a fresh initial-trace seed against a statistical
  band. Sound fix specced in `genmlx-5hhd` + the auto-key sweep.
- `compiled_loss_grad_test` `learn-api-handler-test` fails ~1-in-10 runs
  (measured 2026-07-27: 9/10 solo + 1 battery miss; first appearance after
  four clean batteries) — unseeded `co/learn` init draws `:mu` from the
  prior each run, and a far-tail start misses the ±0.5 convergence band in
  the 1000-iteration budget. Same `auto-key`/bucket-F class; sound fix via
  `genmlx-5hhd` (seed-pin + measured band or a measured budget increase).
