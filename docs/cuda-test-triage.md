# CUDA test triage ledger (Thor)

> **Tracking:** bean `genmlx-4u57` (the systematic triage), release gate in
> `genmlx-0e28`. **Status ledger of every file that does not pass
> `test/run.sh all` on the Thor CUDA host, with its bucket** — kept in-tree so
> a red tier run is never ambiguous. Update this file whenever the residual
> set changes.
>
> **Pins for the numbers below:** genmlx `b579201` → mlx-node `6aabbd9`
> (thor/d58a-up-sync) → mlx `102a90cb` (thor/mlx-sync-d58a), 2026-07-18.

## How to run the suite on this host

```bash
# Tier-capped batch runs (the knob multiplies every tier cap; 8 is the
# measured Thor scale — genmlx-9ox0):
TEST_TIME_SCALE=8 test/run.sh all

# Autonomous / unattended sessions: EVERY GPU run goes through the guarded
# runner (MemAvailable floor watchdog; genmlx-h3p5):
scripts/guarded-run.sh <name> bunx --bun nbb@1.4.208 test/genmlx/<file>.cljs
```

## Current residual (2026-07-18, true binary)

Tier totals from the d58a option-a validation (fast 190/190, medium 144/149,
slow 70/77 — run against the then-stale addon), re-validated file-by-file on
the **true** (post-linkage-fix, genmlx-s8ij) binary. After the 2026-07-18
release-v0 sweep, the honest residual is:

| file | bucket | status |
|---|---|---|
| `sbc_test` | D (heavy) | TIMEOUT at the 4800s scaled slow cap. Simulation-based calibration, the one `results/` producer in the list; needs a split or a dedicated overnight window, not a cap bump. |
| `fused_mcmc_test` | F (statistical band) | Single-assert MALA convergence-band flake, non-deterministic (a different assert flaked in the overnight run than the two filed in `genmlx-5hhd`). Known class: heavy-tailed single-chain burn-in miss; needs across-seed measurement (`genmlx-5hhd`), not blind band-widening. |
| `gradient_mcmc_property_test` | F (statistical band) | `mala-and-hmc-posterior-means-agree` single-assert flake, same class (`genmlx-5hhd`/`genmlx-06xw`). |
| `limitations_fixes_test` | F (statistical band) | `involutive-mcmc` variance flake; passes solo repeatedly (`genmlx-9ox0` observation). |
| `mcmc_diagnostics_test` | F (statistical band) | `hmc-5d-adaptive-acceptance` single-assert flake, same class. |

Everything else in the overnight not-pass dozen is resolved:

| file | was | resolution (2026-07-18) |
|---|---|---|
| `gather_qmm_oracle_test` | FAIL(1) | Stale-binary linkage casualty (genmlx-s8ij) — 13/13 PASS on the true binary. |
| `llm_qwen3_next_native_test` | CRASH 133 (`cannot open source file "cute/..."`) | Same linkage casualty (missing `../include` at the stale copied addon path) — 13/13 PASS on the true binary. |
| `llm_branched_test` | CRASH 133 | Same — 8/8 PASS on the true binary. |
| `llm_codegen_test` | FAIL(asserts) | Stale-binary casualty — 91/91 PASS on the true binary. |
| `llm_gemma4_test` | FAIL(1) ENOENT | Checkpoint not provisioned on this host — now SKIPs cleanly (the `llm_cljs_forward_test` contract). Provision `~/.cache/models/gemma4-e2b-it-4bit-safe` to exercise it. |
| `qwen2_coder_test` | FAIL(1) ENOENT | Same — clean SKIP; provision `~/.cache/models/Qwen2.5-Coder-0.5B-mlx` to exercise it. |
| `resource_recovery_test` | FAIL(4) | Metal-gated with the nvna negative-contract pattern: off-Metal it asserts the limit/count queries report 0 AND the 700k-alloc loop survives with both recovery layers silent (8/8 on CUDA). The full wall experiment still runs on Metal. |

## Bucket taxonomy (history — how 36 became 5)

The first full Thor run (2026-06-24, 327/363) triaged into six buckets. Their
lifecycle, for archaeology:

- **A — Cholesky/Inverse VJP SIGABRT** (the dominant cluster): RESOLVED at
  root by the MLX-fork gradient fixes (Cholesky::vjp, Inverse::vjp,
  singular-Inverse→NaN, CCCL-NVRTC; `genmlx-eb1d`). Residual: a <0.007%
  mvn-log-prob-finite statistical flake (test-side).
- **B — Metal-only assertions**: `memory_test` fixed via `genmlx-nvna`
  (platform-honest asserts); `resource_recovery_test` fixed 2026-07-18 (see
  above). Bucket empty.
- **C — CUDA shape/op differences**: `searchsorted` composite fixed
  (`genmlx-fqqx`); `native_guard` message-detail regression fixed
  (`genmlx-ne1q`); `level0_certification` healed with the VJP/CCCL fixes;
  translator-sparsity/gfi_laws fixed at root (mlx `40f31ae6`,
  `genmlx-612c`). Bucket empty.
- **D — slow-tier timeouts**: addressed by the `TEST_TIME_SCALE` host-speed
  knob (`genmlx-9ox0`, commit `b471445`; use 8 on Thor). Residual: `sbc_test`
  (above).
- **E — LLM crashes/fails**: qwen3.5 hybrid-forward reshape fixed
  (`genmlx-9iqc`); thinking-policy/empty-generation fixed (`genmlx-87ga`);
  the 2026-07-18 residuals were all stale-binary linkage casualties or
  missing checkpoints (above). Bucket empty.
- **F — statistical band flakes**: the four files above. This is the ONLY
  open bucket with test-visible reds. They are single-assert,
  non-deterministic, and pass solo/on reruns; the sound fix (across-seed
  measurement, seed-pin + measured band) is specced in `genmlx-5hhd`.

## Known process-level warts (not test-logic failures)

- **Exit-teardown SIGABRT/SIGSEGV (`genmlx-gr51`)**: on CUDA, a process that
  loaded a model can abort AFTER all assertions pass, during driver-shutdown
  static destruction (`Destroy(handle_) failed: driver shutting down`, ~1/30
  also as a bun exit segfault). Scripting rule: **read the printed
  PASS/FAIL/SKIP output, not the exit code**, for model-loading runs. See
  README "Known platform warts".
- **GPU run discipline (`genmlx-h3p5`)**: one GPU process at a time; heavy
  runs through `scripts/guarded-run.sh`. See `docs/thor-gpu-discipline.md`.
