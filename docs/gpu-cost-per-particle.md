# Cost-per-particle: scalar vs batched inference paths

Measured inventory of host-bound inference paths (bean `genmlx-2gu2`, under
milestone `genmlx-819v`). Produced by `bench/cost_per_particle.cljs` — run it
per arch and add a column; **numbers are per-ARCH, not per-backend**.

The diagnostic: sweep N, watch wall-clock **ms/particle**. Flat in N ⇒
host-bound (per-particle host GFI calls dominate, GPU idles). Falling in N ⇒
amortizing. Model: the 1-latent + 5-obs normal-normal from
`inference_smc_test`, analytical path stripped on both sides (handler path
measured, not L3 elimination).

## sm_110 (Jetson AGX Thor, 2026-07-28, genmlx d463384 / mlx-node 722bf55e / mlx a27ddcaef, bun 1.3.14)

### Anchors (harness credibility)

| anchor | im8n reference | measured here | verdict |
|---|---|---|---|
| membrane micro-latency | ~1 ms/eval | 0.055 ms/eval (tiny add+eval!) | credible — im8n's figure was a model-sized graph on sm_120; same sub-ms order for the membrane roundtrip |
| vgenerate N=3000 | ~21 ms (metareasoner world model) | 5.6 ms (5-site model) | reproduces — smaller model, same order, right direction |

### Scalar paths — ALL host-bound (flat ms/particle in N)

| path | N sweep | ms/particle(-step) | verdict |
|---|---|---|---|
| `p/generate` ×N loop | 1→3000 | 6.3 → **2.4 flat** | host-bound |
| `importance-sampling` | 1→1000 | 9.5 → **3.9 flat** | host-bound (deep-materialize adds ~1.5 ms/p over raw generate) |
| `mh` (per chain-step) | 1→20 chains | ~~61.0–61.6~~ → **10.8–11.1** (S=10 chains), **5.66** long-chain | was the worst offender; **fixed** (genmlx-k1z7, see below) |
| `smc` (per particle-step) | 1→500 | 49.7 → **4.2 flat** | host-bound |
| `smcp3` (per particle-step) | 1→150 | 6.9 → **2.2 flat** | host-bound (no batched counterpart exists — genmlx-im8n option d) |

### Batched paths — all amortizing; total is CONSTANT out to N=3000

| path | total ms, N=1→3000 | ms/particle at N=3000 | speedup vs scalar at 3000 |
|---|---|---|---|
| `vgenerate` | 5.8 → 5.6 (flat total) | 0.0019 | ~1300× |
| `vectorized-importance-sampling` | 6.4 → 5.6 | 0.0019 | ~2000× (vs extrapolated scalar) |
| `vmh` (10 sweeps) | 66 → 74 | 0.0025 /chain-step | ~24,000× /chain-step |
| `vsmc` (5 steps) | 17 → 32 | 0.0022 /particle-step | ~1900× |

Totals not growing at all to N=3000 means the batched paths are **host-floor
bound**: the GPU has so much headroom that particle count is free. The
crossover where the GPU becomes the constraint is far above 3000 on a model
this size.

### GPU-vs-host split of the batched floor (vgenerate, N=3000)

| phase | ms |
|---|---|
| graph build (host) | 3.7 |
| force eval (GPU) | 1.3 |

Even the **good** path is ~74% host graph-construction. Consistent with
`compile-fn` being an identity pass-through: the lazy graph is rebuilt
node-by-node from CLJS on every call.

## sm_120 (RTX PRO 6000 Blackwell, 2026-07-28, genmlx 6e2fc61 / mlx-node 722bf55e / mlx a27ddcaef, bun 1.3.14)

Run strictly serial, nothing else on the GPU (48 MiB idle before start).

**Read the code pin, not just the arch.** This column was measured at 6e2fc61,
which is FOUR commits past the sm_110 column's d463384 — it includes k1z7 +
ugq9 (the tidy cadences), ke97 (`vsmcp3`) and y3ls (batched splice). So the
last two rows of the batched table below have no sm_110 counterpart yet: at
d463384 `vsmcp3` did not exist, which is why the sm_110 smcp3 row still reads
"no batched counterpart exists". Cross-arch differences in the scalar rows are
confounded with those four commits and should not be read as pure arch effects
until Thor re-runs at this pin.

### Anchors (harness credibility)

| anchor | im8n reference | measured here | verdict |
|---|---|---|---|
| membrane micro-latency | ~1 ms/eval | 0.0169 ms/eval (tiny add+eval!) | credible — same sub-ms order as sm_110's 0.055; im8n's ~1 ms was a model-sized graph, not a membrane roundtrip |
| vgenerate N=3000 | ~21 ms (metareasoner world model) | 1.6 ms (5-site model) | reproduces — smaller model, right direction, ~3.5× under sm_110's 5.6 ms |

### Scalar paths — ALL host-bound (ms/particle does not fall in N)

| path | N sweep | ms/particle(-step) | verdict |
|---|---|---|---|
| `p/generate` ×N loop | 1→3000 | 4.85 → **4.94** | host-bound (dips to 1.09 at N=10, then climbs back — fixed-cost noise, no amortization) |
| `importance-sampling` | 1→1000 | 2.52 → **4.14** | host-bound |
| `mh` (per chain-step) | 1→20 chains | 3.92 → **4.95** | host-bound; post-k1z7, so directly comparable to sm_110's 5.66–11.1 |
| `smc` (per particle-step) | 1→500 | 18.2 → **3.64** | host-bound |
| `smcp3` (per particle-step) | 1→150 | 5.23 → **6.00** | host-bound |

### Batched paths — all amortizing; total is CONSTANT out to N=3000

| path | total ms, N=1→3000 | ms/particle at N=3000 | speedup vs scalar at 3000 |
|---|---|---|---|
| `vgenerate` | 5.2 → 1.6 (flat total) | 0.0005 | ~9,900× |
| `vectorized-importance-sampling` | 5.1 → 1.8 | 0.0006 | ~7,000× (vs extrapolated scalar) |
| `vmh` (10 sweeps) | 28.8 → 35.7 | 0.0012 /chain-step | ~4,100× /chain-step |
| `vsmc` (5 steps) | 17.5 → 18.6 | 0.0012 /particle-step | ~3,000× |
| `vsmcp3` (3 steps) | 14.6 → 8.2 | 0.0009 /particle-step | ~6,700× — **no sm_110 row** (path is newer than that column) |
| spliced-Map plate (M=5) | 1.8 → 9.3 | 0.0031 | **no sm_110 row** (genmlx-y3ls fast path is newer) |

CAVEAT on the multipliers: the batched totals at N=3000 are 1.6–1.8 ms for the
generate/importance rows, which is BELOW this harness's own anchor cost (3.4 ms
for 200 tiny evals). At that scale timer noise is a material fraction, and the
totals are non-monotonic in N (vgenerate: 5.2 → 1.8 → 1.5 → 4.0 → 1.6 ms).
Read those speedups as order-of-magnitude, not as measurements. The robust
claim is the one the shape supports: **total wall-clock does not grow with N**,
so particle count is free on this card up to at least 3000.

### GPU-vs-host split of the batched floor (vgenerate, N=3000)

| phase | ms |
|---|---|
| graph build (host) | 1.2 |
| force eval (GPU) | 0.2 |

**~86% host** — even more host-dominated than sm_110's 74%, and in the expected
direction: the discrete card evaluates the graph faster while CLJS-side graph
construction costs the same. This sharpens rather than softens the `compile-fn`
verdict: making the GPU faster moves the ratio further toward the host.

## Metal (Apple M2 Max 64GB, 2026-07-28, genmlx c1543f8 / mlx-node 722bf55e / mlx a27ddcaef, bun 1.3.14)

Run strictly serial, GPU otherwise idle. `c1543f8` is **two docs-only commits**
past the sm_110/sm_120 columns' `6e2fc61` — the bench code and the mlx-node/mlx
pins are identical — so this column is directly comparable with no code confound.

Mind the box: this is the **M2 Max**, not the **M4 mini** that
`docs/metal-test-triage.md` tracks. Host-bound paths follow the host, so the two
Macs are separate columns; the M4 mini can get its own later.

### Anchors (harness credibility)

| anchor | im8n reference | measured here | verdict |
|---|---|---|---|
| membrane micro-latency | ~1 ms/eval | 0.188 ms/eval (tiny add+eval!) | credible — but the HIGHEST of the three arches (sm_110 0.105, sm_120 0.0169). Metal's per-dispatch command-buffer latency is genuinely larger than either CUDA card |
| vgenerate N=3000 | ~21 ms (metareasoner world model) | 1.9 ms (5-site model) | reproduces — smaller model, right order |

### Scalar paths — ALL host-bound (ms/particle does not fall in N)

| path | N sweep | ms/particle(-step) | verdict |
|---|---|---|---|
| `p/generate` ×N loop | 1→3000 | 1.51 → **1.86** | host-bound; the FASTEST scalar generate of the three arches |
| `importance-sampling` | 1→1000 | 1.69 → **2.55** | host-bound (slight climb, no amortization) |
| `mh` (per chain-step, S=10) | 1→20 chains | 4.96 → **4.96 flat** | host-bound; comparable to sm_120's ~5.0 |
| `smc` (per particle-step) | 1→500 | 24.5 (N=1 warmup) → **2.43 flat** | host-bound (2.4 flat past the fixed first-call cost) |
| `smcp3` (per particle-step) | 1→150 | 3.55 → **1.77 flat** | host-bound |

### Batched paths — all amortizing; total is CONSTANT out to N=3000

| path | total ms, N=1→3000 | ms/particle at N=3000 |
|---|---|---|
| `vgenerate` | 1.9 → 1.9 (flat total) | 0.0006 |
| `vectorized-importance` | 1.7 → 1.9 | 0.0006 |
| `vmh` (10 sweeps) | 23.9 → 23.7 | 0.0008 /chain-step |
| `vsmc` (5 steps) | 10.9 → 12.5 | 0.0008 /particle-step |
| `vsmcp3` (3 steps) | 6.2 → 9.8 | 0.0011 /particle-step |
| spliced-Map plate (M=5) | 2.2 → 2.1 | 0.0007 |

Same shape as both CUDA arches — every batched total is flat to N=3000, so
particle count is free on this model. The sm_120 multiplier caveat applies MORE
strongly here: the batched totals (1.9 ms) sit far below the membrane anchor
(37.7 ms for 200 evals), so timer noise is a large fraction. Read the speedups
as order-of-magnitude; the robust claim is the shape (total does not grow in N).

### GPU-vs-host split of the batched floor (vgenerate, N=3000)

| phase | ms |
|---|---|
| graph build (host) | 0.9 |
| force eval (GPU) | 0.6 |

**~60% host** — the LEAST host-dominated of the three (sm_110 74%, sm_120 86%),
in the expected direction: Metal's GPU eval is the slowest of the three, so it
takes relatively more of the (sub-ms, noisy) total. Same `compile-fn` verdict —
the split is noise-dominated at this scale, read as a band, not a point.

### mcmc-family buffer-wall check (genmlx-sv3z's Metal-specific obligation)

The k1z7/ugq9 tidy cadences (mod-5 on NUTS paths, mod-25 elsewhere) were SIZED
against Metal's ~499K buffer-count wall from **Thor-side reasoning**, so Metal
had to confirm them directly. Re-ran the family solo on this box —
`mcmc_defaults`, `mcmc_detailed_balance`, `adaptive_hmc`, `adaptive_nuts`,
`loop_compiled_hmc`, `loop_compiled_mala`, `inference_hmc`,
`nuts_noncentered_funnel`, `fused_mcmc`, `mcmc_diagnostics`,
`inference_convergence`:

**10/11 green, with zero resource exhaustion / SIGTRAP / crash on any file** —
the cadences hold. The 3000-step fused linreg chain, sized "exactly on the
~499000 wall" (`fused_mcmc_test:241`), passes. **The buffer wall is confirmed
comfortable on Metal.**

The lone red is `fused_mcmc_test`, and it is **not a buffer breach and not
introduced by this run** — two absolute wall-**clock** perf budgets:
`fused < 200 ms` (measured **300 ms**) and `cached vectorized < 500 ms`
(measured **977 ms**). This is the known resynced-pin fused-eval buffer-
**retention** regression tracked in **`genmlx-rsgr`** and logged for the M4 mini
in `docs/metal-test-triage.md` (the mini measured 618–637 ms on the 500 ms
budget). The M2 Max is slower than the mini, so it trips **both** budgets where
the mini tripped only one. Per the triage ledger: **do NOT bump the budget** —
the fix is the buffer-retention root cause (`genmlx-rsgr`), not the tolerance.

## Findings beyond the curves

1. **Scalar MH's 61.6 ms/step was NOT inherent — root-caused and FIXED
   (genmlx-k1z7, 2026-07-28).** The bean's original suspects (regenerate
   machinery, `assert-joint!`, Trace rebuild) were all exonerated by profiling:
   a bare `mh-step` chain runs at 4.4–6.9 ms/step. The 49 ms/step was
   `collect-samples` wrapping **every** step in `u/tidy-step`, whose depth-0
   `mx/tidy` exit runs `jsc-cleanup!` — a **full synchronous Bun/JSC GC**
   (`mlx.cljs:1058`) — per MH step. Fix: `:tidy-every` cadence in
   `collect-samples` (default 25, plus one cleanup at loop exit). Measured
   after: 11.1 ms/step for S=10 chains (exit-GC amortized over few steps),
   5.66 ms/step over 2000 steps, memory bounded (0.1 MB peak), fixed-key
   samples bit-identical. Benefits every `collect-samples` client (mh,
   mh-custom, gibbs, involutive-mh, mala, hmc, run-kernel, pmcmc). The same
   tax may hit drivers with their own loops — audit filed as genmlx-ugq9.
2. **`mx/item` sits on the per-iteration path in scalar MCMC, structurally.**
   `mh-step` (`mcmc.cljs:64`) does `mx/realize` on the weight every step;
   `mh-custom` does 2 `mx/item`/step (`mcmc.cljs:202`). Structural because the
   accept/reject **branch** picks between host Trace values, so the decision
   must round-trip. The batched path keeps the decision on-device via
   `mx/where`/mask-merge — that is the remedy, already built.
3. **Silent scalar fallback census.** `IBatchedSplice` is implemented by
   Unfold, Switch, Scan, Mix. **Missing: Map, Mask, Recurse, ContramapGF,
   MapRetvalGF** — splicing any of those under vsimulate/vgenerate drops to
   `combinator-batched-fallback` (`handler.cljs:596`), an N-fold host `mapv`,
   with **no warning, counter, or dev-mode signal** (`runtime.cljs:145`).
   Wrapping a batched-capable combinator in `contramap`/`map-retval` silently
   loses the fast path.
4. **Controls behaved as predicted** (vgenerate / vmh / vsmc all
   amortizing), so the diagnostic isn't confirmation-biased. The
   cone-restricted regenerate control was not re-run here: its own bench
   (`bench/cone_regen_bench.cljs`, genmlx-ltx2) already records 155×/114× and
   its axis is sites-T, not particles-N.

## Ranked offender list → sub-beans under genmlx-819v

| rank | offender | measured | fix shape |
|---|---|---|---|
| 1 | ~~scalar `mh` step overhead~~ **FIXED** | 61.6 → 5.66–11.1 ms/step | was a full JSC GC per step in `collect-samples`; `:tidy-every 25` cadence (genmlx-k1z7) |
| 1b | ~~per-iteration depth-0 tidy exits in other drivers~~ **FIXED** | see table below | light tidy variants + per-loop cadences (genmlx-ugq9) |
| 2 | SMCP3 particle loop | 2.2 ms/p-step flat | vectorize (im8n option d); vsmc equivalent is 0.0022 (genmlx-ke97) |
| 3 | ~~silent batched-splice fallback~~ **FIXED** | spliced Map: 1007.9 ms → 3.9 ms at N=500 (~255×) | Map/Mask IBatchedSplice + contramap/map-retval delegation + dev-mode notice + inspect eligibility (genmlx-y3ls); Recurse documented inherently scalar |
| 4 | batched host floor | 74% of vgenerate wall is graph build | per-call graph rebuild; L4/fused territory — document, don't chase per-op |

## The tidy-exit GC purge (genmlx-k1z7 + genmlx-ugq9, 2026-07-28)

`mx/tidy`'s depth-0 exit is a full synchronous Bun/JSC GC (~49 ms on Thor).
Every scalar driver paid it per iteration — or, for NUTS, per tree node.
Fixed via `mx/tidy-run-light`/`mx/tidy-materialize-light` (eval-and-return, no
unconditional GC) + explicit GC cadences (mod-25 in the compiled/MAP loops,
mod-5 on NUTS paths because a NUTS step builds ~10K arrays). Measured on the
audit harness (Thor sm_110, fixed keys bit-identical old-vs-new):

| driver | GCs/unit before → after | wall before → after | speedup |
|---|---|---|---|
| mala S=25 | 1.08 → 0.12 /sample | 1641 → 299 ms | 5.5× |
| hmc S=25 L=10 | 1.08 → 0.12 /sample | 4944 → 3738 ms | 1.3× (leapfrog-dominated) |
| nuts S=10 | 3.50 → 0.30 /sample | 2279 → 678 ms | 3.4× |
| nuts S=10 +adapt | 4.80 → 0.50 /sample | 3262 → 824 ms | 4.0× |
| tidy-importance N=100 | 1.00 → 0.11 /particle | 5589 → 1004 ms | 5.6× |

Memory stayed bounded where it matters most: adaptive NUTS, 200 steps of
~10K-array trees at cadence 5 → 0.00 MB active after, 0.06 MB peak.

## The compile-fn verdict (genmlx-819v, 2026-07-28)

**`mx/compile-fn` stays an identity pass-through, permanently — at this
layer.** This is a decision, not an omission. Grounds:

1. **The GFI execution model violates `mx::compile`'s contract.** MLX's
   compile traces a pure array-in/array-out function and replays the cached
   graph. Handler execution is exactly not that: every `trace` op consults
   handler state (constraints, selections, old choices) per address, PRNG
   keys thread through host metadata, and addresses can be data-dependent.
   Tracing through it either fails or — the documented failure mode — severs
   the autograd tape at an interior `eval!` and returns **silent zero
   gradients**. The risk asymmetry is decisive: the upside is microseconds,
   the downside is a wrong-answer class of bug.
2. **The measured bottleneck is host structure-decision, which compile does
   not remove.** Everything this milestone measured and fixed was interpreter
   dispatch, per-step GC, and graph construction — ms-scale host work.
   Kernel-launch overhead (what fusion buys back) is 5–10 µs/op, two orders
   below it. Even the batched floor's 74% host share is graph *build* for a
   structure the interpreter must re-decide per call; `mx::compile` only
   helps when the structure is fixed — and…
3. **…where the structure IS fixed, GenMLX already compiles — its own way.**
   The ladder (noise transforms + expression compiler at L1, fused combinator
   loops at M5, single fused graphs at L4, fused-vmh cone runners) hoists
   graph construction out of loops on the CLJS side, and the Rust layer runs
   natively compiled/fused forwards for the LLM paths (its own graph-exec
   caching). Compiled execution exists where it belongs: below the membrane,
   on stable, fixed-shape functions. Duplicating it at the membrane would
   compose *against* the handler (principle 6).
4. **Amortization already wins at the scales that matter.** The batched
   curves are constant-total out to N=3000 (5.6–74 ms); the residual per-call
   build cost is noise per particle. A compile cache keyed on function
   identity would additionally be fragile under SCI, where closures are
   re-created per call.

**What would reopen this verdict** (it is permanent, not unfalsifiable):
- a profile of a batched hot loop where *eval* time dominates *build* time at
  fixed N through many small ops — fusion's actual regime;
- MLX gaining compile support for stateful/traced closures;
- leaving SCI for compiled ClojureScript, making function identity stable
  enough to key a compile cache.

Until one of those happens, "compile" work belongs to the ladder (L4 fused
graphs), not to `mx/compile-fn`.

## Consolidated per-arch table (genmlx-sv3z)

The POST-milestone state — one fresh full bench run per arch at (or after)
genmlx `6e2fc61`, so columns are comparable. The sections above are the
historical inventory + before/after record and the per-arch detail (the
sm_120 caveats live there — its batched totals are timer-noise-scale, read
those speedups as order-of-magnitude). This table is the current state.
`bun run --bun nbb cost_per_particle.cljs` from bench/, strictly serial.

| measurement | sm_110 (Thor) | Metal (M2 Max) | sm_120 (RTX) |
|---|---|---|---|
| provenance | 2026-07-28, 6e2fc61 | 2026-07-28, c1543f8¹ | 2026-07-28, 6e2fc61 |
| membrane eval (tiny add+eval!) | 0.105 ms (0.05–0.11 across runs) | 0.188 ms | 0.0169 ms |
| scalar `p/generate` loop | 1.8–2.7 ms/p flat | 1.5–1.9 ms/p flat | ~4.9 ms/p flat |
| scalar `importance-sampling` | 3.8–3.9 ms/p flat | 1.7–2.5 ms/p flat | ~4.1 ms/p flat |
| scalar `mh` (post-k1z7, S=10) | 11.1 ms/chain-step; 5.66 long-chain | ~5.0 ms/chain-step flat | ~5.0 ms/chain-step |
| scalar `smc` | 4.2 ms/p-step flat | 2.4 ms/p-step flat | ~3.6 ms/p-step flat |
| scalar `smcp3` | 2.1–2.2 ms/p-step flat | 1.8 ms/p-step flat | ~6.0 ms/p-step flat |
| `vgenerate` N=3000 | 6.8 ms total → 0.0023 ms/p | 1.9 ms → 0.0006 | 1.6 ms → 0.0005 |
| — build/eval split at N=3000 | 4.4 host / 1.4 GPU (76% host) | 0.9 / 0.6 (~60% host) | 1.2 / 0.2 (86% host) |
| `vectorized-importance` N=3000 | 5.7 ms → 0.0019 | 1.9 ms → 0.0006 | 1.8 ms → 0.0006 |
| `vmh` N=3000 (10 sweeps) | 70.9 ms → 0.0024 /chain-step | 23.7 ms → 0.0008 | 35.7 ms → 0.0012 |
| `vsmc` N=3000 (5 steps) | 34.0 ms → 0.0023 /p-step | 12.5 ms → 0.0008 | 18.6 ms → 0.0012 |
| `vsmcp3` N=3000 (3 steps) | 21.7 ms → 0.0024 /p-step | 9.8 ms → 0.0011 | 8.2 ms → 0.0009 |
| spliced-Map plate N=3000 | 5.7 ms → 0.0019 /p | 2.1 ms → 0.0007 | 9.3 ms → 0.0031 |

¹ Metal's `c1543f8` is two DOCS-ONLY commits past the CUDA columns' `6e2fc61`
(the sm_110/sm_120 doc-column writes themselves); bench code + mlx-node/mlx pins
are identical, so all three columns are code-comparable.

All three columns are now at effectively the same pin (Metal's c1543f8 is two
docs-only commits past the CUDA columns' 6e2fc61), so cross-arch scalar-row
differences are host-side arch/toolchain effects, not code confounds —
consistent with these paths being host-bound. Notable reads: Thor's scalar
generate/smcp3 are FASTER than RTX (2.4 vs 4.9, 2.2 vs 6.0 ms/p) despite the
slower GPU; Metal's scalar generate is the FASTEST of the three (1.5–1.9 ms/p)
even though its membrane per-eval latency is the HIGHEST (0.188 vs 0.105 vs
0.0169 ms) — not a contradiction, because scalar generate is dominated by host
GFI dispatch, not by the single eval. The batched host-fraction ranks with GPU
speed exactly as predicted: Metal 60% < sm_110 74% < sm_120 86% (faster card ⇒
more host-dominated). No cross-arch reds in the numeric sense — nothing for the
numeric-bounds table.

Metal's extra obligation (genmlx-sv3z) is **discharged**: the mcmc family
re-ran 10/11 green with zero resource exhaustion, so the k1z7/ugq9 tidy cadences
hold under Metal's ~499K buffer wall (the 3000-step fused chain sized on that
wall passes). The lone red is `fused_mcmc_test`'s two wall-CLOCK perf budgets
(300 vs 200 ms; 977 vs 500 ms) — the known resynced-pin buffer-RETENTION
regression tracked in `genmlx-rsgr` (M4 mini logged at 618–637 ms in
`docs/metal-test-triage.md`; the M2 Max is slower and trips both). Not a buffer
breach; do not bump the budget. See the Metal section above for the full run.
