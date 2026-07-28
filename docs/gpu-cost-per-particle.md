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
| `mh` (per chain-step) | 1→20 chains | **61.0–61.6 dead flat** | host-bound, **worst offender by 25×** |
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

## Findings beyond the curves

1. **Scalar MH's 61.6 ms/step is NOT inherent.** `vmh` at N=1 — same model,
   same posterior move — costs 6.6 ms/step: 9.4× less doing strictly more
   general work. ~55 ms/step of the scalar path is removable host overhead
   (regenerate machinery + per-step Trace rebuild + `assert-joint!` +
   `mx/realize` sync), not math.
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
| 1 | scalar `mh` step overhead | 61.6 ms/step vs 6.6 batched-N=1 | profile & strip the 55 ms; or route scalar mh through vmh N=1 |
| 2 | SMCP3 particle loop | 2.2 ms/p-step flat | vectorize (im8n option d); vsmc equivalent is 0.0022 |
| 3 | silent batched-splice fallback | N-fold host loop, no signal | emit signal + implement IBatchedSplice for Map first |
| 4 | batched host floor | 74% of vgenerate wall is graph build | per-call graph rebuild; L4/fused territory — document, don't chase per-op |

## Other arches

| | Metal (M4) | sm_120 (RTX PRO 6000) |
|---|---|---|
| status | not yet run | not yet run |

Run `bun run --bun nbb bench/cost_per_particle.cljs` and record here.
