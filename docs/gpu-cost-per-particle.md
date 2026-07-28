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

## Other arches

| | Metal (M4) | sm_120 (RTX PRO 6000) |
|---|---|---|
| status | not yet run | not yet run |

Run `bun run --bun nbb bench/cost_per_particle.cljs` and record here.
