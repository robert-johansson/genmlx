# GenJAX parity — keeper tables

North star (bean `genmlx-ecsi`): every GenMLX algorithm at least as fast as its
GenJAX/JAX equivalent on the RTX PRO 6000 (sm_120). This doc holds the
**distilled, pinned gap tables**; everything else — specs (source of truth),
runners, raw results, orchestrator — lives in the sibling repo
`~/mlx/genjax-parity` (see its README for the two-repo contract and fidelity
rules). The GenMLX side is `bench/genjax_parity.cljs` (bench tier: strictly
serial, card otherwise idle).

Method, both sides identically: shared JSON spec, float32, literal frozen data,
per-run pre-generated keys, device-synced timing, warmup reported separately
(never amortized), median/p10/p90 over 20 runs. Always-on weight-convention
guard: the generate weight must equal the hand-computed log-likelihood of the
constraints (caught genjax 1.0's per-component weight vectors for broadcast
sites; model uses one scalar site per observation on both sides).

## linreg_is — importance sampling (prior proposal), Bayesian linear regression, 10 obs

sm_120, measured 2026-07-28. Pins: genmlx `d4cca33` (src tree; bench file
landed the following commit), mlx-node `722bf55e`, genjax 1.0.13 (git,
femtomc/genjax), jax 0.7.2 + cuda13 wheels (nvcc 13.3.73 — same toolkit
version mlx-node builds against), driver 595.71.05. Raw results + spec in the
parity repo (`results/linreg_is.*.20260728-*.json`).

| N | GenJAX median | GenMLX median | steady gap | GenJAX warmup | GenMLX warmup |
|---|---|---|---|---|---|
| 1 | 0.018 ms | 1.76 ms | 98x | 326 ms | 4.0 ms |
| 100 | 0.019 ms | 1.77 ms | 96x | 376 ms | 6.7 ms |
| 3000 | 0.020 ms | 1.84 ms | 94x | 351 ms | 2.1 ms |
| 30000 | 0.019 ms | 2.08 ms | 110x | 356 ms | 1.5 ms |

logZ agreement at N=30000: −18.72 (GenJAX) vs −18.76 (GenMLX) — MC noise;
both sides estimate the same joint. (GenMLX N=3000 row reproduces the
cost-per-particle table's vgenerate ~1.6–1.8 ms host floor.)

### Post-Phase-2b (vgenerate-compiled — genmlx-vjnn, 2026-07-28)

The batched handler run traced once through the persistent CompiledFn and
replayed per call (`dyn/vgenerate-compiled`; key splits kept lazy and
single-stream while tracing — mlx-node `b2290f3`). Second measurement:

| N | GenJAX | GenMLX handler | **GenMLX compiled** | gap now |
|---|---|---|---|---|
| any of 1–30000 | 0.019 ms | 1.8–2.1 ms | **0.52 ms** (p10–p90 0.47–0.55) | **~27x** |

logZ is bit-identical to the handler path (same keys → same samples through
the traced graph) — the strongest equivalence evidence available.

**The warmup ledger inverted.** After the one-time NVRTC kernel bill
(~11 s first-ever per box, disk-cached), a NEW SHAPE costs GenMLX
1.4–1.6 ms to trace — one SCI handler run — versus JAX's ~350 ms jit.
Below ~700 same-shape calls, GenMLX's TOTAL wall now beats GenJAX on this
row despite the 27x steady-state deficit.

Reading the residual 27x: the traced graph carries the key-derivation ops
(~13 threefry splits) and 4 outputs, plus the per-call CLJS wrapper
(choicemap rebuild).

**Factoring experiment (genmlx-agcp, same day): measured and REJECTED.**
Three factorings of the same sweep, same protocol:

| factoring | median/call |
|---|---|
| everything-in-graph (key-traced) | **0.52 ms** |
| split ladder hoisted out (subkeys as inputs) | 0.79–1.32 ms |
| splits + sampling hoisted out (noise as inputs) | 1.00–1.45 ms |

On MLX, every op moved OUT of the traced graph costs more in per-call
dispatch/sync round-trips than the fused in-graph version costs — the
Phase-0 probe's 65 µs floor pre-generated its inputs OUTSIDE the timed
region (instrument artifact; recorded). Everything-in-graph stands as the
optimal factoring; the remaining levers for this row are engine-side
(replay kernel count / CUDA-graph capture) and the CLJS wrapper.

Keepers from the experiment: (1) not every vgenerate-able model is
traceable — rejection samplers (gamma's Marsaglia–Tsang loop) call
`mx/item` per draw, the hard boundary of fixed-structure compilation;
(2) `vgenerate-compiled` now degrades LOUDLY and permanently to the plain
handler on that error instead of failing (y3ls doctrine), with test
coverage.

**Reading the row.** Both sides are N-independent to 30000 — the GPU is idle
in both worlds; the contest is host floors. GenJAX's floor is one fused XLA
executable launch (~19 µs); GenMLX's is per-call lazy-graph construction in
nbb/SCI (~1.8 ms, 86% host graph-build per genmlx-sv3z). This is the
compile-fn verdict's predicted gap, now measured: **~95x steady-state** on a
microbenchmark-sized model. The other side of the ledger is warmup: JAX pays
~350 ms jit compile **per shape**, GenMLX ~nothing, so **GenMLX's total wall
is lower until ~173 same-shape calls**. Neither number is the whole story;
report both.

**Findings filed from this row:**
- The steady-state gap's cause is the already-diagnosed host graph-build floor
  (milestone genmlx-819v, compile-fn verdict + its three reopen triggers) —
  follow-up bean tracks whether/how to attack it (see genmlx-ecsi children).
- L3 analytical elimination did NOT trigger on the spec-driven model: the
  bench builds dist params from closure variables, not literals, and the
  unstripped generate weight stayed key-dependent. The benched path strips
  analytical anyway (the spec pins the ALGORITHM), but detection silently
  failing on closure params is a real conjugacy-coverage gap (cf. the
  genmlx-7zuq class).

### Post-Phase-3.5 (wrapper hoist + captured-exec replay — genmlx-w9mz/7prh, 2026-07-28)

Two levers landed the same night, driven by a stage profile of the 0.52 ms
call (genmlx-w9mz: ~0.19 ms SCI wrapper + ~0.31 ms replay/eval, of which
67 us is the FIXED eval! floor and ~5 us/op the scheduler walk):

1. **Wrapper hoist** (genmlx 99f4cec): per-call-invariant work moved to
   factory time (transient choicemap rebuild, pre-tagged trace template,
   raw-array NAPI returns, shape-raw key check). 0.52 -> 0.42 ms.
2. **Captured-exec replay** (genmlx-7prh, the engine-side lever): the
   persistent CompiledFn's FIRST call now captures its eval into retained
   CUDA graph execs with retained buffers; every later call memcpys the key
   into a staged buffer, relaunches the retained execs, and returns
   evaluated output copies — no tape clone, no per-op scheduler walk, no
   separate eval round-trip. Found and fixed en route: the CUDA fork's
   Buffer::raw_ptr() migrates device-pool buffers to unified memory on any
   host read, orphaning (and use-after-freeing) the captured params — the
   capture paths now use the no-migrate gpu_ptr accessor.

| N | GenJAX | pre (2b) | **post-3.5** | gap now |
|---|---|---|---|---|
| any of 1–30000 | 0.019 ms | 0.52 ms | **0.29 ms** (p10 0.28) | **~15x** |

logZ at N=30000: −18.76 (unchanged, MC noise vs GenJAX −18.72). Equivalence
pinned by capture_replay_test (same-key bitwise vs the plain replay,
aliasing tripwire, shape-drift fallback) and the untouched
vgenerate_compiled_test suite. Residual 0.29 ms: ~0.12 ms SCI wrapper
(ensure-key + rebuild + record assoc) + ~0.15 ms captured call (memcpy +
launches + sync + 4 output-copy allocations). Next levers: per-call output
clone allocation (double-buffer), the remaining SCI wrapper, or accept —
the row is now within 2x of the pip-MLX Phase-0 floor measurement.

### Floor decomposition + budget (genmlx-zco8/pqb5, 2026-07-29)

Measure-first probe of the captured-call floor on minimal graphs: a
3-in/4-out captured call costs **0.0295 ms** — per-output-clone ~5.1 µs,
per-input staging ~0.4 µs. The bean's double-buffer and batched-memcpy
levers were therefore REJECTED with numbers (combined ceiling ~20 µs of a
~290 µs call); the residual is ~2/3 SCI wrapper + ~1/3 the real graph's
launches — the 819v compile-fn boundary. Fresh row stamp (post-family/
fusion binary): 0.289–0.297 ms at every N, logZ agreement intact, and the
bench now also prints the EXACT analytical logZ (−18.6097, L3
elimination on the literal-param model — the kzoy anchor). Robert
approved **0.35 ms/captured-call** as the floor-regime budget
(genmlx-pqb5); the orchestrator asserts it as an absolute ceiling for
floor-bound cells.

## linreg_mala — single-chain joint MALA, same posterior, sweep over chain length

sm_120, measured 2026-07-28. Pins: genmlx `4518858` src tree (bench's mala
support landed the following commit), mlx-node `722bf55e`, genjax 1.0.13,
jax 0.7.2/cuda13, driver 595.71.05. GenMLX side: `mcmc/fused-mala` (whole
chain as one lazy graph — the scan analog), `:chain-fn` reused across calls,
device chosen by probe. Timed unit both sides: init (constrained generate,
latents from prior) + S-step chain, materialized. Algorithm identity
verified: joint Gen.jl-style MALA both sides, eps 0.08, acceptance 0.80–0.89
(GenJAX) vs 0.82 (GenMLX), slope-tails bracket the analytic posterior mean
1.9917 ± 0.11.

| S (steps) | GenJAX median | GenMLX median | steady gap | GenJAX warmup | GenMLX warmup | break-even calls |
|---|---|---|---|---|---|---|
| 10 | 0.21 ms | 18.2 ms | 87x | 1656 ms | 26.8 ms | ~91 |
| 100 | 1.62 ms | 197 ms | 122x | 1619 ms | 162 ms | ~7 |
| 1000 | 15.6 ms | 1748 ms | 112x | 1684 ms | 1813 ms | — (GenJAX wins, barely on warmup) |

(Post-genmlx-lmmn numbers. The first measurement of this row had S=1000 at
**6021 ms / 386x** with a 5.6 s warmup: `fused-mala` tripped its Metal-sized
graph fallback on the CUDA card. Making the fused limits per-backend and
probing the CUDA boundary — validated to 16000 ops with no cliff, see the
`fused-ops-limits` comment in `mcmc.cljs` — restored the fused path: 3.4x
faster at S=1000, acceptance reported for real (0.800, exactly matching
GenJAX's 0.80 at the same eps), and the row now shows one consistent
host-floor ratio across all S.)

**Reading the row.** A chain is sequential, so GenJAX cannot amortize
launches across steps the way vmap amortizes across particles: it pays its
floor per STEP (~15.6 µs/step, linear in S). GenMLX pays its floor per step
too — ~1.7–2.1 ms/step on the fused path — so the gap (~87–122x) is the
SAME host-floor ratio as the IS row, now applied per step. Warmup is where
MCMC differs from IS: tracing the chain costs JAX ~1.65 s per shape (10x its
IS warmup), so at S=100 GenMLX's total wall wins under ~7 same-shape calls.
Device probe: cpu 188.0 vs gpu 184.6 ms at S=100 — near-identical, the
host-bound signature.

### Post-Phase-2a (persistent compile wired — genmlx-0vwj, same day)

`persist-chain` (mcmc.cljs) now wraps the four fused chain builders in the
persistent CompiledFn (genmlx-z2gt Phase 1): first call traces, replays run
in C++. Gated per-method at MEASURED trace-depth boundaries (mh 1250 /
mala 2500 / hmc 2500 ops — MLX's recursive compile passes overflow the 8 MB
stack on deeper chains; past the gate = today's identity behavior; durable
fix beaned). Sampling behavior bit-identical (acceptances and slope-tails
unchanged). The row, third measurement:

| S | GenJAX | GenMLX pre-lmmn | post-lmmn | **post-2a** | gap now |
|---|---|---|---|---|---|
| 10 | 0.21 ms | 16.5 ms | 18.2 ms | **6.8 ms** | 33x |
| 100 | 1.62 ms | 213 ms | 197 ms | **23.5 ms** | 15x |
| 1000 | 15.6 ms | 6021 ms | 1748 ms | **229 ms** | **15x** |

The warmup ledger changed as predicted: GenMLX now pays trace+compile per
shape (0.2–6.0 s; order-dependent — the first shape in a process absorbs
the cold NVRTC kernel bill, disk-cached after) vs GenJAX's 1.6–1.7 s. At
S=100 the break-even is ~64 same-shape calls in GenMLX's favor.

Reading the residual 15x: S=1000 replay measures 229 ms vs the 116 ms
pip-MLX Phase-0 floor — the timed unit includes the SCALAR init generate
(~5 ms, SCI), per-call noise pre-generation, and result marshaling on top
of the raw replay. At S=10 those overheads ARE the number (6.8 ms for a
~0.1 ms replay). The next lever is therefore not the chain — it's the
per-call scaffolding around it (init + noise), which is Phase 2b territory
alongside the vgenerate/IS row.

**Findings from this row (both resolved 2026-07-28):**
- genmlx-lmmn (FIXED, above): fused limits were Metal-buffer-wall-sized and
  fired on CUDA; now per-backend with the CUDA boundary measured, tests
  backend-aware and fallback-forcing tests sized relative to the active
  limit.
- genmlx-d62h (root-caused): the "acceptance 0.000" was the bench coercing
  the fallback's honest `:acceptance-rate nil` through CLJS `+` (nil→0).
  Bench now reports null; `fused-mala`/`fused-hmc` docstrings state the nil.
  The block path itself still does not track acceptance.

### Post-Phase-3.5 (captured-exec replay — genmlx-7prh, 2026-07-28)

The chain replay's 223 ms at S=1000 split (genmlx-w9mz) as 58 ms tape clone
+ ~158 ms per-op eval walk + ~8 ms scaffolding — 97% exactly what the
captured-exec replay eliminates. With `persist-chain`/`persist-chain1` on
the captured path (first call captures the chain eval into ~hundreds of
retained CUDA graph execs at the 100-node commit cadence; replays are
memcpy + launches + sync):

| S | GenJAX | post-2a | **post-3.5** | gap now |
|---|---|---|---|---|
| 10 | 0.21 ms | 6.8 ms | 7.6 ms | 36x (scaffolding-bound) |
| 100 | 1.62 ms | 23.5 ms | **10.7 ms** | 6.6x |
| 1000 | 15.6 ms | 229 ms | **42.6 ms** | **2.7x** |

Acceptance 0.80–0.82 (GenJAX 0.80–0.89 at the same eps), slope-tails
bracket the analytic posterior mean — chain behavior bit-identical to the
pre-capture path (fused_mcmc + warmup-reproducibility pins unchanged).

Reading the residual: the S=1000 captured replay itself measures ~34 ms —
now plausibly EXECUTION-bound (~30k sequential tiny kernels; XLA fuses the
step body into far fewer) — with ~8 ms per-call scaffolding on top (scalar
init generate in SCI + noise pre-gen + marshaling, the h5wg remnant). At
S=10 the scaffolding IS the number.

### Post-3.5b (whole-call factory — fused-mala-compiled, genmlx 3815718, same night)

The h5wg scaffolding lever: `fused-mala-compiled` traces the ENTIRE
per-call unit (init constrained generate + start-point val-grad + noise
pre-gen + chain) as ONE captured graph of the PRNG key; replays are
launch-only. The stream layout replicates eager fused-mala exactly (same
chain; kernel-fusion rounding ~1e-6; repeat calls bit-exact). Found en
route: `u/extract-params`' scalar path calls mx/realize (eval-in-trace) —
the factory extracts lazily; and the first "bit-exact" test pass was
VACUOUS (the factory had silently degraded on that very realize — the
degrade note now prints the underlying cause). The bench rides the
factory on :gpu.

| S | GenJAX | post-2a | post-3.5 | **post-3.5b** | gap now |
|---|---|---|---|---|---|
| 10 | 0.21 ms | 6.8 ms | 7.6 ms | **0.67 ms** | **3.2x** |
| 100 | 1.62 ms | 23.5 ms | 10.7 ms | **4.40 ms** | **2.7x** |
| 1000 | 15.6 ms | 229 ms | 42.6 ms | **34.5 ms** | **2.2x** |

Acceptance 0.820/0.824/0.800 and slope-tails 2.156/2.038/2.050 —
IDENTICAL to the eager rows above. Warmup (trace+capture per shape):
42 ms / 239 ms / 7.0 s vs GenJAX's ~1.65 s at every S — GenMLX's total
wall now wins at S=10/100 from call one AND in steady state the gap is
2-3x everywhere. The row's remaining levers are all engine-side kernel
economics (fewer/larger kernels per step — XLA-class step fusion), plus
the eager adaptive-warmup path if a spec ever pins it.

### Post-family-scoring (vectorized family score — genmlx-yopl, 2026-07-29)

The census-driven lever that came out of the kzoy refutation: the
tensor-native score (and a new batched variant) now detects homogeneous
OBSERVED site families — same dist type, dist-arg source forms identical
up to numeric literals — and scores each family as ONE stacked [G]
log-prob (per-literal [G] constant columns, one arg graph, elementwise lp,
sum). The forward was already fused; the point is the VJP, which now
vectorizes to a handful of [G]-kernels instead of ~2 scalar backward
kernels + gather/scatter/squeeze plumbing per site. Census:
fused-mala-compiled S=100 tape 6553 → **5025** (~50/step; the 20 per-site
backward kernels/step are gone). Equivalence pinned by
`family_score_test` (tensor == handler score and grads to ~1e-7 rel;
batched == GFI batched; heterogeneous/subset declines) — scores match the
handler up to float32 summation order; chains statistically identical
(same acceptance/tails at every cell).

| S | GenJAX | post-3.5b | **post-family** | gap now | warmup now |
|---|---|---|---|---|---|
| 10 | 0.21 ms | 0.67 ms | **0.599 ms** | 2.9x | 23.9 ms |
| 100 | 1.62 ms | 4.40 ms | **4.04 ms** | 2.5x | 69.6 ms |
| 1000 | 15.6 ms | 34.5 ms | **26.2 ms** | **1.68x** | **1449 ms** |

### Post-reduce-fusion (fused Sum epilogues — genmlx-7dm0, 2026-07-29)

The engine half of the kernel-economics ladder: MLX's compile_fuse now
admits a float32 Sum as a fusion ROOT (full 1-D or last-axis 2-D domains;
CUDA-only, MLX_DISABLE_REDUCE_FUSION kill switch), pulling its exclusive
elementwise producer chain into one fused reduce kernel (block per row,
per-input stride pairs, cg block reduction — new FusedReduceKernelBuilder,
NVRTC-jitted). Fused == eager at ≤2.2e-7 rel (1-D bit-identical); guards
12/12 + the 10-suite MCMC/capture batch green, chains statistically
identical. Census: MALA tape 5025 → **4623** (~46/step).

| S | GenJAX | post-family | **post-fusion** | gap now |
|---|---|---|---|---|
| 10 | 0.21 ms | 0.599 ms | **0.576 ms** | 2.7x |
| 100 | 1.62 ms | 4.04 ms | **3.87 ms** | 2.4x |
| 1000 | 15.6 ms | 26.2 ms | **24.6 ms** | **1.58x** |

Warmup 23/79/1366 ms — still under jit at every S. Residual at S=1000:
~30 launches/step (proposal/accept machinery + per-eval gather/scatter
pairs + shared-producer Sums that multi-parent chains keep unfused) vs
XLA's ≤4 — the next levers are cross-shape fusion / redundant-producer
duplication (XLA-style) and gather/scatter elimination, recorded on the
milestone.

**GenMLX now wins warmup at every S on this row** (jit is ~1.65 s flat;
the family-scored whole-call trace is 24 ms–1.4 s) — the first row where
both total-wall AND per-shape warmup favor GenMLX while steady state is
under 2x. Residual at S=1000: ~36 launches/step (proposal/accept
machinery ~12-15, val-grad ~12-15 incl. 2 gathers + 2 scatters + 3
standalone Sums) vs XLA's ≤4 — the remaining lever is fusable Reduce +
view fusion (genmlx-7dm0). S=10/100 cells are scaffolding/floor-bound
(the 819v compile-fn boundary), not kernel-bound.

## linreg_hmc — single-chain joint HMC, same posterior, sweep over chain length × leapfrog steps

sm_120, measured 2026-07-29 (bean genmlx-klwf). Pins: genmlx `ed2be85` src
tree (bench's hmc support + the :hmc limit changes land in this doc's
commit), mlx-node `885c2ee`, genjax 1.0.13, jax 0.7.2/cuda13, driver
595.71.05. Both sides: identical algorithm — genjax's `inference.hmc` and
GenMLX's `fused-hmc` implement the same leapfrog scheme (merged half-kicks,
L+1 gradient evals per step), same MH criterion on −ΔH, identity mass,
N(0,1) momentum; eps 0.08 frozen in the spec after a genjax probe showed
non-saturated acceptance there (0.84–0.93). GenMLX rides the whole-call
factory (`fused-hmc-compiled`) from the start — this row is the first
measured END-TO-END on the post-3.5b machinery rather than iterated onto
it. Timed unit both sides: init (constrained generate, latents from prior)
+ S-step chain of L-leapfrog HMC, materialized.

| S | L | GenJAX median | GenMLX median | steady gap | GenJAX warmup | GenMLX warmup |
|---|---|---|---|---|---|---|
| 10 | 8 | 0.363 ms | 2.52 ms | 7.0x | 1404 ms | **235 ms** |
| 100 | 8 | 3.67 ms | 15.5 ms | 4.2x | 1436 ms | 4755 ms |
| 1000 | 8 | 32.6 ms | 198 ms | 6.1x | 1474 ms | 220 s |
| 10 | 32 | 1.06 ms | 5.50 ms | 5.2x | 1434 ms | 1429 ms |
| 100 | 32 | 10.5 ms | 66.3 ms | 6.3x | 1465 ms | 44.8 s |
| 1000 | 32 | 100.8 ms | **7234 ms** | **72x** * | 1487 ms | 16.3 s |

\* block-compiled fallback, not the captured factory — see below.

**Algorithm identity held tightly across every cell**: acceptance
0.995/1.000, 0.914/0.915, 0.902/0.900, 0.930/0.910, 0.836/0.841 (GenJAX/
GenMLX; the 72x cell's block path doesn't track acceptance — genmlx-d62h —
reported null). Slope-tails at S=1000: 1.997/2.001 (L=8) and 1.987/1.996
(L=32) against the analytic posterior mean 1.9917.

**Reading the steady gaps.** The five captured-factory cells sit at
4.2–7.0x — the MALA row's 2.2–3.2x scaled up by HMC's heavier per-step
kernel count (~L+1 gradient evals per step, each a multi-kernel
score+vjp subgraph; XLA fuses the step body into a handful of kernels).
Cause-bean: genmlx-lnzc (step fusion / kernel economics). The 72x cell is
S×L = 32000 > the 16000 CUDA fused limit: it falls to block-compiled HMC
at 226 µs/leapfrog vs the captured path's 25 µs/leapfrog — suspected
per-call re-trace of block handles plus per-block sync cadence.
Cause-bean: genmlx-dys7 (chunked captured chains / block-handle
persistence).

**The warmup story is this row's headline finding.** GenJAX's jit is
1.40–1.49 s FLAT at every cell — lax.scan keeps the XLA program
constant-size in S and L. GenMLX's whole-call trace cost is **quadratic
in graph size**: 800 ops → 4.8 s, 3200 → 44.8 s, 8000 → 220 s (t ∝ ops²
almost exactly), and the 32000-op trace was killed at a 30-minute wall
cap (extrapolation: ~58 min). GenMLX still wins warmup where traces are
small (S=10: 235 ms / 1429 ms vs ~1.4 s) — but past ~1000 ops the ledger
inverts hard. Per the user directive recorded on epic genmlx-1ixc
(2026-07-29), warmup gaps must approach zero too: the quadratic (an
accidental O(n²), not fundamental unroll cost, which would be linear) and
the missing loop/scan primitive are tracked as genmlx-geiw — the
highest-leverage item this row produced.

**Limits changed by this row** (`mcmc.cljs`): persist-trace-ops :hmc
2500 → 8000 (measured-OK whole-call traces at 3200/8000; no SIGSEGV — the
HMC boundary is trace TIME, not stack depth). CUDA fused :hmc stays
16000: 32000 was attempted and REJECTED on the 30-min trace DNF. Both
comments now state measured trace time as an acceptance criterion for any
future raise.

### Post-fix (compile_fuse quadratic killed + chunked chains — genmlx-geiw/dys7, same day)

Two levers landed hours after the row, driven by its findings:

1. **The trace quadratic, root-caused and fixed** (mlx fork `6c9d641`):
   gdb sampling put 11/15 profiles inside `mlx::core::detail::compile_fuse`
   — it rescanned each fusion input's FULL parents vector once per fusion,
   O(tape²/depth) for tape-shared inputs (step-size scalars, noise
   tensors). Fix: append-only input parents, pruning moved to merge time
   against `global_cache` (amortized linear). Chains bit-exact at the same
   keys; contract guards + full battery 434/434.
2. **Chunked captured chains** (genmlx `f15cc59`): past the 16000 fused
   limit, CUDA now decomposes the chain into persist-gate-sized chunks on
   captured replay instead of block-compiling — same chain to
   kernel-fusion rounding, reusable :chain-fn, and REAL acceptance (the
   d62h nil is gone on this path).

| S | L | GenJAX | pre | **post** | gap now | warmup pre → post |
|---|---|---|---|---|---|---|
| 100 | 8 | 3.67 ms | 15.5 ms | 16.2 ms | 4.4x | 4.8 s → **2.2 s** |
| 1000 | 8 | 32.6 ms | 198 ms | 199 ms | 6.1x | 220 s → **55.6 s** |
| 100 | 32 | 10.5 ms | 66.3 ms | 66.4 ms | 6.3x | 44.8 s → **11.1 s** |
| 1000 | 32 | 100.8 ms | 7234 ms | **690 ms** | **6.8x** | 16.3 s → 16.0 s |

Acceptance on the chunked cell: **0.821 vs GenJAX 0.825** — the cell's
first real cross-side agreement (it reported null before). All other
cells bit-identical to the first measurement. Reading the residuals:
warmups are ~4x cheaper everywhere but still ~n^1.7 above 8000 ops
(geiw stays open — suspects: simplify CSE grouping, capture exec
instantiation); the chunked cell now runs 21.5 µs/leapfrog — FASTER than the
single-graph captured path's 25 (dys7 residual CLOSED, same day: the
capture sink's input staging slow-paths VIEW-backed buffers at ~187 ms
per chunk — the raw_ptr migration hazard class — fixed by handing the
captured calls standalone slice copies; engine-side durable fix beaned). The
compile_fuse fix is upstream-relevant (ml-explore/mlx has the same
quadratic; genmlx-xz93 batch).

### Post-site-batching experiment (static-unrolled model — genmlx-kzoy, 2026-07-29)

The lnzc census pivot's cheap half, run and REFUTED. The bench's linreg
model switched from the doseq form to a GENERATED static-unrolled source
(`dyn/make-gen-fn` with per-site literal forms — the gen macro's own
expansion target; probe-verified `tensor-native? = true`, tensor score ==
handler score exactly, weight convention intact). MLX_COMPILE_DEBUG census
of the captured chain graphs, doseq vs unrolled:

| regime | doseq tape | unrolled tape |
|---|---|---|
| fused-mala-compiled S=100 | 6554 (~65/step) | 6553 |
| fused-vectorized-mala N=64 S=100 | 9256 (~92/step) | 9256 (byte-identical) |
| fused-hmc-compiled S=100 L=8 | 25219 | 25219 |

**The graph does not change.** The GFI-handler score and the tensor-native
score build the SAME lazy MLX graph, and post-geiw `compile_fuse` already
fuses the forward score into ONE Compiled kernel either way. The earlier
census reading "18 per-site score kernels" was a misattribution: the
per-site kernels in the tape are the BACKWARD (VJP) chains — ~10x
`CompiledMultiplyAddSubtractDivide` + ~10x `CompiledMultiplyDivideNegative`
per MALA step, plus Gather 4/step, Scatter-Sum 2/step, Squeeze 6/step,
ExpandDims 3/step — which mirror the per-site forward structure regardless
of which score path built it. Steady-state rows: unchanged (chains
bit-identical, same acceptance/tails). What DID move is **warmup** — the
tensor-native score is host-cheaper to trace: MALA S=1000 7.0 s → 3.9 s
(combined with geiw round 2), HMC S=1000×L=8 55.6 s → 36.9 s, 100×32
11.1 s → 9.3 s (post-geiw baselines). Bonus: literal dist params close the
genmlx-bg67 detection gap for this bench — the unstripped generate weight
is now key-independent (exact analytical logZ −18.6097 available as a
correctness anchor).

**Verdict:** score-PATH selection does nothing for steady state; ≤2x is
NOT reachable this way. The kernel-economics lever is re-routed to
emitting a VECTORIZED score graph — stack each homogeneous observed site
family into one [G]-shaped lp so the VJP vectorizes to a handful of
[G]-kernels (genmlx-yopl, re-scoped), then fusable-Reduce (genmlx-7dm0)
for the remaining Sums.

### Post-family-scoring (vectorized family score — genmlx-yopl, 2026-07-29)

Same lever as the MALA row (family-vectorized tensor score; census
25219 → 19407 at S=100 L=8 — the per-grad-eval backward now a handful of
[G]-kernels). Chains statistically identical (acceptance/tails equal).

| S | L | GenJAX | pre | **post-family** | gap now | warmup pre → post |
|---|---|---|---|---|---|---|
| 10 | 8 | 0.363 ms | 2.52 ms | **2.02 ms** | 5.6x | 235 → **63 ms** |
| 100 | 8 | 3.67 ms | 16.2 ms | **13.5 ms** | 3.7x | 2.2 s → **512 ms** |
| 1000 | 8 | 32.6 ms | 199 ms | **171 ms** | 5.2x | 36.9 s → 11.1 s |
| 10 | 32 | 1.06 ms | 5.50 ms | **5.99 ms** | 5.6x | 1.0 s → **329 ms** |
| 100 | 32 | 10.5 ms | 66.4 ms | **50.2 ms** | 4.8x | 9.3 s → 2.6 s |
| 1000 | 32 | 100.8 ms | 690 ms | **565 ms** | 5.6x | 13.2 s → 4.0 s |

GenMLX now wins warmup at 10×8, 100×8, 10×32; the 8000+ op cells (11.1 s,
2.6 s, 4.0 s vs jit's ~1.45 s) remain geiw's residual. The steady
residual is per-grad-eval plumbing the family lever cannot reach — per
step: ~2 gathers + 2 scatters + ~2 sums PER grad eval (× L+1 evals) plus
the 2L unfused integrator updates — squarely the fusable-Reduce +
view-fusion territory (genmlx-7dm0).

### Post-reduce-fusion + chunk routing (genmlx-7dm0 + geiw experiment, 2026-07-29)

Reduce fusion (see the MALA section) shaved every cell (census
19407 → 18107): 171 → **160.5** (4.9x), 50.2 → **46.4** (4.4x),
565 → **527** (5.2x chunked); chains identical. Then the
`GENMLX_HMC_CHUNK_OPS` routing experiment — route large chains through
the chunked captured runner with SMALL chunks instead of one whole-graph
trace:

| cell | whole-graph | **chunks=1000 ops** | GenJAX / jit |
|---|---|---|---|
| 1000×8 warmup | 11.1 s | **928 ms — UNDER jit** | 1.45 s |
| 1000×8 steady | 160.5 ms | **133.0 ms (4.1x)** | 32.6 ms |
| 1000×32 warmup | 4.0 s | 2.1 s | 1.49 s |
| 1000×32 steady | 527 ms | **448 ms (4.4x)** | 100.8 ms |
| 100×32 steady | 46.4 ms | 56.0 ms (worse) | 10.5 ms |

Chunked wins BOTH warmup and steady at large S (the small-chunk graphs
fuse better per-kernel and replay identically), but loses mid-size cells
where the whole-call factory's in-graph init (~10 ms/call, SCI) matters.
Chains bit-equal per-cell across routings (same acceptance/tails).
Reproduced with a warm NVRTC cache (second run): 1000×8 warmup 916 ms /
steady 134.4 ms; 100×32 warmup 3.24 s (the trace itself — NVRTC was not
the cost), 1000×32 2.09 s. So chunk routing puts 1000×8 durably under
jit; 100×32 and 1000×32 warmups remain over (accepted residuals, geiw).
Follow-up design recorded on geiw: compose the whole-call factory WITH
chunked internals ("capture-through") — the outer capture records the
inner chunks' replayed kernels, giving flat warmup AND in-graph init.

**The manychain analog was built and withdrawn the same day**
(genmlx-b1gx): a chunked captured N-chain MALA runner is bit-exact vs
the whole sweep with capture disabled, but captured REPLAYS of the
[N,D] chunk graph compute wrong post-seam dynamics (staged inputs
verified correct by readback; kernels reference staged buffers; the
identical clone→copy_into cycle works for 1-D HMC chunks and for
whole-graph [N,D] replays). Until that capture-layer bug is closed, the
manychain S=1000 warmup (~2.8–3.1 s vs jit ~1.6–1.9 s) stays an
accepted residual; the withdrawn runner had measured 177–184 ms.

## linreg_mala_manychain — N-chain vectorized MALA, the decisive amortization regime

sm_120, measured 2026-07-29 (bean genmlx-zebd). Pins: genmlx `244aebd` src
tree (fused-vectorized-mala + bench mode land in this doc's commit),
mlx-node `b6a08a08` (incl. the compile_fuse quadratic fix), genjax 1.0.13,
jax 0.7.2/cuda13, driver 595.71.05. Same model/data/eps as linreg_mala;
N independent joint-MALA chains — GenJAX as `jit(vmap(seed(chain)))` over
N keys, GenMLX as shape-based [N,D] batching. Sweep N ∈ {8,64,512,4096} ×
S ∈ {100,1000}, per-chain acceptance pooled, tails judged at S=1000.

**The eager baseline measured first** (`vectorized-mala`, the public API
until today): host-bound at ~4 syncs/step plus N scalar generates at init —
N=4096×S=100 measured **30.7 s vs JAX 1.6 ms (~19,500x)**; even N=8×S=100
was ~1000x. Acceptance agreed to ±0.01 everywhere — right algorithm,
wrong execution model. (Raw: `linreg_mala_manychain.genmlx.20260729-084714`.)

**The lever, same day** (`fused-vectorized-mala`, the bean's named move):
batched vgenerate init (one handler run for all N chains), [T,N,D] noise
pre-generation, and the whole sweep as ONE captured-replay graph with zero
host syncs — per-step graph size is N-independent, so the existing :mala
persist gates apply unchanged. Statistical-agreement + determinism pins in
`fused_vectorized_mala_test`. The row, fused:

| N | S | GenJAX | GenMLX eager | **GenMLX fused** | gap now | GJ warmup | GM warmup |
|---|---|---|---|---|---|---|---|
| 8 | 100 | 1.54 ms | 866 ms | **10.4 ms** | 6.8x | 1870 ms | 45.2 s* |
| 64 | 100 | 1.45 ms | 1101 ms | **9.8 ms** | 6.8x | 1825 ms | **843 ms** |
| 512 | 100 | 1.45 ms | 3698 ms | **11.4 ms** | 7.9x | 1747 ms | **913 ms** |
| 4096 | 100 | 1.57 ms | 30745 ms | **17.4 ms** | 11x | 1530 ms | **914 ms** |
| 8 | 1000 | 11.2 ms | 9687 ms | **57.3 ms** | 5.1x | 1869 ms | 23.1 s |
| 64 | 1000 | 11.1 ms | 10159 ms | **66.7 ms** | 6.0x | 1876 ms | 23.7 s |
| 512 | 1000 | 13.2 ms | 13966 ms | **75.5 ms** | 5.7x | 1653 ms | 34.0 s |
| 4096 | 1000 | 13.2 ms | 57247 ms | **142 ms** | 11x | 1551 ms | 31.4 s |

\* first-in-process cell: cold NVRTC kernel bill for the new [N,D] shapes.

Pooled acceptance 0.80–0.81 both sides at every cell; pooled slope-tails
1.97–2.01 vs the analytic 1.9917 at S=1000. (N=8 S=100's tail 2.46 is
8 chains × 50 samples of noise — the spec judges tails at S=1000 only.)

**Reading the row.** Both sides are near-N-independent to 512 chains —
vmap and [N,D] broadcasting amortize identically in shape. The steady gap
in this regime is **5–8x, rising to 11x at N=4096**: per-step kernel
COUNT (GenMLX's captured replay launches ~a dozen tiny kernels per MALA
step where XLA fuses the step body; at 4096×2 floats per kernel the
launch overhead still dominates) — the genmlx-lnzc step-fusion lever, now
with its cleanest datapoint. Warmup: at S=100 GenMLX now BEATS JAX's jit
(0.9 s vs 1.5–1.9 s, break-even 39–117 calls in GenMLX's favor); at
S=1000 the 23–34 s trace is the genmlx-geiw ~n^1.7 residual, unchanged
priority. The eager→fused delta (80–2000x) is the single largest
one-lever improvement in the parity suite so far.

### Post-family-scoring (batched tensor score — genmlx-yopl, 2026-07-29)

`fused-vectorized-mala` now rides a batched tensor-native score
(`make-batched-tensor-score-with-index`: one-hot matmul column extraction
— transpose+index does not differentiate — family lps broadcast
[N,1]×[G] → [N,G], keepdims sum; equivalence vs the batched-handler score
and its grads pinned to ~1e-7 rel in `family_score_test`). Census at
N=64 S=100: tape 9256 → **5736**. Isolated eval cost ([N,2] score+grad
pair, eager): 1.6–2.2 ms (GFI) → **0.35–0.38 ms** (tensor) at every N up
to 4096.

| N | S | GenJAX | pre | **post-family** | gap now | warmup pre → post |
|---|---|---|---|---|---|---|
| 8 | 100 | 1.54 ms | 10.4 ms | **10.2 ms** | 6.6x | 45.2 s* → 366 ms |
| 64 | 100 | 1.45 ms | 9.8 ms | **10.0 ms** | 6.9x | 843 ms → **154 ms** |
| 512 | 100 | 1.45 ms | 11.4 ms | **12.3 ms** | 8.5x | 913 ms → **141 ms** |
| 4096 | 100 | 1.57 ms | 17.4 ms | **26.2 ms** † | 17x | 914 ms → **143 ms** |
| 8 | 1000 | 11.2 ms | 57.3 ms | **43.7 ms** | 3.9x | 23.1 s → **2.8 s** |
| 64 | 1000 | 11.1 ms | 66.7 ms | **43.9 ms** | 4.0x | 23.7 s → **2.9 s** |
| 512 | 1000 | 13.2 ms | 75.5 ms | **55.8 ms** | 4.2x | 34.0 s → **3.1 s** |
| 4096 | 1000 | 13.2 ms | 142 ms | **89.9 ms** | 6.8x | 31.4 s → **3.1 s** |

Acceptance/tails identical to the pre-family row at every cell. Warmups
collapsed ~7-11x (the batched tensor score bypasses the batched-handler
trace): S=100 cells now warm up in ~140-370 ms vs jit's 1.5-1.9 s, and
the S=1000 cells' 23-34 s traces are down to ~3 s — still ~2x over jit
(genmlx-geiw's scan-primitive territory), no longer 15-20x.

† The one regression, REPRODUCED twice and isolated: at N=4096 S=100 the
median rose 17.4 → 26.2 ms (p10 16.1 → 19.5) while the same shape at
S=1000 improved 38% and the isolated per-eval cost improved 6x — the
extra cost is in the capture/replay layer for that (largest-N,
shortest-S) shape, not the score graph. Filed with the capture-sink bean
(genmlx-qkx6) to attribute alongside the input-staging work.

### Post-reduce-fusion + qkx6 (2026-07-29, same binary)

Census 5736 → 5036. S=1000: **42.8 / 42.7 / 56.2 / 90.3 ms**
(3.8x/3.8x/4.3x/6.8x); S=100 cells 11.6–18.1 ms with ±30% run-to-run
noise at these sizes — notably N=4096 S=100 came back to **18.1 ms**
(from the 25–26 ms family-scoring regression; the qkx6 staging relax
and/or the smaller fused tape absorbed it). Warmups unchanged (~140–360 ms
at S=100, ~2.8–3.1 s at S=1000 — the S=1000 trace remains ~1.7x over jit,
geiw territory).

## ndreg_is — bigger-model regime: importance sampling over the (D, M) grid

sm_120, measured 2026-07-29 (bean genmlx-1s7i, IS row). Pins: genmlx
`bf142ea` src tree (bench nd mode lands in this doc's commit), mlx-node
`6dc7b5ee`, genjax 1.0.13, jax 0.7.2/cuda13. D-dimensional regression
(w ~ iid N(0,100) [D], y ~ N(X·w, 1) [M]), ONE vector site each —
GenMLX uses iid-gaussian both sites (plain gaussian vector sites keep
per-component weights, the same broadcast convention genjax 1.0 has;
both runners sum, guarded per cell). Data frozen by generator script +
seed (`gen_ndreg_data.py`, numpy 20260729) with exact float64 conjugate
references; tensor files regenerable, gitignored. N=1000 particles.

| D | M | GenJAX | GenMLX | gap | GJ warmup | GM warmup |
|---|---|---|---|---|---|---|
| 10 | 1000 | 0.020 ms | 0.301 ms | 15x | 427 ms | **110 ms** |
| 100 | 1000 | 0.037 ms | 0.293 ms | 7.9x | 367 ms | **2.5 ms** |
| 1000 | 1000 | 0.097 ms | 0.892 ms | 9.2x | 485 ms | **1.2 ms** |
| 10 | 10000 | 0.159 ms | 0.293 ms | **1.8x** | 296 ms | **1.1 ms** |
| 100 | 10000 | 0.229 ms | 0.297 ms | **1.3x** | 404 ms | **2.7 ms** |
| 1000 | 10000 | 0.194 ms | 0.882 ms | 4.5x | 996 ms | **2.1 ms** |

### Post-analysis (genmlx-hhnc census + reduce fusion re-measure, 2026-07-29)

The D=1000 residual is attributed: the cell's tape is only 28 entries
(~13 launches — kernel count is NOT the cause here), the cost scales with
N×D (the [N,D] prior-site pipeline; M-independent, collapses to the floor
at N=100), and the slow piece is the threefry RandomBits kernel
(~30–45 GB/s effective, ~15x under elementwise bandwidth) plus
RandomBits/Reduce acting as fusion barriers (~6 passes over the [N,D]
buffers where XLA pays ~2). Post-reduce-fusion re-measure: D=1000 cells
0.90–0.94 ms — unchanged, as predicted (rng-bound). Routed:
genmlx-nznb (threefry throughput + bits→transform fusion). Floor-gating
note: GenJAX does these cells in 0.097–0.194 ms, so 1.2×GenJAX is BELOW
GenMLX's ~0.28 ms captured-call floor — the D=1000 cells (and the
currently-green D≤100 M=10000 cells, which sit AT the floor) belong to
the floor-budget regime (genmlx-pqb5), with nznb/7dm0 closing only the
over-floor part.

**The ecsi hypothesis, confirmed.** GenMLX's ~0.3 ms captured-call floor
is flat in D and M; as real per-particle GPU work grows into it the gap
collapses — **1.3–1.8x at M=10000 with D ≤ 100**, against the 95x of the
microbenchmark-sized linreg_is row. The "gap crosses under 2x" point the
bean asked for: M=10000, D ≤ 100. GenMLX wins warmup at EVERY cell
(vgenerate-compiled traces in 1–110 ms vs jit's 0.3–1.0 s), so total
wall favors GenMLX from call one everywhere. Residual: the D=1000 cells
sit at 0.88–0.89 ms (4.5–9.2x) — the [N,1000] site kernels lag XLA's;
same kernel-economics family as lnzc. logZ estimates agree between sides
in the same-order-of-badness sense the spec defines (prior-proposal IS
degrades exponentially in D, identically on both sides; exact float64
references reported per cell). MALA/HMC nd rows remain (per-D eps tuning
needed).
